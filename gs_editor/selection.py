"""Screen-space selection shared by native splats and LTS reference meshes."""
import math
from time import perf_counter
import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader
from . import reference, view
from .splats import SplatSet, SELECT_ATTR, BASE_ATTR, HIDDEN_ATTR, OPACITY_COLUMNS, _LAYOUT, is_splat_object

# Keep saved enum numbers stable; the retired circle selector used number 1.
TOOLS = [('BOX', '框选', '拖动矩形选择', 'NONE', 0),
         ('LASSO', '套索', '按住左键自由绘制闭合选区', 'NONE', 2),
         ('POLYGON', '多边形', '逐点单击，回车闭合，退格删除上一个顶点', 'NONE', 3),
         ('BRUSH', '笔刷', '连续选点，或切换涂抹与还原', 'NONE', 4)]
# Keep RESTORE at 3: saved COVER actions used the retired enum number 2.
BRUSH_ACTIONS = [('SELECT', '选点', '先刷出选区，再执行擦除', 'NONE', 0),
                 ('ERASE', '涂抹', '用强度和羽化控制直接擦除', 'NONE', 1),
                 ('RESTORE', '还原', '用笔刷恢复擦除区域', 'NONE', 3)]
HIDDEN = 'gse_selection_hidden'
NATIVE_SELECTION = '.selection'
NATIVE_TOOLS = {'BOX': 'builtin.select_box', 'BRUSH': 'builtin.select_circle', 'LASSO': 'builtin.select_lasso'}
TOOL_ICONS = {'BOX': 'ops.generic.select_box',
              'LASSO': 'ops.generic.select_lasso', 'POLYGON': 'ops.gpencil.draw.poly',
              'BRUSH': 'brush.draw'}
_active = None
_handle = None
_cache = {}
_projection_cache = None
_registered_tools = []
COMMIT_INTERVAL = 1 / 20


def finish_for_mode_switch(context):
    """Commit a valid gesture before the caller changes Blender's mode."""
    if _active is not None:
        return _active.complete_for_mode_switch(context)
    return {'FINISHED'}


def supported(obj):
    return reference.is_reference(obj) or (bpy.app.version >= (5, 3, 0) and is_splat_object(obj))


def count(obj):
    return len(obj.data.vertices) if reference.is_reference(obj) else len(obj.data.points)


def selection_mode(context):
    return context.mode == 'OBJECT' or (context.mode == 'EDIT_POINTCLOUD' and is_splat_object(context.active_object))


def read_mask(obj, name=SELECT_ATTR):
    native = name == SELECT_ATTR and is_splat_object(obj)
    if native:
        name = NATIVE_SELECTION
    attr = obj.data.attributes.get(name)
    # Blender treats a missing .selection attribute as all points selected.
    if attr is None:
        return np.full(count(obj), native, bool)
    result = np.empty(count(obj), np.float32 if attr.data_type == 'FLOAT' else bool)
    attr.data.foreach_get('value', result)
    return result != 0


def write_mask(obj, values, name=SELECT_ATTR):
    if name == SELECT_ATTR and is_splat_object(obj):
        name = NATIVE_SELECTION
    attr = obj.data.attributes.get(name) or obj.data.attributes.new(name, 'BOOLEAN', 'POINT')
    attr.data.foreach_set('value', np.ascontiguousarray(values, dtype=np.float32 if attr.data_type == 'FLOAT' else bool))
    obj.data.update_tag()
    if name in {SELECT_ATTR, NATIVE_SELECTION}:
        # Changing selection does not change projected positions or visibility.
        _cache.clear()
    else:
        clear_cache()
    if reference.is_reference(obj):
        reference.invalidate()


def cloud(obj):
    if reference.is_reference(obj):
        pos, _, rgba = reference.arrays(obj)
        return pos, (rgba[:, 3] > 1 / 255) & ~read_mask(obj, HIDDEN)
    n = count(obj)
    pos = np.empty((n, 3), np.float32)
    obj.data.attributes['position'].data.foreach_get('vector', pos.ravel())
    visible = ~read_mask(obj, HIDDEN_ATTR)
    base = obj.data.attributes.get(BASE_ATTR)
    if base is not None:
        width, key, dtype = _LAYOUT[base.data_type]
        rgba = np.empty((n, width), dtype)
        base.data.foreach_get(key, rgba.ravel())
        visible &= rgba[:, 3] > 1 / 255
    return pos, visible


def screen_points(pos, matrix, width, height):
    clip = pos @ matrix[:, :3].T + matrix[:, 3]
    valid = (clip[:, 3] > 1e-6) & (clip[:, 2] >= -clip[:, 3]) & (clip[:, 2] <= clip[:, 3])
    ndc = clip[:, :3] / np.maximum(clip[:, 3:4], 1e-6)
    xy = (ndc[:, :2] * .5 + .5) * (width, height)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return xy, valid


def surface_mask(xy, depth, valid, width, height, tolerance):
    """Front layer per 4px tile, across the whole cloud, not just the gesture.

    This is a point-centre depth approximation, not a scene mesh ray cast.
    """
    result = np.zeros(len(xy), bool)
    ids = np.flatnonzero(valid)
    columns, rows = int(math.ceil(width / 4)), int(math.ceil(height / 4))
    cells = (xy[ids, 1].astype(np.int64) // 4) * columns + xy[ids, 0].astype(np.int64) // 4
    front = np.full(columns * rows, np.inf)
    np.minimum.at(front, cells, depth[ids])
    result[ids] = depth[ids] <= front[cells] + tolerance
    return result


def polygon_mask(xy, vertices):
    result = np.zeros(len(xy), bool)
    if len(vertices) < 3:
        return result
    x, y = xy.T
    # Sorting is unnecessary for short contours or small candidate sets.
    if len(vertices) <= 12 or len(xy) < 256:
        a = vertices[-1]
        for b in vertices:
            if a[1] != b[1]:
                crossing = ((a[1] > y) != (b[1] > y))
                result ^= crossing & (x <= (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1]) + a[0])
            a = b
        return result

    # An edge can only cross points in its half-open vertical interval.
    # Sort once, then evaluate intersections on that contiguous slice instead
    # of every candidate for every sampled lasso segment. No vertex is removed.
    order = np.argsort(y)
    sorted_y, sorted_x = y[order], x[order]
    sorted_result = np.zeros(len(xy), bool)
    a = vertices[-1]
    for b in vertices:
        if a[1] != b[1]:
            # Match NumPy's scalar/array comparison promotion, including a
            # float32 candidate array with Python float contour coordinates.
            ay = np.asarray(a[1], dtype=np.result_type(y, a[1])).item()
            by = np.asarray(b[1], dtype=np.result_type(y, b[1])).item()
            start = np.searchsorted(sorted_y, min(ay, by), side='left')
            stop = np.searchsorted(sorted_y, max(ay, by), side='left')
            if start < stop:
                section = slice(start, stop)
                sorted_result[section] ^= (
                    sorted_x[section] <= (b[0]-a[0]) * (sorted_y[section]-a[1]) / (b[1]-a[1]) + a[0]
                )
        a = b
    result[order] = sorted_result
    return result


def shape_mask(xy, tool, points, radius):
    if tool == 'BOX':
        low, high = np.minimum(points[0], points[-1]), np.maximum(points[0], points[-1])
        return np.all((xy >= low) & (xy <= high), axis=1)
    if tool in {'LASSO', 'POLYGON'}:
        return polygon_mask(xy, points)
    end = np.asarray(points[-1])
    if len(points) == 1:
        return np.sum((xy-end)**2, axis=1) <= radius**2
    start = np.asarray(points[-2])
    segment = end-start
    t = np.clip(((xy-start) @ segment) / max(float(segment @ segment), 1e-12), 0, 1)
    return np.sum((xy-start-t[:, None]*segment)**2, axis=1) <= radius**2


class ScreenIndex:
    """Exact selection on local candidates; no point sampling.

    Build 64px bins only when touched, without sorting the whole cloud first.
    Large jumps and single-shot tools use an exact bounding-box filter.
    """
    def __init__(self, xy, eligible, width, height, moving=True):
        self.xy = xy
        self.ids = np.flatnonzero(eligible)
        self.cells = None
        self.bins = {}
        if moving:
            self.columns = int(math.ceil(width / 64))
            self.rows = int(math.ceil(height / 64))
            cells = (xy[self.ids, 1].astype(np.uint32) // 64) * self.columns + xy[self.ids, 0].astype(np.uint32) // 64
            self.cells = cells.astype(np.min_scalar_type(self.columns*self.rows-1), copy=False)

    def hit_ids(self, tool, points, radius):
        if tool in {'LASSO', 'POLYGON'} and len(points) < 3:
            return self.ids[:0]
        bounds = np.asarray(points[-2:] if tool == 'BRUSH' else points, dtype=float)
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        if tool == 'BRUSH':
            low, high = low-radius, high+radius
        ids = self.ids
        if self.cells is not None:
            x0, y0 = np.maximum(np.floor(low/64), 0).astype(int)
            x1, y1 = np.minimum(np.floor(high/64), (self.columns-1, self.rows-1)).astype(int)
            if x0 > x1 or y0 > y1:
                return ids[:0]
            # Do not scan every bin along a fast, long mouse jump. A single
            # bounding-box pass is cheaper in that case and keeps the full path.
            if (x1-x0+1)*(y1-y0+1) <= 16:
                spans = []
                for y in range(y0, y1+1):
                    for x in range(x0, x1+1):
                        cell = y*self.columns+x
                        if cell not in self.bins:
                            self.bins[cell] = self.ids[self.cells == cell]
                        spans.append(self.bins[cell])
                ids = spans[0] if len(spans) == 1 else np.concatenate(spans)
        xy = self.xy[ids]
        inside = (xy[:, 0] >= low[0]) & (xy[:, 0] <= high[0]) & (xy[:, 1] >= low[1]) & (xy[:, 1] <= high[1])
        ids, xy = ids[inside], xy[inside]
        return ids if tool == 'BOX' else ids[shape_mask(xy, tool, points, radius)]


def projection_index(obj, rv, region, depth_mode, tolerance):
    """Reuse one view's exact index, validating even edits made outside GS Editor."""
    global _projection_cache
    matrix = np.array(rv.perspective_matrix @ obj.matrix_world)
    view_matrix = np.array(rv.view_matrix @ obj.matrix_world) if depth_mode == 'SURFACE' else None
    key = (obj.as_pointer(), obj.data.as_pointer(), obj.mode,
           region.width, region.height, depth_mode,
           tolerance if view_matrix is not None else None, matrix.tobytes(),
           view_matrix.tobytes() if view_matrix is not None else None)
    # Native transforms and attribute edits do not bump gsp_revision. Comparing
    # fresh arrays is cheap relative to projecting and depth-filtering millions
    # of points, and avoids reusing stale coordinates after native edits.
    pos, visible = cloud(obj)
    cached = _projection_cache
    if (cached is not None and cached[0] == key
            and np.array_equal(pos, cached[1]) and np.array_equal(visible, cached[2])):
        return cached[3]
    _projection_cache = cached = None
    xy, valid = screen_points(pos, matrix, region.width, region.height)
    valid &= visible
    if view_matrix is not None:
        depth = -(pos @ view_matrix[2, :3] + view_matrix[2, 3])
        valid = surface_mask(xy, depth, valid, region.width, region.height, tolerance)
    # One index works for all four tools; changing shape keeps the warm view.
    index = ScreenIndex(xy, valid, region.width, region.height, moving=True)
    # Reference arrays are owned by the viewer; keep an independent snapshot.
    _projection_cache = (key, pos.copy() if reference.is_reference(obj) else pos, visible, index)
    return index


class SelectionStroke:
    """Keep the full selection in memory; change only hit indices per event."""
    def __init__(self, original, operation):
        self.original = original
        self.operation = operation
        self.result = np.zeros_like(original) if operation == 'REPLACE' else original.copy()
        self.dirty = operation == 'REPLACE' and bool(original.any())
        self.previous = np.empty(0, np.int64)

    def apply(self, ids, accumulate):
        # Non-brush gestures replace their footprint instead of accumulating it.
        if not accumulate:
            if np.array_equal(ids, self.previous):
                return
            self.result[self.previous] = False if self.operation == 'REPLACE' else self.original[self.previous]
            self.dirty |= bool(len(self.previous))
            self.previous = ids
        value = self.operation != 'SUBTRACT'
        self.dirty |= bool(np.any(self.result[ids] != value))
        self.result[ids] = value


def combine(original, hits, operation):
    if operation == 'ADD':
        return original | hits
    if operation == 'SUBTRACT':
        return original & ~hits
    return hits.copy()


@persistent
def clear_cache(*_):
    global _projection_cache
    _cache.clear()
    _projection_cache = None


def draw_selection():
    context = bpy.context
    obj = context.active_object
    # Native point clouds use Blender's edit-mode highlight and .selection.
    if not reference.is_reference(obj) or context.mode != 'OBJECT' or not context.region_data:
        return
    if not obj.visible_get() or not context.space_data.overlay.show_overlays:
        return
    if reference.is_reference(obj) and not context.scene.gse.reference_show:
        return
    key = (obj.as_pointer(), obj.data.as_pointer(), count(obj), obj.data.get('gsp_revision', 0))
    if key not in _cache:
        _cache.clear()
        pos, visible = cloud(obj)
        mask = read_mask(obj) & visible
        _cache[key] = pos[mask]
    pos = _cache[key]
    if not len(pos):
        return
    # Drawing cap only; the stored selection always includes every selected point.
    pos = pos[::max(1, int(math.ceil(len(pos)/200000)))]
    xy, ok = screen_points(pos, np.array(context.region_data.perspective_matrix @ obj.matrix_world), context.region.width, context.region.height)
    if not ok.any():
        return
    blend = gpu.state.blend_get()
    try:
        gpu.state.blend_set('ALPHA')
        gpu.state.point_size_set(3)
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'POINTS', {'pos': xy[ok].astype(np.float32)})
        shader.bind()
        shader.uniform_float('color', (1, .5, .05, .85))
        batch.draw(shader)
    finally:
        gpu.state.point_size_set(1)
        gpu.state.blend_set(blend)


class GSE_OT_select(bpy.types.Operator):
    bl_idname = 'gse.select'
    bl_label = '开始选择'
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    shape: bpy.props.EnumProperty(items=TOOLS, default='BOX', options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        from .ops import STATE
        return context.area and context.area.type == 'VIEW_3D' and selection_mode(context) and supported(context.active_object) and _active is None and not any(STATE.values())

    def invoke(self, context, event):
        global _active
        self._finished = False
        if self.shape == 'BRUSH' and is_splat_object(context.active_object) and context.scene.gse.brush_action != 'SELECT':
            label = {'ERASE': '开始涂抹', 'RESTORE': '开始还原'}[context.scene.gse.brush_action]
            self.report({'INFO'}, '请在笔刷参数下点击“' + label + '”')
            return {'CANCELLED'}
        self.obj, self.area = context.active_object, context.area
        self.window_manager = context.window_manager
        self.workspace = context.workspace
        self.mode = context.mode
        if self.obj.data.users > 1 or self.obj.data.library:
            self.report({'ERROR'}, '请先将高斯数据设为本地单用户数据')
            return {'CANCELLED'}
        self.region = next(r for r in self.area.regions if r.type == 'WINDOW')
        self.rv = self.area.spaces.active.region_3d
        self.pos = self.visible = None
        self.mesh_pointer = self.obj.data.as_pointer()
        self.n = count(self.obj)
        self.points, self.mouse = [], (0, 0)
        self.dragging = False
        self.tool = self.shape
        self.timer = None
        self.preview_handle = None
        self.written = False
        self.next_commit = 0
        context.scene.gse.select_shape = self.shape
        self.start(context, event, (event.mouse_x-self.region.x, event.mouse_y-self.region.y))
        try:
            self.preview_handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_preview, (), 'WINDOW', 'POST_PIXEL')
            if self.tool == 'BRUSH':
                self.timer = context.window_manager.event_timer_add(COMMIT_INTERVAL, window=context.window)
            _active = self
            context.window_manager.modal_handler_add(self)
            self.mouse = self.points[0]
            if self.tool == 'BRUSH':
                self.apply(context)
        except (ReferenceError, RuntimeError, ValueError) as ex:
            self.cancel(context)
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        context.workspace.status_text_set('Shift 加选 · Ctrl 减选 · 滚轮 半径 · 多边形回车确认/退格回退 · Esc 取消 · Tab 返回物体模式')
        return {'RUNNING_MODAL'}

    def start(self, context, event, mouse):
        p = context.scene.gse
        self.operation = 'SUBTRACT' if event.ctrl else 'ADD' if event.shift else p.select_operation
        self.stroke = SelectionStroke(read_mask(self.obj), self.operation)
        self.index = projection_index(self.obj, self.rv, self.region, p.select_depth, p.select_thickness)
        self.points = [mouse]
        self.dragging = True

    def apply(self, context, force=False):
        ids = self.index.hit_ids(self.tool, self.points, context.scene.gse.select_radius)
        self.stroke.apply(ids, accumulate=self.tool == 'BRUSH')
        self.flush(force=force)

    def flush(self, force=False):
        if self.stroke.dirty and (force or perf_counter() >= self.next_commit):
            write_mask(self.obj, self.stroke.result)
            self.written = True
            self.stroke.dirty = False
            self.next_commit = perf_counter() + COMMIT_INTERVAL
            self.area.tag_redraw()

    def finish(self, context):
        global _active
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        self._finished = True
        if self.timer is not None:
            try:
                self.window_manager.event_timer_remove(self.timer)
            except (ReferenceError, RuntimeError, ValueError):
                pass
            self.timer = None
        if self.preview_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self.preview_handle, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass
            self.preview_handle = None
        if _active is self:
            _active = None
            try:
                self.workspace.status_text_set(None)
            except ReferenceError:
                pass
        try:
            self.area.tag_redraw()
        except ReferenceError:
            pass
        self.dragging = False
        self.pos = self.visible = self.index = self.stroke = None
        self.points = []
        return {'FINISHED'}

    def complete_for_mode_switch(self, context):
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        try:
            if self.tool == 'POLYGON':
                # A polygon is not committed until Enter closes it. Leaving
                # edit mode cancels that unfinished gesture, not prior choices.
                if self.written:
                    write_mask(self.obj, self.stroke.original)
            elif self.tool == 'BRUSH':
                self.flush(force=True)
            elif self.tool == 'BOX' and len(self.points) >= 2:
                if np.all(np.asarray(self.points[0]) != self.points[-1]):
                    self.apply(context, force=True)
            elif self.tool == 'LASSO' and len(self.points) >= 3:
                self.apply(context, force=True)
        finally:
            self.finish(context)
        return {'FINISHED'}

    def cancel(self, context):
        if getattr(self, '_finished', False):
            return
        try:
            if self.written:
                write_mask(self.obj, self.stroke.original)
        finally:
            self.finish(context)

    def modal(self, context, event):
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        try:
            if event.type == 'TAB':
                if (event.value == 'PRESS' and context.mode == 'EDIT_POINTCLOUD'
                        and not (event.ctrl or event.alt or event.shift or event.oskey)):
                    result = bpy.ops.gse.set_mode(mode='OBJECT')
                    return {'FINISHED'} if self._finished or 'FINISHED' in result else {'PASS_THROUGH'}
                return {'PASS_THROUGH'}
            if context.active_object != self.obj or self.obj.data.as_pointer() != self.mesh_pointer or count(self.obj) != self.n or context.mode != self.mode or self.area.type != 'VIEW_3D':
                if (context.active_object == self.obj and self.obj.data.as_pointer() == self.mesh_pointer
                        and count(self.obj) == self.n and context.mode != self.mode):
                    return self.complete_for_mode_switch(context)
                return self.finish(context)
            return self.handle(context, event)
        except (ReferenceError, RuntimeError, ValueError) as ex:
            self.report({'ERROR'}, str(ex))
            return self.finish(context)

    def handle(self, context, event):
        if event.type == 'TIMER':
            if self.timer is not None:
                self.flush()
            return {'PASS_THROUGH'}
        p = context.scene.gse
        mouse = (event.mouse_x-self.region.x, event.mouse_y-self.region.y)
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            self.cancel(context)
            return {'CANCELLED'}
        from .ops import _in_ui_region
        inside = (0 <= mouse[0] < self.region.width and 0 <= mouse[1] < self.region.height
                  and not _in_ui_region(self.area, event.mouse_x, event.mouse_y))
        if not inside:
            # Release outside the viewport finishes the last valid footprint;
            # sidebar coordinates must never be appended to the gesture.
            if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.dragging and self.tool != 'POLYGON':
                return self.complete_for_mode_switch(context)
            if not self.dragging or self.tool == 'POLYGON':
                return {'PASS_THROUGH'}
            return {'RUNNING_MODAL'}
        self.mouse = mouse
        self.area.tag_redraw()
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if self.tool == 'POLYGON' and self.dragging:
                if len(self.points) >= 3:
                    self.apply(context, force=True)
                    return self.finish(context)
                return {'RUNNING_MODAL'}
            if self.tool == 'BOX':
                self.points = [self.points[0], mouse]
            elif self.tool in {'BRUSH', 'LASSO'}:
                self.points.append(mouse)
            self.apply(context, force=True)
            return self.finish(context)
        if event.type == 'BACK_SPACE' and event.value == 'PRESS' and self.tool == 'POLYGON' and self.dragging:
            if len(self.points) > 1:
                self.points.pop()
            return {'RUNNING_MODAL'}
        if event.type == 'X' and event.value == 'PRESS' and not self.dragging:
            p.select_depth = 'SURFACE' if p.select_depth == 'THROUGH' else 'THROUGH'
            return {'RUNNING_MODAL'}
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and self.tool == 'BRUSH':
            p.select_radius = max(1, min(1000, p.select_radius + (5 if event.type == 'WHEELUPMOUSE' else -5)))
            self.points = [mouse]
            self.apply(context)
            return {'RUNNING_MODAL'}
        if not self.dragging and event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'NDOF_MOTION'}:
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and inside:
            if not self.dragging:
                self.start(context, event, mouse)
            elif self.tool == 'POLYGON':
                self.points.append(mouse)
            if self.tool == 'BRUSH':
                self.apply(context)
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE' and self.dragging and self.tool != 'POLYGON':
            if self.tool == 'BOX':
                self.points = [self.points[0], mouse]
            elif self.tool == 'BRUSH':
                self.points.append(mouse)
            elif self.tool == 'LASSO' and np.linalg.norm(np.array(mouse)-self.points[-1]) >= 2:
                self.points.append(mouse)
            if self.tool == 'BRUSH':
                self.apply(context)
                self.points = self.points[-1:]
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.dragging and self.tool != 'POLYGON':
            if self.tool == 'BOX':
                self.points = [self.points[0], mouse]
            elif self.tool in {'BRUSH', 'LASSO'}:
                self.points.append(mouse)
            self.apply(context, force=True)
            return self.finish(context)
        return {'RUNNING_MODAL'}

    def draw_preview(self):
        if getattr(self, '_finished', False):
            return
        if bpy.context.area != self.area or bpy.context.region != self.region:
            return
        points = list(self.points)
        if self.tool == 'BRUSH':
            radius = bpy.context.scene.gse.select_radius
            points = [(self.mouse[0]+radius*math.cos(a), self.mouse[1]+radius*math.sin(a)) for a in np.linspace(0, 2*math.pi, 65)]
        elif self.tool == 'BOX' and len(points) >= 2:
            a, b = points[0], points[-1]
            points = [a, (b[0], a[1]), b, (a[0], b[1]), a]
        elif self.tool == 'POLYGON' and points:
            points += [self.mouse, points[0]]
        elif self.tool == 'LASSO' and points:
            points += points[:1]
        if len(points) >= 2:
            view.draw_lines([[(x, y, 0) for x, y in points]], (1, .6, .1, 1), 2)


class GSE_OT_selection_action(bpy.types.Operator):
    bl_idname = 'gse.selection_action'
    bl_label = '操作高斯选区'
    bl_options = {'REGISTER', 'UNDO'}
    action: bpy.props.EnumProperty(items=[('CLEAR', '清空选区', ''), ('ERASE', '擦除选中', '')])

    @classmethod
    def description(cls, context, properties):
        return '清空当前选区，保留所有高斯点' if properties.action == 'CLEAR' else '隐藏选中的高斯点，可撤销或局部还原'

    @classmethod
    def poll(cls, context):
        from .ops import STATE
        return supported(context.active_object) and selection_mode(context) and _active is None and not any(STATE.values())

    def execute(self, context):
        obj = context.active_object
        if obj.data.users > 1 or obj.data.library:
            self.report({'ERROR'}, '请先将高斯数据设为本地单用户数据')
            return {'CANCELLED'}
        mask = read_mask(obj)
        if self.action == 'ERASE':
            mask &= cloud(obj)[1]
            if reference.is_reference(obj):
                write_mask(obj, read_mask(obj, HIDDEN) | mask, HIDDEN)
            else:
                splats = SplatSet.read(obj.data, only=OPACITY_COLUMNS + (SELECT_ATTR,))
                splats.hide(mask)
                splats.write(obj.data, only=OPACITY_COLUMNS + (SELECT_ATTR,))
                from .ops import bump_revision
                bump_revision(obj)
            self.report({'INFO'}, f'已擦除 {int(mask.sum()):,} 个选中点')
        write_mask(obj, np.zeros(len(mask), bool))
        reference.redraw(self, context)
        return {'FINISHED'}


def tool_id(shape, mode):
    suffix = '_reference' if mode == 'OBJECT' else ''
    return 'gse.select_' + shape.lower() + suffix


def active_tool(context):
    if not selection_mode(context) or (is_splat_object(context.active_object) and context.mode != 'EDIT_POINTCLOUD'):
        return None
    return context.workspace.tools.from_space_view3d_mode(context.mode, create=False)


def current_shape(context):
    tool = active_tool(context)
    if tool:
        if tool.idname in {'gse.select_circle', 'gse.select_circle_reference'}:
            return 'BRUSH'
        for shape, *_ in TOOLS:
            native = NATIVE_TOOLS.get(shape) if context.mode == 'EDIT_POINTCLOUD' else None
            if tool.idname in {tool_id(shape, context.mode), native}:
                return shape
    return None


def native_properties(context):
    if context.mode != 'EDIT_POINTCLOUD':
        return None
    tool = active_tool(context)
    if tool and tool.idname in NATIVE_TOOLS.values():
        return tool.operator_properties('view3d.' + tool.idname.split('.')[1])
    return None


def activate_shape(context, shape):
    region = next(r for r in context.area.regions if r.type == 'WINDOW')
    with context.temp_override(region=region):
        if is_splat_object(context.active_object) and context.mode == 'OBJECT':
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.wm.tool_set_by_id(name=tool_id(shape, context.mode))
    context.scene.gse.select_shape = shape
    context.area.tag_redraw()


class GSE_OT_selection_tool(bpy.types.Operator):
    bl_idname = 'gse.selection_tool'
    bl_label = '高斯选择工具'
    bl_description = '激活左侧工具栏中的高斯选择工具，与 Blender 原生点云选区共用数据'
    shape: bpy.props.EnumProperty(items=TOOLS)

    @classmethod
    def poll(cls, context):
        return GSE_OT_select.poll(context)

    @classmethod
    def description(cls, context, properties):
        return next(label + '：' + description for shape, label, description, *_ in TOOLS if shape == properties.shape)

    def execute(self, context):
        if context.active_object.data.users > 1 or context.active_object.data.library:
            self.report({'ERROR'}, '请先将高斯数据设为本地单用户数据')
            return {'CANCELLED'}
        if self.shape == 'BRUSH' and current_shape(context) == 'BRUSH':
            props = native_properties(context)
            if props is not None:
                p = context.scene.gse
                p.brush_action = 'SELECT'
                p.select_radius = props.radius
                p.select_depth = 'THROUGH'
                p.select_operation = {'SET': 'REPLACE', 'ADD': 'ADD', 'SUB': 'SUBTRACT'}.get(props.mode, 'REPLACE')
        activate_shape(context, self.shape)
        return {'FINISHED'}


class GSE_OT_brush_action(bpy.types.Operator):
    bl_idname = 'gse.brush_action'
    bl_label = '笔刷功能'
    action: bpy.props.EnumProperty(items=BRUSH_ACTIONS)

    @classmethod
    def poll(cls, context):
        return GSE_OT_selection_tool.poll(context) and is_splat_object(context.active_object)

    def execute(self, context):
        p = context.scene.gse
        props = native_properties(context)
        if props is not None and current_shape(context) == 'BRUSH':
            p.select_radius = props.radius
            p.select_depth = 'THROUGH'
            p.select_operation = {'SET': 'REPLACE', 'ADD': 'ADD', 'SUB': 'SUBTRACT'}.get(props.mode, 'REPLACE')
        p.brush_action = self.action
        if self.action != 'SELECT':
            p.tool = self.action
        activate_shape(context, 'BRUSH')
        return {'FINISHED'}


class GSE_OT_selection_depth(bpy.types.Operator):
    bl_idname = 'gse.selection_depth'
    bl_label = '选择深度'
    depth: bpy.props.EnumProperty(items=[('SURFACE', '表面', ''), ('THROUGH', '穿透', '')])

    @classmethod
    def poll(cls, context):
        return GSE_OT_selection_tool.poll(context)

    def execute(self, context):
        shape = current_shape(context)
        props = native_properties(context)
        if props is not None:
            context.scene.gse.select_operation = {'SET': 'REPLACE', 'ADD': 'ADD', 'SUB': 'SUBTRACT'}.get(props.mode, 'REPLACE')
            if shape == 'BRUSH':
                context.scene.gse.select_radius = props.radius
                context.scene.gse.brush_action = 'SELECT'
        context.scene.gse.select_depth = self.depth
        if shape:
            activate_shape(context, shape)
        return {'FINISHED'}


def parameter_row(layout, label):
    row = layout.row()
    row.use_property_split = False
    row.use_property_decorate = False
    split = row.split(factor=.30, align=True)
    name = split.column()
    name.alignment = 'RIGHT'
    name.label(text=label)
    return split.row(align=True)


def draw_settings(context, layout, tool=None):
    p = context.scene.gse
    shape = current_shape(context) or p.select_shape
    props = native_properties(context)
    if shape == 'BRUSH' and is_splat_object(context.active_object):
        action = 'SELECT' if props is not None else p.brush_action
        row = layout.row(align=True)
        row.scale_y = 1.1
        for ident, label, *_ in BRUSH_ACTIONS:
            row.operator('gse.brush_action', text=label, depress=(ident == action)).action = ident
        if action != 'SELECT':
            from .ui import draw_brush_params
            settings = layout.box()
            draw_brush_params(context, settings, action)
            row = layout.row()
            row.scale_y = 1.2
            label = '开始还原' if action == 'RESTORE' else '开始涂抹'
            row.operator('gse.brush', text=label, icon='BRUSH_DATA').tool = action
            return
    # Blender's native point-cloud tools always project through the cloud.
    depth = 'THROUGH' if props is not None else p.select_depth
    settings = layout.box()
    settings.use_property_split = True
    settings.use_property_decorate = False
    buttons = parameter_row(settings, '作用范围')
    for ident, label in (('SURFACE', '表面'), ('THROUGH', '穿透')):
        buttons.operator('gse.selection_depth', text=label, depress=(depth == ident)).depth = ident
    if depth == 'SURFACE':
        settings.prop(p, 'select_thickness')
    if shape == 'BRUSH':
        settings.prop(props, 'radius', text='笔刷半径') if props is not None else settings.prop(p, 'select_radius', text='笔刷半径')
    modes = parameter_row(settings, '选区操作')
    if props is not None:
        for value, label in (('SET', '替换'), ('ADD', '加选'), ('SUB', '减选')):
            modes.prop_enum(props, 'mode', value, text=label, icon='NONE')
    else:
        for value, label in (('REPLACE', '替换'), ('ADD', '加选'), ('SUBTRACT', '减选')):
            modes.prop_enum(p, 'select_operation', value, text=label, icon='NONE')


def draw_controls(layout, context):
    from .ops import STATE
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    controls = layout.column()
    controls.enabled = _active is None and not any(STATE.values())
    row = controls.grid_flow(row_major=True, columns=len(TOOLS), even_columns=True, even_rows=True, align=True)
    row.scale_y = 1.45
    row.scale_x = row.scale_y * 1.1
    current = current_shape(context)
    for shape, *_ in TOOLS:
        icon = ToolSelectPanelHelper._icon_value_from_icon_handle(TOOL_ICONS[shape])
        row.column(align=True).operator('gse.selection_tool', text='', icon_value=icon, depress=(shape == current)).shape = shape
    draw_settings(context, controls)
    painting = (is_splat_object(context.active_object) and (current or context.scene.gse.select_shape) == 'BRUSH'
                and native_properties(context) is None and context.scene.gse.brush_action != 'SELECT')
    if not painting:
        row = controls.row(align=True)
        row.scale_y = 1.2
        split = row.split(factor=.65, align=True)
        split.operator('gse.selection_action', text='擦除选中', icon='TRASH').action = 'ERASE'
        split.operator('gse.selection_action', text='清空选区').action = 'CLEAR'
    if is_splat_object(context.active_object) and (current or context.scene.gse.select_shape) != 'BRUSH':
        controls.operator('gse.brush_action', text='局部还原', icon='LOOP_BACK').action = 'RESTORE'


def workspace_tools():
    modes = ['OBJECT'] + (['EDIT_POINTCLOUD'] if bpy.app.version >= (5, 3, 0) else [])
    for mode in modes:
        for shape, label, description, *_ in TOOLS:
            yield type('GSE_WST_' + mode + '_' + shape, (bpy.types.WorkSpaceTool,), {
                '__module__': __name__, 'bl_space_type': 'VIEW_3D', 'bl_context_mode': mode,
                'bl_idname': tool_id(shape, mode), 'bl_label': '高斯' + label,
                'bl_description': description, 'bl_icon': TOOL_ICONS[shape],
                'bl_keymap': (('gse.select', {'type': 'LEFTMOUSE', 'value': 'PRESS', 'any': True},
                              {'properties': [('shape', shape)]}),),
                'draw_settings': staticmethod(draw_settings),
            })


@persistent
def migrate_retired_actions(*_):
    """Read raw ID properties before RNA resolves the removed enum value."""
    for scene in bpy.data.scenes:
        p = getattr(scene, 'gse', None)
        if p is None:
            continue
        if p.get('brush_action') in {2, 'COVER'}:
            p.brush_action = 'SELECT'
        if p.get('tool') in {2, 'COVER'}:
            p.tool = 'ERASE'


@persistent
def migrate_circle_selection(*_):
    for scene in bpy.data.scenes:
        if scene.gse.get('select_shape') in {1, 'CIRCLE'}:
            scene.gse.select_shape = 'BRUSH'
            scene.gse.brush_action = 'SELECT'
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is None:
                continue
            with bpy.context.temp_override(window=window, area=area, region=region):
                mode = bpy.context.mode
                if mode not in {'OBJECT', 'EDIT_POINTCLOUD'}:
                    continue
                tool = window.workspace.tools.from_space_view3d_mode(mode, create=False)
                if tool and tool.idname in {'gse.select_circle', 'gse.select_circle_reference'}:
                    window.scene.gse.brush_action = 'SELECT'
                    window.scene.gse.select_shape = 'BRUSH'
                    bpy.ops.wm.tool_set_by_id(name=tool_id('BRUSH', mode))


def register():
    global _handle
    # Blender restricts bpy.data during add-on enable. Register tools first;
    # scene migrations must wait until the registration context is released.
    for cls in workspace_tools():
        bpy.utils.register_tool(cls, after={'builtin.select_lasso'})
        _registered_tools.append(cls)
    if not bpy.app.background and _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(draw_selection, (), 'WINDOW', 'POST_PIXEL')
    for name in ('load_post', 'undo_post', 'redo_post'):
        getattr(bpy.app.handlers, name).append(clear_cache)
        getattr(bpy.app.handlers, name).append(migrate_retired_actions)
    bpy.app.handlers.load_post.append(migrate_circle_selection)
    bpy.app.timers.register(migrate_retired_actions, first_interval=0.0)
    bpy.app.timers.register(migrate_circle_selection, first_interval=0.0)


def unregister():
    global _handle
    for callback in (migrate_retired_actions, migrate_circle_selection):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    if migrate_circle_selection in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(migrate_circle_selection)
    if _active is not None:
        _active.cancel(bpy.context)
    for cls in reversed(_registered_tools):
        bpy.utils.unregister_tool(cls)
    _registered_tools.clear()
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for name in ('load_post', 'undo_post', 'redo_post'):
        handlers = getattr(bpy.app.handlers, name)
        for handler in (clear_cache, migrate_retired_actions):
            if handler in handlers:
                handlers.remove(handler)
    clear_cache()
