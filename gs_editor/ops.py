import time

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper

from . import core, merge, view, stash
from .splats import _TRACKING
from .splats import (SplatSet, is_splat_object, major_axes, concat_rows, set_from_rows,
                     HIDDEN_ATTR, OPACITY_COLUMNS)
from .ply_io import Y_UP, write_pointcloud_ply

TOOL_ITEMS = [
    ('ERASE', "擦除", "涂抹删除泼溅点：按强度和羽化渐进删点，松开再画可继续累积", 'TRASH', 0),
    ('RESTORE', "还原", "涂抹局部回退：恢复被擦除的泼溅点，移除新增的点", 'RECOVER_LAST', 1),
]

# 选择深度：高频切换，笔刷面板上直接并排两个按钮
DEPTH_ITEMS = [
    ('SURFACE', "表面", "只作用于鼠标命中处的球状区域"),
    ('THROUGH', "穿透", "作用于屏幕区域内所有深度的泼溅点"),
]

# 工具标识 -> 中文名，状态栏、按钮和撤销记录都用它
TOOL_NAMES = {item[0]: item[1] for item in TOOL_ITEMS}

_REV_KEY = "gsp_revision"

# Whether a brush modal is running (module state, so a crash never leaves a stale flag in the file).
STATE = {"running": False}
_active_brush = None


def finish_for_mode_switch(context):
    """Commit the in-progress brush stroke before changing Blender's mode."""
    brush = _active_brush
    if brush is not None and not getattr(brush, '_finished', False):
        try:
            if brush.stroke is not None:
                brush.end_stroke(context)
        finally:
            brush.finish(context)
    return {'FINISHED'}

PAINT_TOOLS = {'ERASE'}

# 涂抹时切换工具的按键：数字按面板顺序，字母为惯用键
TOOL_KEYS = {'ONE': 'ERASE', 'TWO': 'RESTORE', 'E': 'ERASE', 'R': 'RESTORE'}

# 涂抹（擦除 / 还原）时的按键
SHORTCUTS_BRUSH = [
    ("E  R", "擦除 / 还原（也可按 1、2）"),
    ("F", "调整笔刷大小"),
    ("Shift F", "调整羽化"),
    ("[   ]", "缩小 / 放大笔刷"),
    ("X", "切换 表面 / 穿透"),
    ("Esc", "结束笔刷"),
    ("Tab", "结束工具并返回物体模式（编辑模式下）"),
]

SHORTCUTS = SHORTCUTS_BRUSH
# Tools that can start and continue a stroke with nothing solid under the mouse.
FREE_TOOLS = PAINT_TOOLS | {'RESTORE'}

HELP = {
    'ERASE': "左键涂抹擦除",
    'RESTORE': "左键还原 · 绿色 = 编辑新增，红色 = 已擦除，黄色 = 已淡化",
}


def props_of(context):
    return context.scene.gse


def bump_revision(obj):
    rev = int(obj.data.get(_REV_KEY, 0)) + 1
    obj.data[_REV_KEY] = rev
    return rev


# Above this many splats, splat edits don't create Blender undo steps: each step would keep
# another full copy of the cloud in memory (about 1.2 GB per step at 5M splats, up to 32
# steps), which is what ran large scans out of memory. Restore is the undo for these.
UNDO_SPLAT_LIMIT = 1_000_000


def undo_push(obj, message):
    if obj is not None and len(obj.data.points) <= UNDO_SPLAT_LIMIT:
        bpy.ops.ed.undo_push(message=message)


def save(obj, S, only=None):
    """Write splats back (only the named columns if given); legacy removals go to the stash."""
    S.write(obj.data, only)
    stash.push(obj, S)


def commit(context, obj, S, message, only=None):
    """Write a brush stroke back. No undo step: the Restore brush is the undo, and one
    step is recorded when the brush ends so Ctrl+Z can't silently drop the session."""
    save(obj, S, only)
    rev = bump_revision(obj)
    obj.data.update_tag()
    for area in context.screen.areas if context.screen else ():
        if area.type == 'VIEW_3D':
            area.tag_redraw()
    return rev


def active_splats(context):
    obj = context.active_object
    return obj if is_splat_object(obj) else None


def _in_ui_region(area, x, y):
    for r in area.regions:
        if r.type in {'UI', 'TOOLS', 'HEADER', 'TOOL_HEADER', 'ASSET_SHELF', 'NAVIGATION_BAR'} \
                and r.x <= x < r.x + r.width and r.y <= y < r.y + r.height and r.width > 1:
            return True
    return False


# =====================================================================
class GSE_OT_set_tool(bpy.types.Operator):
    """切换当前工具：只换工具，不开始涂抹（涂抹要点下面的「开始涂抹」）"""
    bl_idname = "gse.set_tool"
    bl_label = "选择工具"

    tool: bpy.props.EnumProperty(items=TOOL_ITEMS, default='ERASE')

    def execute(self, context):
        props_of(context).tool = self.tool
        return {'FINISHED'}


class GSE_OT_set_depth(bpy.types.Operator):
    """切换选择深度：表面只动鼠标命中处，穿透动屏幕范围内的所有深度"""
    bl_idname = "gse.set_depth"
    bl_label = "选择深度"
    bl_description = "表面：只擦鼠标命中处的球状区域 · 穿透：擦屏幕范围内所有深度的泼溅点（涂抹时按 X 也能切）"

    mode: bpy.props.EnumProperty(items=DEPTH_ITEMS, default='SURFACE')

    def execute(self, context):
        props_of(context).depth_mode = self.mode
        return {'FINISHED'}


# =====================================================================
class GSE_OT_brush(bpy.types.Operator):
    """用当前工具在泼溅点上涂抹，按 Esc 或 Enter 结束"""
    bl_idname = "gse.brush"
    bl_label = "泼溅笔刷"
    bl_options = {'REGISTER'}

    tool: bpy.props.EnumProperty(items=TOOL_ITEMS + [('KEEP', "当前", "", 'NONE', 3)], default='KEEP')

    _handle = None

    @classmethod
    def poll(cls, context):
        return context.area and context.area.type == 'VIEW_3D' and active_splats(context)

    # ---------------------------------------------------------- setup
    def invoke(self, context, event):
        global _active_brush
        p = props_of(context)
        if self.tool != 'KEEP':
            p.tool = self.tool
        p.brush_action = p.tool
        if STATE["running"]:
            return {'CANCELLED'}  # the running brush picks up the new tool
        self._finished = False
        self.obj = context.active_object
        self.area = context.area
        self.workspace = context.workspace
        self.mode = context.mode
        self.region = next(r for r in context.area.regions if r.type == 'WINDOW')
        self.rv3d = context.area.spaces.active.region_3d
        self.load()
        self.proj = view.Projection()
        self.overlay = view.Overlay()
        self.stroke = None
        self.hit = None
        self.hit_normal = None
        # Set by update_hit once the mouse is over a surface; defaults so a stroke that
        # starts over empty space or in Through mode still works.
        self.radius = 0.0
        self.view_dir = np.array([0.0, 0.0, -1.0], np.float32)
        self.last_dab = None
        self.stroke_ctrl = False
        self.mouse = (0, 0)
        # The orange object outline would hide the selection overlay; restore it on exit.
        self.overlay_settings = context.space_data.overlay
        self.had_outline = self.overlay_settings.show_outline_selected
        self.overlay_settings.show_outline_selected = False
        # Splats look dark under Solid mode's studio lighting; show their real colours.
        shading = context.space_data.shading
        if shading.type == 'SOLID':
            shading.light = 'FLAT'
        self.resizing = None
        self.shown_tool = None
        self._handle = bpy.types.SpaceView3D.draw_handler_add(self.draw_overlay, (), 'WINDOW', 'POST_VIEW')
        self._handle_px = bpy.types.SpaceView3D.draw_handler_add(self.draw_overlay_px, (), 'WINDOW', 'POST_PIXEL')
        context.window_manager.modal_handler_add(self)
        _active_brush = self
        STATE["running"] = True
        self.edited = False
        self.status(context)
        return {'RUNNING_MODAL'}

    def load(self):
        self.S = core_S = SplatSet.read(self.obj.data)
        self.axes = major_axes(core_S)
        self.update_reach()
        if core_S.n:
            core_S.near(core_S.pos[0], 0.0)     # build the spatial grid now, not on the first hover
        self.T = stash.read(self.obj)       # erased originals, for the restore brush
        self.rev = int(self.obj.data.get(_REV_KEY, 0))
        self.ptr = self.obj.data.as_pointer()
        return core_S

    def update_reach(self):
        """Very long splats are few; they're always tested, everything else via the grid."""
        reach = np.linalg.norm(self.axes, axis=1)
        self.reach_cap = float(np.percentile(reach, 99)) if len(reach) else 0.0
        self.long_idx = np.nonzero(reach > self.reach_cap)[0]

    def refresh_axes(self):
        """Axes for splats added by the last stroke (rows are only ever appended during a session)."""
        n_old = len(self.axes)
        if self.S.n > n_old:
            tail = SplatSet(self.S.n - n_old, {k: [dt, a[n_old:]] for k, (dt, a) in self.S.arrays.items()})
            self.axes = np.concatenate([self.axes, major_axes(tail)])
            reach = np.linalg.norm(self.axes[n_old:], axis=1)
            self.long_idx = np.concatenate([self.long_idx, n_old + np.nonzero(reach > self.reach_cap)[0]])
        elif self.S.n < n_old:
            self.axes = major_axes(self.S)
            self.update_reach()

    def status(self, context):
        p = props_of(context)
        context.workspace.status_text_set(
            f"泼溅·{TOOL_NAMES[p.tool]}：{HELP[p.tool]} · E 擦除 / R 还原 · F 大小 · Shift+F 羽化 · "
            f"X 表面/穿透 · Esc 完成")

    def finish(self, context):
        global _active_brush
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        self._finished = True
        owns_session = _active_brush is self
        if owns_session:
            _active_brush = None
            STATE["running"] = False
            props_of(context).brush_action = props_of(context).tool
        if self._handle:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass
            self._handle = None
        if self._handle_px:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_px, 'WINDOW')
            except (ReferenceError, RuntimeError, ValueError):
                pass
            self._handle_px = None
        if getattr(self, "edited", False):
            try:
                undo_push(self.obj, "泼溅笔刷")
            except (ReferenceError, RuntimeError):
                pass
        try:
            self.overlay_settings.show_outline_selected = self.had_outline
        except ReferenceError:
            pass
        if owns_session:
            try:
                self.workspace.status_text_set(None)
            except ReferenceError:
                pass
        try:
            self.area.tag_redraw()
        except ReferenceError:
            pass
        self.resizing = None
        self.release()
        return {'FINISHED'}

    def release(self):
        """Drop the brush's working copy of the cloud. Blender keeps finished operator
        instances alive, so without this every brush session leaked gigabytes on big scans."""
        for name in ("S", "T", "axes", "long_idx", "proj", "overlay", "stroke",
                     "rs_remove", "rs_unfade", "rs_restore"):
            if name in self.__dict__:
                setattr(self, name, None)
        import gc
        gc.collect()

    # ---------------------------------------------------------- helpers
    def stale(self):
        d = self.obj.data
        return (d.as_pointer() != self.ptr or int(d.get(_REV_KEY, 0)) != self.rev
                or len(d.points) != self.S.n)

    def mouse_local(self, event):
        return event.mouse_x - self.region.x, event.mouse_y - self.region.y

    def update_hit(self, context, mx, my):
        p = props_of(context)
        mw = self.obj.matrix_world
        self.proj.update(self.region, self.rv3d, self.S, mw, self.rev, self.axes)
        hit = self.proj.pick(self.S, mx, my, p.radius_px)
        if hit is None:
            self.hit = None
            return
        self.view_dir = view.view_vector_local(self.region, self.rv3d, mx, my, mw)
        self.radius = view.pixel_radius_to_local(self.region, self.rv3d, mx, my, p.radius_px, hit, mw)
        self.hit = hit
        # Normal estimates are cheap enough to track live on moderate clouds.
        self.hit_anchor, self.hit_normal = core.surface_frame(self.S, hit, self.radius, self.view_dir)


    # ---------------------------------------------------------- dabs
    def dab(self, context, mx, my, ctrl):
        p = props_of(context)
        st = self.stroke
        S = self.S
        if p.tool == 'RESTORE':
            self.restore_dab(context, mx, my)
            self.last_dab = (mx, my)
            return
        # Through mode, or nothing solid under the mouse (wisps in empty space):
        # take everything under the brush circle at any depth.
        if p.depth_mode == 'THROUGH' or self.hit is None:
            idx, dpx = self.proj.disc(mx, my, p.radius_px, extents=True)
            w = core.feather_weight(dpx, p.radius_px, p.feather)
        else:
            idx, w = core.sphere_hits_S(S, self.hit, self.radius, p.feather, self.axes,
                                        self.reach_cap, self.long_idx)
        # Already-erased splats are hidden in place; leave them out, so repainting
        # a cleaned area doesn't mark (and preview) them again.
        live = S.visible[idx]
        idx, w = idx[live], w[live]
        st.erase(idx, w, p.strength)
        self.last_dab = (mx, my)

    # ---------------------------------------------------------- modal
    def modal(self, context, event):
        if getattr(self, '_finished', False):
            return {'FINISHED'}
        if event.type == 'TAB':
            if (event.value == 'PRESS' and context.mode == 'EDIT_POINTCLOUD'
                    and not (event.ctrl or event.alt or event.shift or event.oskey)):
                result = bpy.ops.gse.set_mode(mode='OBJECT')
                return {'FINISHED'} if self._finished or 'FINISHED' in result else {'PASS_THROUGH'}
            return {'PASS_THROUGH'}
        p = props_of(context)
        active = context.active_object
        if not STATE["running"] or active is None \
                or active.as_pointer() != self.obj.as_pointer() or not is_splat_object(self.obj):
            self.finish(context)
            return {'CANCELLED'}
        if context.mode != self.mode:
            return finish_for_mode_switch(context)
        if self.stroke is None and self.stale():
            self.load()
        if p.tool != self.shown_tool:
            self.shown_tool = p.tool
            self.status(context)
        self.area.tag_redraw()

        # Esc/Enter stop the brush wherever the mouse is (e.g. over the sidebar).
        if event.type in {'ESC', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS' and self.resizing is None:
            if self.stroke is not None:
                self.end_stroke(context)
            self.finish(context)
            return {'FINISHED'}

        x, y = event.mouse_x, event.mouse_y
        inside = (self.region.x <= x < self.region.x + self.region.width
                  and self.region.y <= y < self.region.y + self.region.height)
        if self.stroke is None and (not inside or _in_ui_region(self.area, x, y)):
            self.hit = None
            return {'PASS_THROUGH'}

        mx, my = self.mouse_local(event)
        self.mouse = (mx, my)

        if self.resizing is not None:
            return self.modal_resize(context, event, mx, my)

        if event.type == 'Z' and (event.ctrl or event.oskey) and event.value == 'PRESS':
            self.report({'INFO'}, "涂抹时无法撤销：请使用还原笔刷（按 2）")
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS' and self.stroke is None:
            plain = not (event.ctrl or event.alt or event.oskey)
            if event.type in TOOL_KEYS and plain and not event.shift:
                p.tool = TOOL_KEYS[event.type]
                self.status(context)
                return {'RUNNING_MODAL'}
            if event.type == 'F' and plain:
                self.start_resize(context, mx, my, 'FEATHER' if event.shift else 'RADIUS')
                return {'RUNNING_MODAL'}
            if event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'}:
                grow = 1.15 if event.type == 'RIGHT_BRACKET' else 1 / 1.15
                if event.shift:
                    p.feather = min(1.0, max(0.0, p.feather + (0.1 if grow > 1 else -0.1)))
                else:
                    p.radius_px = int(min(1000, max(3, round(p.radius_px * grow))))
                self.update_hit(context, mx, my)
                return {'RUNNING_MODAL'}
            if event.type == 'X':
                p.depth_mode = 'SURFACE' if p.depth_mode == 'THROUGH' else 'THROUGH'
                return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            self.update_hit(context, mx, my)
            if self.stroke is not None and (self.hit is not None or p.tool in FREE_TOOLS):
                spacing = max(p.radius_px * p.spacing, 1.0)
                if self.last_dab is None or np.hypot(mx - self.last_dab[0], my - self.last_dab[1]) >= spacing:
                    self.dab(context, mx, my, event.ctrl or event.alt)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                self.update_hit(context, mx, my)
                if self.hit is None and p.tool not in FREE_TOOLS:
                    return {'RUNNING_MODAL'}
                self.stroke = core.Stroke(self.S)
                self.rs_remove = np.zeros(self.S.n, bool)
                self.rs_unfade = np.zeros(self.S.n, bool)
                self.rs_restore = np.zeros(0 if self.T is None else self.T.n, bool)
                self.stroke_ctrl = event.ctrl or event.alt
                self.last_dab = None
                self.dab(context, mx, my, event.ctrl or event.alt)
                return {'RUNNING_MODAL'}
            if event.value == 'RELEASE' and self.stroke is not None:
                self.end_stroke(context)
                return {'RUNNING_MODAL'}

        if self.stroke is not None:
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

    def end_stroke(self, context):
        p = props_of(context)
        st, S = self.stroke, self.S
        self.stroke = None
        try:
            if p.tool == 'RESTORE':
                self.end_restore(context)
            else:
                faded_before = int(S.faded.sum())
                removed, added = st.commit_clones_and_erase(False)
                if removed or added or int(S.faded.sum()) != faded_before:
                    self.rev = commit(context, self.obj, S, f"泼溅{TOOL_NAMES[p.tool]}",
                                      only=None if added else OPACITY_COLUMNS)
        except Exception as ex:  # keep the modal alive and the data consistent
            self.report({'ERROR'}, f"笔画处理失败：{ex}")
            self.load()
            raise
        self.ptr = self.obj.data.as_pointer()
        self.refresh_axes()
        self.T = stash.read(self.obj)
        self.proj.key = None
        self.edited = True

    # ---------------------------------------------------------- shortcuts
    def start_resize(self, context, mx, my, kind):
        """Blender-style F resize: the circle stays put and follows the mouse's distance."""
        p = props_of(context)
        r = float(p.radius_px)
        if kind == 'RADIUS':
            centre = (mx - r, my)
        else:
            centre = (mx - r * (1.0 - p.feather), my)
        self.resizing = dict(kind=kind, centre=centre, radius=p.radius_px, feather=p.feather)
        context.workspace.status_text_set(
            f"移动鼠标设置笔刷{'大小' if kind == 'RADIUS' else '羽化'} · 点击或按 F 确认 · "
            f"Esc 或右键取消")

    def modal_resize(self, context, event, mx, my):
        p = props_of(context)
        rs = self.resizing
        d = float(np.hypot(mx - rs["centre"][0], my - rs["centre"][1]))
        if event.type == 'MOUSEMOVE':
            if rs["kind"] == 'RADIUS':
                p.radius_px = int(min(1000, max(3, round(d))))
            else:
                p.feather = float(min(1.0, max(0.0, 1.0 - d / max(p.radius_px, 1))))
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type in {'LEFTMOUSE', 'F', 'RET', 'NUMPAD_ENTER', 'SPACE'}:
            self.resizing = None
            self.status(context)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.type in {'RIGHTMOUSE', 'ESC'}:
            p.radius_px, p.feather = rs["radius"], rs["feather"]
            self.resizing = None
            self.status(context)
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


    def restore_dab(self, context, mx, my):
        """Mark added splats to remove and erased/faded splats to bring back, under the brush."""
        p = props_of(context)
        S, T, r = self.S, self.T, float(p.radius_px)
        surface = p.depth_mode == 'SURFACE' and self.hit is not None
        if surface:
            idx, w = core.sphere_hits_S(S, self.hit, self.radius, p.feather)
        else:
            idx, d = self.proj.disc(mx, my, r)
            w = core.feather_weight(d, r, p.feather)
        idx = idx[w > 0.5]
        if p.remove_added:
            self.rs_remove[idx[S.added[idx] & S.visible[idx]]] = True
        if not p.restore_erased:
            return
        self.rs_unfade[idx[S.faded[idx] & ~S.added[idx]]] = True
        # Erased splats have no surface to hit, so pick them on screen, but not ones
        # hidden behind the surface the brush is on.
        V = np.array(self.rv3d.view_matrix @ self.obj.matrix_world, np.float32)
        hdepth = -(self.hit @ V[2, :3] + V[2, 3]) if surface else np.inf
        hidx, d = self.proj.disc(mx, my, r)
        hidx = hidx[(core.feather_weight(d, r, p.feather) > 0.5)]
        hidx = hidx[S.hidden[hidx] & ~S.added[hidx] & (self.proj.depth[hidx] <= hdepth + self.radius)]
        self.rs_unfade[hidx] = True
        if T is not None and T.n:     # splats a 0.1.3 session moved to the stash
            sx, sy, ok = self.proj.project(T.pos)
            m = ok & (core.feather_weight(np.hypot(sx - mx, sy - my), r, p.feather) > 0.5)
            if surface:
                m &= -(T.pos @ V[2, :3] + V[2, 3]) <= hdepth + self.radius
            self.rs_restore |= m

    def end_restore(self, context):
        back, unfaded, removed = core.apply_restore(self.S, self.T, self.rs_remove, self.rs_unfade,
                                                    self.rs_restore)
        if not (back or unfaded or removed):
            return
        if self.T is not None and self.rs_restore.any():
            stash.write(self.obj, self.T)
        self.report({'INFO'}, f"已还原 {back} 个擦除点、{unfaded} 个淡化点；移除 {removed} 个新增点")
        self.rev = commit(context, self.obj, self.S, "泼溅还原", only=OPACITY_COLUMNS)

    # ---------------------------------------------------------- drawing
    def draw_overlay(self):
        try:
            self._draw()
        except ReferenceError:
            pass

    def _draw(self):
        if getattr(self, '_finished', False):
            return
        area = bpy.context.area
        if area is None or area.as_pointer() != self.area.as_pointer():
            return
        p = bpy.context.scene.gse
        ov = self.overlay
        ov.clear()
        ov.matrix = self.obj.matrix_world
        S = self.S
        st = self.stroke
        tool = p.tool
        if tool not in ('ERASE', 'RESTORE'):     # 工具切到覆盖时，等 modal 收尾，别画
            return

        if tool == 'RESTORE':
            self.draw_restore(ov, p, st)

        if st is not None and tool == 'ERASE':
            idx = np.nonzero((st.factor < 0.999) & S.visible)[0]
            if len(idx):
                a = 1.0 - st.factor[idx]
                rgba = np.column_stack([np.ones_like(a), 0.15 * np.ones_like(a),
                                        0.15 * np.ones_like(a), 0.35 + 0.6 * a]).astype(np.float32)
                pos, rgba = view.cap(S.pos[idx], rgba)
                ov.points.append((pos, rgba, 2.0))

        if self.hit is not None:
            R = self.radius
            n = self.hit_normal
            col = {'ERASE': (1.0, 0.3, 0.3), 'RESTORE': (1.0, 1.0, 1.0)}[tool]
            if p.depth_mode == 'THROUGH' and tool in {'ERASE', 'RESTORE'}:
                n = -self.view_dir
            ov.rings.append((self.hit, n, R, (*col, 1.0), 2.0))
            if p.feather > 0:
                ov.rings.append((self.hit, n, R * (1 - p.feather), (*col, 0.45), 1.5))
        ov.draw()

    def draw_restore(self, ov, p, st):
        S, T = self.S, self.T
        stroking = st is not None
        none = np.zeros(S.n, bool)
        layers = []
        if p.restore_erased:
            mark = self.rs_unfade if stroking else none
            erased = S.hidden & ~S.added
            faded = S.faded & ~S.hidden
            layers += [(S.pos[erased & ~mark], (1.0, 0.25, 0.25, 0.85)),
                       (S.pos[faded & ~mark], (1.0, 0.8, 0.2, 0.85)),
                       (S.pos[(erased | faded) & mark], (1.0, 1.0, 1.0, 0.95))]
            if T is not None and T.n:
                tmark = self.rs_restore if stroking else np.zeros(T.n, bool)
                layers += [(T.pos[~tmark], (1.0, 0.25, 0.25, 0.85)), (T.pos[tmark], (1.0, 1.0, 1.0, 0.95))]
        if p.remove_added:
            mark = self.rs_remove if stroking else none
            added = S.added & ~S.hidden
            layers += [(S.pos[added & ~mark], (0.2, 1.0, 0.4, 0.85)),
                       (S.pos[added & mark], (0.3, 0.3, 0.3, 0.9))]
        for pos, col in layers:
            if len(pos):
                pos, rgba = view.cap(pos, view.solid_rgba(len(pos), col))
                ov.points.append((pos, rgba, 2.0))

    def draw_overlay_px(self):
        # Screen circle when nothing solid is under the mouse, so the brush is
        # still visible (and usable) over wisps in empty space.
        if getattr(self, '_finished', False):
            return
        try:
            area = bpy.context.area
            if area is None or area.as_pointer() != self.area.as_pointer():
                return
            p = bpy.context.scene.gse
            if self.resizing is not None:
                cx, cy = self.resizing["centre"]
                view.draw_screen_ring(cx, cy, p.radius_px, (1.0, 1.0, 1.0, 0.95), 2.0)
                if p.feather > 0:
                    view.draw_screen_ring(cx, cy, p.radius_px * (1 - p.feather), (1.0, 1.0, 1.0, 0.45))
                return
            if self.hit is not None or p.tool not in PAINT_TOOLS | {'RESTORE'}:
                return
            view.draw_screen_ring(self.mouse[0], self.mouse[1], p.radius_px, (1.0, 1.0, 1.0, 0.7))
            if p.feather > 0:
                view.draw_screen_ring(self.mouse[0], self.mouse[1], p.radius_px * (1 - p.feather),
                                      (1.0, 1.0, 1.0, 0.3))
        except ReferenceError:
            pass


# =====================================================================


def export_yup(context):
    """导出的坐标系：YUP = glTF / 网页端的 Y-up，ZUP = Blender 自己的 Z-up。"""
    props = getattr(context.scene, "gse", None)
    return props is None or props.export_up_axis != 'ZUP'


class GSE_OT_export_ply(bpy.types.Operator, ExportHelper):
    """把当前泼溅对象保存为标准 3D 高斯泼溅 PLY（带上物体的位置、旋转和缩放）"""
    bl_idname = "gse.export_ply"
    bl_label = "导出泼溅 PLY"
    filename_ext = ".ply"
    filter_glob: bpy.props.StringProperty(default="*.ply", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return active_splats(context) is not None

    def invoke(self, context, event):
        obj = active_splats(context)
        if not self.filepath:
            self.filepath = bpy.path.clean_name(obj.name) + "_patched.ply"
        # The brush keeps its own copy of the whole cloud; stop it so that memory is freed
        # before writing (on multi-million-splat scans that copy is gigabytes).
        STATE["running"] = False
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        obj = active_splats(context)
        STATE["running"] = False
        # A PLY holds no scene graph, so the object's own transform is baked into the
        # point coordinates: the file shows the splats where they are seen, not where
        # they sit in object space.
        yup = export_yup(context)
        M = np.array(obj.matrix_world, np.float64)
        if yup:                               # Blender Z-up -> exported Y-up coordinates
            M = Y_UP @ M
        n = write_pointcloud_ply(self.filepath, obj.data, matrix=M)
        msg = f"已写入 {n} 个泼溅点到 {self.filepath}（{'Y-up' if yup else 'Z-up'}）"
        notes = merge.transform_notes(M)
        if notes:
            self.report({'WARNING'}, msg + "。以下为近似处理：" + "；".join(notes))
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}

def _selected_splats(context):
    return [o for o in context.selected_objects if is_splat_object(o)]


class GSE_OT_merge_splats(bpy.types.Operator):
    """把选中的泼溅对象按当前位置合并成一个模型（以活动物体的空间为准）：
    每个泼溅点的位置、旋转、尺寸和视角相关颜色都会带过去，已擦除的点和还原记录
    一并保留，之后可以作为单个 PLY 导出"""
    bl_idname = "gse.merge_splats"
    bl_label = "合并泼溅对象"
    bl_options = {'REGISTER'}

    originals: bpy.props.EnumProperty(
        name="原对象", default='HIDE',
        items=[('HIDE', "隐藏", "保留原对象，但在视图和渲染中隐藏"),
               ('KEEP', "保持可见", "原对象保持原样"),
               ('DELETE', "删除", "删除原对象")])

    @classmethod
    def poll(cls, context):
        return len(_selected_splats(context)) >= 2

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        STATE["running"] = False               # the brush's copy of a cloud is freed first
        objs = _selected_splats(context)
        target = active_splats(context)
        if target not in objs:
            target = objs[0]
        to_target = np.linalg.inv(np.array(target.matrix_world, np.float64))
        parts, stashed, notes = [], [], []
        total = 0
        for ob in objs:
            M = to_target @ np.array(ob.matrix_world, np.float64)
            S = SplatSet.read(ob.data)
            rows, why = merge.transform_rows(S.arrays, M)
            del S
            parts.append(rows)
            total += len(rows["position"][1])
            notes += [f"{ob.name}: {w}" for w in why]
            T = stash.read(ob)
            if T is not None:
                stashed.append(merge.transform_rows(T.arrays, M)[0])
        merged = set_from_rows(concat_rows(parts))
        parts.clear()
        pc = bpy.data.pointclouds.new(target.data.name + "_merged")
        merged.write(pc)
        for m in target.data.materials:
            pc.materials.append(m)
        new = target.copy()                     # its transform, modifiers and display settings
        new.data = pc
        new.name = target.name + "_merged"
        for coll in target.users_collection:
            coll.objects.link(new)
        if stashed:
            stash.write(new, set_from_rows(concat_rows(stashed)))
        for ob in objs:
            if self.originals == 'DELETE':
                bpy.data.objects.remove(ob)
            elif self.originals == 'HIDE':
                ob.hide_set(True)
                ob.hide_render = True
        for ob in context.view_layer.objects:
            ob.select_set(False)
        new.select_set(True)
        context.view_layer.objects.active = new
        undo_push(new, self.bl_label)
        msg = f"已把 {len(objs)} 个对象合并为“{new.name}”：{total:,} 个泼溅点"
        if notes:
            msg += "。以下为近似处理：" + "；".join(notes)
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def _edit_counts(obj):
    """(erased splats still in the model, erased originals in the stash, tracking columns)."""
    pc = obj.data
    hidden = 0
    attr = pc.attributes.get(HIDDEN_ATTR)
    if attr is not None and len(pc.points):
        buf = np.empty(len(pc.points), bool)
        attr.data.foreach_get("value", buf)
        hidden = int(buf.sum())
    st = stash.get(obj)
    tracked = [n for n in _TRACKING if pc.attributes.get(n) is not None]
    return hidden, (len(st.points) if st is not None else 0), tracked


class GSE_OT_bake_edits(bpy.types.Operator):
    """把编辑变成最终结果：删除已擦除的泼溅点，丢弃为还原笔刷保留的原始点，去掉
    编辑用的记录数据，让模型在内存和 .blend 里只占用可见的泼溅点。
    烘焙之后，还原笔刷无法再恢复烘焙之前擦除的内容"""
    bl_idname = "gse.bake_edits"
    bl_label = "烘焙编辑"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return active_splats(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, confirm_text="烘焙")

    def draw(self, context):
        hidden, stashed, tracked = _edit_counts(active_splats(context))
        col = self.layout.column()
        col.label(text=f"将删除 {hidden:,} 个已擦除的泼溅点。")
        col.label(text=f"将丢弃 {stashed:,} 个存档的原始泼溅点。")
        col.label(text="烘焙之前的编辑无法再用还原笔刷撤回。", icon='ERROR')

    def execute(self, context):
        STATE["running"] = False               # the brush's working copy goes too
        obj = active_splats(context)
        pc = obj.data
        hidden, stashed, _ = _edit_counts(obj)
        S = SplatSet.read(pc)
        keep = ~S.hidden
        rows = {k: [dt, a[keep]] for k, (dt, a) in S.arrays.items() if k not in _TRACKING}
        n = int(keep.sum())
        del S
        if hidden:
            pc.resize(n)
        for name in list(_TRACKING):
            attr = pc.attributes.get(name)
            if attr is not None:
                pc.attributes.remove(attr)
        if hidden:
            SplatSet(n, rows).write(pc)
        del rows
        st = stash.get(obj)
        if st is not None:
            del pc[stash.STASH_KEY]
            st.use_fake_user = False
            if st.users == 0:
                bpy.data.pointclouds.remove(st)
        bump_revision(obj)
        pc.update_tag()
        undo_push(obj, self.bl_label)
        self.report({'INFO'}, f"已烘焙“{obj.name}”：{n:,} 个泼溅点；删除 {hidden:,} 个擦除点，"
                              f"丢弃 {stashed:,} 个存档点")
        return {'FINISHED'}


class GSE_OT_stop_brush(bpy.types.Operator):
    """退出泼溅笔刷"""
    bl_idname = "gse.stop_brush"
    bl_label = "停止笔刷"

    def execute(self, context):
        STATE["running"] = False
        return {'FINISHED'}


classes = (
    GSE_OT_set_tool,
    GSE_OT_set_depth,
    GSE_OT_brush,
    GSE_OT_export_ply,
    GSE_OT_merge_splats,
    GSE_OT_bake_edits,
    GSE_OT_stop_brush,
)
