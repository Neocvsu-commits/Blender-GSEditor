"""Portable PLY reference objects and sorted Gaussian viewport drawing (Blender 4.2+).

Mesh vertices/attributes are the persistent storage, not a reconstructed surface.
The renderer uses Blender's GPU abstraction, including Metal, without native wheels.
"""
import os
import numpy as np
import bpy
import gpu
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ImportHelper
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

TAG = 'gse_reference'
PREFIX = 'gse_ply_'
_handle = None
_shader = None
_data = {}
_batches = {}
_error = ''
_types = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2',
          'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8',
          'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2',
          'int32': 'i4', 'uint32': 'u4', 'float32': 'f4', 'float64': 'f8'}
REQUIRED = ('x', 'y', 'z', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2',
            'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')


def read_ply(path):
    """Read standard uncompressed 3DGS PLY; reject unsupported layouts explicitly."""
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('文件不是 PLY')
        props, fmt, count, element, size = [], None, None, None, 0
        while True:
            line = f.readline(65536)
            size += len(line)
            if not line or size > 1048576:
                raise ValueError('PLY 文件头不完整或过大')
            fields = line.decode('ascii').strip().split()
            if not fields:
                continue
            if fields[0] == 'end_header':
                break
            if fields[0] == 'format':
                fmt = fields[1]
            elif fields[0] == 'element':
                element = fields[1]
                if count is None and element != 'vertex' and int(fields[2]):
                    raise ValueError('只支持 vertex 为首个非空元素的标准高斯 PLY')
                if element == 'vertex':
                    count = int(fields[2])
            elif fields[0] == 'property' and element == 'vertex':
                if len(fields) != 3 or fields[1] not in _types:
                    raise ValueError('不支持列表或压缩编码的高斯顶点属性')
                props.append((fields[2], _types[fields[1]]))
        names = [p[0] for p in props]
        if len(set(names)) != len(names):
            raise ValueError('PLY 含重复属性')
        missing = set(REQUIRED) - set(names)
        if missing:
            raise ValueError('需要标准 3DGS PLY，缺少属性：' + ', '.join(sorted(missing)))
        if count is None or count < 1 or count > 20000000:
            raise ValueError('点数必须在 1 到 2000 万之间')
        if fmt in ('binary_little_endian', 'binary_big_endian'):
            endian = '<' if fmt == 'binary_little_endian' else '>'
            dtype = np.dtype([(name, endian + kind) for name, kind in props])
            if count * dtype.itemsize > os.fstat(f.fileno()).st_size - f.tell():
                raise ValueError('PLY 顶点数据不完整')
            rows = np.fromfile(f, dtype=dtype, count=count)
            result = {name: rows[name].astype(np.float32) for name in names}
        elif fmt == 'ascii':
            rows = np.empty((count, len(props)), np.float32)
            for i in range(count):
                values = f.readline().split()
                if len(values) != len(props):
                    raise ValueError('PLY 顶点列数不一致或数据不完整')
                rows[i] = values
            result = dict(zip(names, rows.T))
        else:
            raise ValueError('不支持的 PLY 编码')
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('PLY 包含 NaN 或无穷数值')
        return result


def create_reference(context, path, up_axis):
    values = read_ply(path)
    mesh = bpy.data.meshes.new(os.path.basename(path))
    obj = None
    try:
        n = len(values['x'])
        mesh.vertices.add(n)
        mesh.vertices.foreach_set('co', np.column_stack([values[k] for k in ('x', 'y', 'z')]).ravel())
        for name, data in values.items():
            if name in ('x', 'y', 'z'):
                continue
            if len((PREFIX + name).encode('utf-8')) > 63:
                raise ValueError('PLY 属性名过长：' + name)
            mesh.attributes.new(PREFIX + name, 'FLOAT', 'POINT').data.foreach_set('value', data)
        mesh.update()
        obj = bpy.data.objects.new(os.path.splitext(os.path.basename(path))[0], mesh)
        obj[TAG] = True
        # Hide the storage mesh's loose vertices; the draw handler supplies the image.
        obj.display_type = 'BOUNDS'
        if up_axis == 'YUP':
            obj.matrix_world = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
        context.collection.objects.link(obj)
        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        return obj
    except Exception:
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise


def is_reference(obj):
    return obj is not None and obj.type == 'MESH' and bool(obj.get(TAG))


def arrays(obj):
    mesh = obj.data
    key = mesh.as_pointer()
    if key in _data:
        return _data[key]
    n = len(mesh.vertices)
    pos = np.empty((n, 3), np.float32)
    mesh.vertices.foreach_get('co', pos.ravel())

    def get(name):
        v = np.empty(n, np.float32)
        attr = mesh.attributes.get(PREFIX + name)
        if attr is None:
            raise ValueError('高斯参考属性缺失：' + name)
        attr.data.foreach_get('value', v)
        return v

    scale = np.exp(np.clip(np.column_stack([get('scale_' + str(i)) for i in range(3)]), -20, 20))
    q = np.column_stack([get('rot_' + str(i)) for i in range(4)])
    lengths = np.linalg.norm(q, axis=1)
    q[lengths < 1e-12] = (1, 0, 0, 0)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                         2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                         2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1).reshape(n, 3, 3)
    axes = rotation * scale[:, None, :]
    rgba = np.empty((n, 4), np.float32)
    rgba[:, :3] = np.clip(0.5 + 0.28209479177387814 * np.column_stack([get('f_dc_' + str(i)) for i in range(3)]), 0, 1)
    rgba[:, 3] = 1 / (1 + np.exp(-np.clip(get('opacity'), -80, 80)))
    _data[key] = pos, axes, rgba
    return _data[key]


def project(pos, axes, rgba, matrix, view, width, height):
    """Project covariance through the perspective Jacobian; also works orthographically."""
    clip = pos @ matrix[:, :3].T + matrix[:, 3]
    valid = (clip[:, 3] > 1e-6) & (clip[:, 2] > -clip[:, 3]) & (clip[:, 2] < clip[:, 3])
    pos, axes, rgba, clip = pos[valid], axes[valid], rgba[valid], clip[valid]
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None]
    jac = (matrix[None, :2, :3] - ndc[:, :2, None] * matrix[None, 3, :3]) / w[:, None, None]
    jac *= np.array([width / 2, height / 2], np.float32)[None, :, None]
    projected_axes = jac @ axes
    cov = projected_axes @ projected_axes.transpose(0, 2, 1)
    # Subpixel low-pass filter, in pixel units.
    a, b, c = cov[:, 0, 0] + 0.3, cov[:, 0, 1], cov[:, 1, 1] + 0.3
    radius = np.sqrt(np.maximum((a-c)**2 + 4*b*b, 0))
    angle = 0.5 * np.arctan2(2*b, a-c)
    cosine, sine = np.cos(angle), np.sin(angle)
    major = np.sqrt(np.maximum((a+c+radius)/2, 0)) * 3
    minor = np.sqrt(np.maximum((a+c-radius)/2, 0)) * 3
    e1 = np.column_stack((cosine*major, sine*major)) * (2/width, 2/height)
    e2 = np.column_stack((-sine*minor, cosine*minor)) * (2/width, 2/height)
    reach = np.abs(e1) + np.abs(e2)
    keep = np.all(np.abs(ndc[:, :2]) < 1 + reach, axis=1) & (rgba[:, 3] > 1/255)
    depth = -(pos @ view[2, :3] + view[2, 3])
    return ndc[keep], e1[keep], e2[keep], rgba[keep], depth[keep]


def shader():
    global _shader
    if _shader is None:
        interface = gpu.types.GPUStageInterfaceInfo('gse_reference_interface')
        interface.smooth('VEC2', 'uv')
        interface.smooth('VEC4', 'tint')
        info = gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0, 'VEC3', 'position')
        info.vertex_in(1, 'VEC2', 'corner')
        info.vertex_in(2, 'VEC4', 'color')
        info.vertex_out(interface)
        info.fragment_out(0, 'VEC4', 'fragColor')
        info.vertex_source('void main() { gl_Position = vec4(position, 1.0); uv = corner * 3.0; tint = color; }')
        info.fragment_source('void main() { float r = dot(uv, uv); if (r > 9.0) discard; float a = tint.a * exp(-0.5*r); if (a < 0.00392157) discard; fragColor = vec4(tint.rgb, min(a, 0.99)); }')
        _shader = gpu.shader.create_from_info(info)
    return _shader


def build_batch(context, objects):
    region, rv = context.region, context.region_data
    parts = []
    total = sum(len(o.data.vertices) for o in objects)
    stride = max(1, int(np.ceil(total / context.scene.gse.reference_limit)))
    for obj in objects:
        pos, axes, rgba = arrays(obj)
        rgba = rgba[::stride].copy()
        rgba[:, 3] *= obj.gse_reference_opacity
        matrix = np.array(rv.perspective_matrix @ obj.matrix_world, np.float32)
        view = np.array(rv.view_matrix @ obj.matrix_world, np.float32)
        parts.append(project(pos[::stride], axes[::stride], rgba, matrix, view, region.width, region.height))
    if not parts:
        return None
    ndc, e1, e2, rgba, depth = [np.concatenate(x) for x in zip(*parts)]
    order = np.argsort(-depth, kind='stable')
    ndc, e1, e2, rgba = [x[order] for x in (ndc, e1, e2, rgba)]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], np.float32)
    positions = np.repeat(ndc[:, None, :], 6, axis=1)
    positions[:, :, :2] += e1[:, None, :] * corners[None, :, :1] + e2[:, None, :] * corners[None, :, 1:]
    if not len(positions):
        return None
    return batch_for_shader(shader(), 'TRIS', {'position': positions.reshape(-1, 3),
                            'corner': np.tile(corners, (len(ndc), 1)), 'color': np.repeat(rgba, 6, axis=0)})


def draw():
    global _error
    context = bpy.context
    if not context.region_data or not hasattr(context.scene, 'gse'):
        return
    if not context.scene.gse.reference_show:
        return
    objects = [o for o in context.visible_objects if is_reference(o) and o.visible_in_viewport_get(context.space_data)]
    if not objects:
        return
    key = (context.region.as_pointer(), context.scene.as_pointer(), context.scene.gse.reference_limit,
           context.region.width, context.region.height, tuple(v for row in context.region_data.perspective_matrix for v in row),
           tuple((o.as_pointer(), o.data.as_pointer(), tuple(v for row in o.matrix_world for v in row), o.gse_reference_opacity) for o in objects))
    try:
        if key not in _batches:
            if len(_batches) >= 4:
                _batches.clear()
            _batches[key] = build_batch(context, objects)
        batch = _batches[key]
        if batch is not None:
            blend, test, mask = gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get()
            try:
                gpu.state.blend_set('ALPHA')
                gpu.state.depth_test_set('LESS_EQUAL')
                gpu.state.depth_mask_set(False)
                shader().bind()
                batch.draw(shader())
            finally:
                gpu.state.depth_mask_set(mask)
                gpu.state.depth_test_set(test)
                gpu.state.blend_set(blend)
        _error = ''
    except Exception as ex:
        message = str(ex)
        if message != _error:
            print('GS Editor reference:', message)
        _error = message


@persistent
def invalidate(*_args):
    _data.clear()
    _batches.clear()


@persistent
def restore_reference_display(*_args):
    """Migrate existing references without changing viewport overlays for other objects."""
    invalidate()
    for obj in bpy.data.objects:
        if is_reference(obj) and obj.library is None and obj.display_type != 'BOUNDS':
            obj.display_type = 'BOUNDS'


@persistent
def changed(scene, depsgraph):
    for update in depsgraph.updates:
        data = update.id
        if (isinstance(data, bpy.types.Mesh) and data.original.as_pointer() in _data) or (isinstance(data, bpy.types.Object) and is_reference(data) and update.is_updated_geometry):
            invalidate()
            break


def redraw(self, context):
    _batches.clear()
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class GSE_OT_import_reference(bpy.types.Operator, ImportHelper):
    bl_idname = 'gse.import_reference'
    bl_label = '导入高斯参考 PLY'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.ply'
    filter_glob: bpy.props.StringProperty(default='*.ply', options={'HIDDEN'})
    up_axis: bpy.props.EnumProperty(name='文件朝上轴', items=(('ZUP', 'Z 轴', '保持文件坐标'), ('YUP', 'Y 轴', '转为 Blender Z-up')), default='ZUP')

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        try:
            obj = create_reference(context, self.filepath, self.up_axis)
        except Exception as ex:
            self.report({'ERROR'}, str(ex))
            return {'CANCELLED'}
        invalidate()
        redraw(self, context)
        self.report({'INFO'}, f'已导入 {len(obj.data.vertices):,} 个高斯点')
        return {'FINISHED'}


class GSE_PT_reference(bpy.types.Panel):
    bl_label = '建模参考'
    bl_parent_id = 'GSE_PT_panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GS 编辑器'

    def draw(self, context):
        layout = self.layout
        layout.operator('gse.import_reference', icon='IMPORT')
        layout.prop(context.scene.gse, 'reference_show')
        layout.prop(context.scene.gse, 'reference_limit')
        obj = context.active_object
        if is_reference(obj):
            layout.label(text=f'{len(obj.data.vertices):,} 个高斯点')
            layout.prop(obj, 'gse_reference_opacity')
        if _error:
            layout.label(text='高斯显示失败：' + _error[:100], icon='ERROR')


def register():
    global _handle
    bpy.types.Object.gse_reference_opacity = bpy.props.FloatProperty(name='参考不透明度', default=1.0, min=0, max=1, update=redraw)
    if not bpy.app.background and _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
    for name, fn in (('load_post', restore_reference_display), ('undo_post', restore_reference_display), ('redo_post', restore_reference_display), ('depsgraph_update_post', changed)):
        handlers = getattr(bpy.app.handlers, name)
        if fn not in handlers:
            handlers.append(fn)
    # Blender restricts bpy.data while enabling an add-on. Defer scene migration
    # until registration finishes; load_post also covers opening a blend at startup.
    if not bpy.app.timers.is_registered(restore_reference_display):
        bpy.app.timers.register(restore_reference_display, first_interval=0.0)


def unregister():
    global _handle, _shader, _error
    if bpy.app.timers.is_registered(restore_reference_display):
        bpy.app.timers.unregister(restore_reference_display)
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')
        _handle = None
    for name, fn in (('load_post', restore_reference_display), ('undo_post', restore_reference_display), ('redo_post', restore_reference_display), ('depsgraph_update_post', changed)):
        handlers = getattr(bpy.app.handlers, name)
        if fn in handlers:
            handlers.remove(fn)
    if hasattr(bpy.types.Object, 'gse_reference_opacity'):
        del bpy.types.Object.gse_reference_opacity
    invalidate()
    _shader = None
    _error = ''
