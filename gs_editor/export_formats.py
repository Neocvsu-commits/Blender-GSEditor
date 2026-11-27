"""Cancellable format conversion of a frozen, transformed PLY snapshot."""
import atexit
import json
import os
import signal
from pathlib import Path
import subprocess
import tempfile
import zipfile

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ExportHelper

from . import merge
from .ply_io import Y_UP, write_pointcloud_ply

FORMATS = (
    ('SOG', 'SOG (.sog)', '压缩交付；量化位置、颜色和高阶球谐'),
    ('SPZ', 'SPZ (.spz)', 'SPZ v2 压缩格式；保留量化后的高阶球谐'),
    ('SPLAT', 'Splat (.splat)', '兼容传统查看器；仅基础颜色，不保留高阶球谐'),
    ('LCC', 'LCC + LOD (.zip)', '旧版 LCC：分块索引与多级高斯，打包全部依赖文件'),
    ('LCC2', 'LCC2 + LOD (.zip)', 'LCC2：空间 LOD 树与独立 SOG 压缩块，打包全部依赖文件'),
)
LOD_FORMATS = {'LCC', 'LCC2'}
_job = None
_workers = []


def _terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=5)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        if process.poll() is None:
            try:
                if os.name != 'nt':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError) as ex:
                print('GS export worker cleanup:', ex)


def exit_workers():
    # No RNA access during interpreter shutdown.
    for process, temp in _workers[:]:
        _terminate(process)
        try:
            temp.cleanup()
        except OSError:
            pass
    _workers.clear()


atexit.register(exit_workers)


def _write_lod_archive(source, target, format, options):
    from .lod import build_dataset
    from .binary_formats import _destination
    with tempfile.TemporaryDirectory(prefix='.gs-lod-', dir=str(Path(target).parent)) as temp:
        folder = Path(temp)
        dataset = build_dataset(source, folder / 'levels',
                                levels=options.get('lod_levels', 4),
                                chunk_points=options.get('chunk_points', 65536))
        dataset['name'] = options.get('name', 'Gaussian Scene')
        axis = options.get('up_axis', 'ZUP')
        axis_note = ('本包使用 Y-up 网页兼容坐标；适用于直接按 Y 朝上显示数据的网页。'
                     '它不是 LCC 标准 Z-up 朝向，若查看器另行转轴，请改用 Z 轴朝上重新导出。'
                     if axis == 'YUP' else
                     '本包使用 LCC 标准 Z-up 坐标；查看器需按其自身的坐标约定加载。')
        package = folder / 'package'
        if format == 'LCC':
            from .lcc_io import write_lcc
            notes = write_lcc(dataset, package)
        else:
            from .lcc2_io import write_lcc2
            _, notes = write_lcc2(dataset, package)
        manifest = 'meta.lcc2' if format == 'LCC2' else 'meta.lcc'
        guide = (
            f'GS Editor — {format} 网页交付包\n\n'
            f'入口文件：{manifest}\n'
            '上传支持该格式的网站时，选择完整 ZIP；自行部署时先解压，保留所有目录和文件。\n'
            '网页必须支持此格式的 LOD 调度和分块读取。仅能导入、或转成单层点云的网页不会自动获得 LOD 加速。\n'
            f'{axis_note}\n'
            f'空间分块：{len(dataset["cells"])}\n'
            + '\n'.join(f'LOD{i}: {count:,} 高斯点' for i, count in enumerate(dataset['lod_counts']))
            + '\n\nLOD0 保留全部未擦除点（格式编码可能量化属性）。'
            '后续层通过空间近邻高斯的协方差、颜色、透明度与球谐合并生成，越远越粗；'
            '不是重训练，不保证达到其域原生采集与训练的画质。\n'
            '格式归属、白皮书及分发条件见 FORMAT_NOTICE.txt。\n')
        (package / 'USER_GUIDE.txt').write_text(guide, encoding='utf-8')
        with _destination(target) as out:
            with zipfile.ZipFile(out, 'w', allowZip64=True) as archive:
                for path in sorted(package.rglob('*')):
                    if path.is_file():
                        # SOG is already compressed. Compress LCC's raw binary
                        # blocks for upload; the website unpacks them for ranges.
                        method = zipfile.ZIP_STORED if path.suffix == '.sog' else zipfile.ZIP_DEFLATED
                        archive.write(path, path.relative_to(package).as_posix(),
                                      compress_type=method, compresslevel=6 if method else None)
        if Path(target).stat().st_size > 500 * 1024 * 1024:
            notes.append('交付包超过 500 MB；上传前请确认网页的文件大小限制')
        return dataset['count'], dataset['notes'] + notes


def convert(source, target, format, options=None):
    if format in LOD_FORMATS:
        return _write_lod_archive(source, target, format, options or {})
    if format == 'SOG':
        from .sog_io import write_sog_from_ply
        return write_sog_from_ply(source, target)
    if format not in {'SPZ', 'SPLAT'}:
        raise ValueError('不支持的导出格式：' + format)
    from .binary_formats import write_splat_from_ply, write_spz_from_ply
    return (write_spz_from_ply if format == 'SPZ' else write_splat_from_ply)(source, target)


def worker(source, target, format, result, options=None):
    try:
        count, notes = convert(source, target, format, options)
        data = {'ok': True, 'count': count, 'notes': notes}
    except Exception as ex:
        data = {'ok': False, 'error': str(ex)}
    Path(result).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


class GSE_OT_export_format(bpy.types.Operator, ExportHelper):
    bl_idname = 'gse.export_format'
    bl_label = '导出高斯'
    filename_ext = '.sog'
    format: bpy.props.EnumProperty(items=FORMATS, default='SOG', options={'HIDDEN'})
    filter_glob: bpy.props.StringProperty(default='*.sog', options={'HIDDEN'})
    lod_levels: bpy.props.IntProperty(name='LOD 层数', default=4, min=2, max=6,
        description='包含完整精细层；每层将约四个相邻高斯合并为一个，远景点数逐级减少')
    chunk_points: bpy.props.IntProperty(name='每块目标点数', default=65536, min=4096, max=262144,
        description='更小的空间块有利于按需加载，但增加文件数；极密集或竖直点云可能超过目标')

    @classmethod
    def poll(cls, context):
        from .ops import active_splats, STATE
        return _job is None and active_splats(context) is not None and not any(STATE.values())

    def invoke(self, context, event):
        self.filename_ext = '.zip' if self.format in LOD_FORMATS else '.' + self.format.lower()
        self.filter_glob = '*' + self.filename_ext
        tag = '_' + self.format.lower() if self.format in LOD_FORMATS else '_patched'
        self.filepath = bpy.path.clean_name(context.active_object.name) + tag + self.filename_ext
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        col = self.layout.column()
        col.prop(context.scene.gse, 'export_up_axis')
        if self.format in LOD_FORMATS:
            col.prop(self, 'lod_levels')
            col.prop(self, 'chunk_points')
        if self.format == 'SPLAT':
            col.label(text='Splat 仅保留基础颜色', icon='INFO')
        if self.format in LOD_FORMATS:
            col.separator()
            col.label(text='Data Organization Format')
            col.label(text='originated from XGRIDS')
            col.operator('wm.url_open', text='格式说明与许可', icon='URL').url = (
                'https://github.com/xgrids/' + self.format + 'Whitepaper')

    def execute(self, context):
        global _job
        from .ops import active_splats, export_yup
        self._timer = self._process = self._temp = None
        self._wm = context.window_manager
        self._workspace = context.workspace
        self.cancel_requested = False
        extension = '.zip' if self.format in LOD_FORMATS else '.' + self.format.lower()
        self.filepath = bpy.path.ensure_ext(self.filepath, extension)
        self._target = Path(bpy.path.abspath(self.filepath)).resolve()
        try:
            if _job is not None:
                raise RuntimeError('已有导出任务正在进行')
            obj = active_splats(context)
            if obj is None:
                raise ValueError('请选择原生高斯点云')
            self._temp = tempfile.TemporaryDirectory(prefix='.gs-export-', dir=str(self._target.parent))
            folder = Path(self._temp.name)
            source = folder / 'input.ply'
            self._output = folder / ('output' + self._target.suffix)
            self._result = folder / 'result.json'
            self._log = folder / 'conversion.log'
            matrix = np.asarray(obj.matrix_world, dtype=np.float64)
            yup = export_yup(context)
            if yup:
                matrix = Y_UP @ matrix
            self._count = write_pointcloud_ply(str(source), obj.data, matrix=matrix)
            if not self._count:
                raise ValueError('没有可导出的高斯点（所有点已擦除）')
            self._notes = merge.transform_notes(matrix)
            # Only our code runs in a clean worker. No .blend or user startup scripts are opened.
            parent = str(Path(__file__).resolve().parent.parent)
            options = {'lod_levels': self.lod_levels, 'chunk_points': self.chunk_points,
                       'name': obj.name, 'up_axis': 'YUP' if yup else 'ZUP'}
            code = (f'import sys; sys.path.insert(0, {parent!r}); '
                    f'from gs_editor.export_formats import worker; '
                    f'worker({str(source)!r}, {str(self._output)!r}, {self.format!r}, '
                    f'{str(self._result)!r}, {options!r})')
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            with self._log.open('wb') as log:
                self._process = subprocess.Popen(
                    [bpy.app.binary_path, '--background', '--factory-startup', '--disable-autoexec',
                     '--python-exit-code', '1', '--python-expr', code],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                    start_new_session=os.name != 'nt')
            _workers.append((self._process, self._temp))
            _job = self
            self._timer = self._wm.event_timer_add(0.25, window=context.window)
            self._wm.modal_handler_add(self)
            self._workspace.status_text_set(f'正在导出 {self.format} · Esc 取消')
            return {'RUNNING_MODAL'}
        except Exception as ex:
            self.cleanup()
            self.report({'ERROR'}, f'导出失败：{ex}')
            return {'CANCELLED'}

    def modal(self, context, event):
        if self.cancel_requested or (event.type == 'ESC' and event.value == 'PRESS'):
            self.cleanup()
            self.report({'INFO'}, '已取消导出，原文件保持不变')
            return {'CANCELLED'}
        if event.type != 'TIMER' or self._process.poll() is None:
            return {'PASS_THROUGH'}
        try:
            if not self._result.exists():
                detail = self._log.read_text(encoding='utf-8', errors='replace')[-500:]
                raise RuntimeError('转换进程未完成：' + detail)
            result = json.loads(self._result.read_text(encoding='utf-8'))
            if not result['ok']:
                raise RuntimeError(result['error'])
            if not self._output.exists() or not self._output.stat().st_size:
                raise RuntimeError('转换器未生成有效文件')
            os.replace(self._output, self._target)
            detail = '；'.join(self._notes + result.get('notes', []))
            self.report({'WARNING'} if detail else {'INFO'},
                        f'已导出 {self._count:,} 个高斯点：{self._target.name}' + ('；' + detail if detail else ''))
            return {'FINISHED'}
        except Exception as ex:
            self.report({'ERROR'}, f'导出失败：{ex}')
            return {'CANCELLED'}
        finally:
            self.cleanup()

    def cancel(self, context):
        self.cleanup()

    def cleanup(self):
        global _job
        process = getattr(self, '_process', None)
        _terminate(process)
        _workers[:] = [(p, t) for p, t in _workers if p is not process]
        if getattr(self, '_timer', None) is not None:
            try:
                self._wm.event_timer_remove(self._timer)
            except (ReferenceError, RuntimeError):
                pass
            self._timer = None
        if _job is self:
            _job = None
        if getattr(self, '_temp', None) is not None:
            try:
                self._temp.cleanup()
            except OSError as ex:
                print('GS export temporary files:', ex)
            self._temp = None
        try:
            if getattr(self, '_workspace', None):
                self._workspace.status_text_set(None)
        except ReferenceError:
            pass


class GSE_OT_cancel_export(bpy.types.Operator):
    bl_idname = 'gse.cancel_export'
    bl_label = '取消导出'

    def execute(self, context):
        if _job:
            _job.cancel_requested = True
        return {'FINISHED'}


class GSE_MT_export(bpy.types.Menu):
    bl_label = '导出高斯'
    bl_idname = 'GSE_MT_export'

    def draw(self, context):
        self.layout.operator('gse.export_ply', text='PLY (.ply)…', icon='EXPORT')
        for value, label, _ in FORMATS:
            self.layout.operator('gse.export_format', text=label + '…', icon='EXPORT').format = value


@persistent
def shutdown(_=None):
    if _job is not None:
        _job.cleanup()


classes = (GSE_OT_export_format, GSE_OT_cancel_export, GSE_MT_export)
