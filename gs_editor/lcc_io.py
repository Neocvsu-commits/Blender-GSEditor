"""Legacy LCC writer, independently implemented from the XGRIDS white paper.

https://github.com/xgrids/LCCWhitepaper

The caller supplies a spatially partitioned dataset of canonical PLY
snapshots. Each cell has the same LOD levels, finest first. This module does
not create LODs or change coordinates (standard LCC is Z-up; the export operator
can also bake a user-selected web compatibility rotation). Only a chunk of one PLY is materialized
at a time. All binary fields have explicit little-endian layouts.
"""

from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import struct
import uuid

import numpy as np

from .binary_formats import (
    _alpha, _bytes_round, _columns, _input_ply, _quaternions, _slices,
)


_C0 = 0.28209479177387814
_SCALE_NAMES = ('scale_0', 'scale_1', 'scale_2')
_POSITION_NAMES = ('x', 'y', 'z')
_COLOR_NAMES = ('f_dc_0', 'f_dc_1', 'f_dc_2')
_NORMAL_NAMES = ('nx', 'ny', 'nz')
_RECORD = np.dtype([
    ('position', '<f4', (3,)),
    ('rgba', 'u1', (4,)),
    ('scale', '<u2', (3,)),
    ('rotation', '<u4'),
    ('normal', '<u2', (3,)),
], align=False)

FORMAT_NOTICE = """Data Organization Format originated from XGRIDS
White Paper and format license: https://github.com/xgrids/LCCWhitepaper

GS Editor independently implements this data writer. It is not an official
XGRIDS tool, product, or endorsement. The binary data organization has not
been modified; GS Editor supplies the scene data, spatial partitioning, and
generated levels of detail.

The data organization format is used and distributed under the authorization
provided by XGRIDS in the linked White Paper. XGRIDS and its affiliates retain
all intellectual property and other rights in the data organization format.
Redistribution is subject to that White Paper, including its attribution,
openness, redistribution, and usage conditions. Without prior written consent
from XGRIDS, the format must not be used to train or fine-tune an AI model
competing directly or indirectly with XGRIDS products or services. Recipients
must be informed of that restriction, and it must be incorporated into the
applicable use or distribution agreement. See the White Paper for all terms.
"""


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('LCC 的' + name + '必须是整数')
    result = int(value)
    if result != value or result < minimum or (maximum is not None and result > maximum):
        raise ValueError('LCC 的' + name + '超出允许范围')
    return result


def _bounds(value, name):
    bounds = np.asarray(value, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
        raise ValueError('LCC 的' + name + '无效')
    return bounds


def _dataset_info(dataset):
    levels = _integer(dataset['levels'], 'LOD 层数', 1)
    count = _integer(dataset['count'], '原始点数', 1)
    bounds = _bounds(dataset['bounds'], '整体边界')
    length = np.asarray(dataset['cell_length'], dtype=np.float64)
    if length.shape != (2,) or not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError('LCC 的空间块尺寸必须大于零')
    lod_counts = [_integer(n, 'LOD 点数') for n in dataset['lod_counts']]
    if len(lod_counts) != levels or lod_counts[0] != count:
        raise ValueError('LCC 的 LOD 点数与原始点数不一致')
    actual_counts = [0] * levels
    cells = list(dataset['cells'])
    seen = set()
    for cell in cells:
        # The white paper uses uint16. Existing readers interpret these as
        # int16, so the shared partitioner keeps indices in the common range.
        key = tuple(_integer(cell[axis], '空间块索引', 0, 32767) for axis in ('x', 'y'))
        if key in seen:
            raise ValueError('LCC 存在重复的空间块索引')
        seen.add(key)
        _bounds(cell['bounds'], '空间块边界')
        if len(cell['levels']) != levels:
            raise ValueError('LCC 各空间块的 LOD 层数必须相同')
        for index, node in enumerate(cell['levels']):
            # Leave sizes representable in readers using signed int32 too.
            n = _integer(node['count'], '空间块点数', 0, 0x7fffffff // 32)
            actual_counts[index] += n
    if actual_counts != lod_counts or not cells:
        raise ValueError('LCC 空间块点数与 LOD 统计不一致')
    return cells, levels, bounds, length, lod_counts


def _sh_size(values):
    names = {name for name in values if name.startswith('f_rest_')}
    count = len(names)
    if count not in (0, 9, 24, 45) or names != {'f_rest_' + str(i) for i in range(count)}:
        raise ValueError('LCC 仅支持完整的 SH0、SH1、SH2 或 SH3 属性')
    return count // 3


def _sh_values(values, section, coefficient_count):
    n = section.stop - section.start
    # PLY stores channel-major SH, while LCC packs RGB per coefficient.
    result = np.zeros((n, 15, 3), dtype=np.float64)
    if coefficient_count:
        for channel in range(3):
            names = tuple('f_rest_' + str(channel * coefficient_count + i)
                          for i in range(coefficient_count))
            result[:, :coefficient_count, channel] = _columns(values, names, section)
    return result


def _scales(values, section):
    logs = _columns(values, _SCALE_NAMES, section).astype(np.float64)
    with np.errstate(over='ignore', under='ignore'):
        result = np.exp(logs)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('高斯尺度过大或过小，无法编码为 LCC 线性尺度')
    return result


def _normals(values, section):
    existing = [name in values for name in _NORMAL_NAMES]
    if any(existing) and not all(existing):
        raise ValueError('PLY 的法线属性不完整')
    if all(existing):
        return _columns(values, _NORMAL_NAMES, section).astype(np.float64)
    # Gaussian snapshots without surface normals use the format's zero normal.
    return np.zeros((section.stop - section.start, 3), dtype=np.float64)


def _range_update(limits, values):
    axes = tuple(range(values.ndim - 1))
    limits[0] = np.minimum(limits[0], np.min(values, axis=axes))
    limits[1] = np.maximum(limits[1], np.max(values, axis=axes))


def _scan(cells):
    limits = {name: np.array([[math.inf] * size, [-math.inf] * size], dtype=np.float64)
              for name, size in (('position', 3), ('normal', 3), ('color', 3),
                                 ('shcoef', 3), ('opacity', 1), ('scale', 3))}
    has_sh = False
    padded_sh = False
    clipped_color = False
    for cell in cells:
        for node in cell['levels']:
            if not node['count']:
                continue
            with _input_ply(node['path']) as values:
                if len(values['x']) != node['count']:
                    raise ValueError('LCC 空间块的 PLY 点数与 LOD 索引不一致')
                sh_count = _sh_size(values)
                has_sh |= sh_count > 0
                padded_sh |= sh_count in (3, 8)
                for section in _slices(node['count']):
                    _range_update(limits['position'], _columns(values, _POSITION_NAMES, section))
                    _range_update(limits['scale'], _scales(values, section))
                    _range_update(limits['normal'], _normals(values, section))
                    colors = _columns(values, _COLOR_NAMES, section).astype(np.float64) * _C0 + 0.5
                    clipped_color |= bool(np.any((colors < 0) | (colors > 1)))
                    _range_update(limits['color'], _bytes_round(colors * 255).astype(np.float64) / 255)
                    alpha = _bytes_round(_alpha(values['opacity'][section]) * 255)
                    _range_update(limits['opacity'], alpha.astype(np.float64)[:, None] / 255)
                    _range_update(limits['shcoef'], _sh_values(values, section, sh_count))
                    # Validate before writing output files, including rotations.
                    _quaternions(values, section)
    return limits, has_sh, padded_sh, clipped_color


def _quantize(values, limits, maximum):
    minimum, maximum_value = limits
    span = maximum_value - minimum
    out = np.zeros_like(values, dtype=np.float64)
    np.divide(values - minimum, span, out=out, where=span != 0)
    # Limits come from all input LODs. Clipping only removes numerical epsilon.
    if np.any(out < -1e-7) or np.any(out > 1.0000001):
        raise ValueError('LCC 输入数据在编码过程中发生变化，请重新导出')
    return np.floor(np.clip(out, 0, 1) * maximum + 0.5).astype(np.uint32)


def _packed_rotations(values, section):
    q = _quaternions(values, section)[:, (1, 2, 3, 0)]  # WXYZ -> XYZW
    largest = np.argmax(np.abs(q), axis=1)
    rows = np.arange(len(q))
    q *= np.where(q[rows, largest] < 0, -1.0, 1.0)[:, None]
    keep = np.arange(4)[None, :] != largest[:, None]
    remaining = q[keep].reshape(-1, 3)
    normalized = (remaining + math.sqrt(0.5)) / math.sqrt(2.0)
    bits = np.floor(np.clip(normalized, 0, 1) * 1023 + 0.5).astype(np.uint32)
    return bits[:, 0] | (bits[:, 1] << 10) | (bits[:, 2] << 20) | (largest.astype(np.uint32) << 30)


def _write_node(node, data_file, sh_file, limits):
    if not node['count']:
        return
    with _input_ply(node['path']) as values:
        if len(values['x']) != node['count']:
            raise ValueError('LCC 输入点数在编码过程中发生变化')
        sh_count = _sh_size(values)
        for section in _slices(node['count']):
            n = section.stop - section.start
            records = np.empty(n, dtype=_RECORD)
            records['position'] = _columns(values, _POSITION_NAMES, section)
            rgb = _columns(values, _COLOR_NAMES, section).astype(np.float64) * _C0 + 0.5
            records['rgba'][:, :3] = _bytes_round(rgb * 255)
            records['rgba'][:, 3] = _bytes_round(_alpha(values['opacity'][section]) * 255)
            records['scale'] = _quantize(_scales(values, section), limits['scale'], 65535)
            records['rotation'] = _packed_rotations(values, section)
            records['normal'] = _quantize(_normals(values, section), limits['normal'], 65535)
            data_file.write(records.tobytes())
            if sh_file is not None:
                sh = _sh_values(values, section, sh_count)
                encoded = _quantize(sh, limits['shcoef'], np.array([2047, 1023, 2047]))
                packed = np.zeros((n, 16), dtype='<u4')
                packed[:, :15] = encoded[:, :, 0] | (encoded[:, :, 1] << 11) | (encoded[:, :, 2] << 21)
                sh_file.write(packed.tobytes())


def write_lcc(dataset, folder):
    """Write a legacy LCC directory and return user-readable conversion notes.

    ``dataset`` contains ``cells``, ``bounds`` ([min,max]), ``cell_length``
    ([x,y]), ``levels``, ``count`` (LOD0), and ``lod_counts``. A cell contains
    integer ``x``/``y``, ``bounds``, and ``levels`` entries with ``path``/``count``.
    The caller owns the fresh staging folder and cleans it if writing fails.
    """
    cells, level_count, bounds, cell_length, lod_counts = _dataset_info(dataset)
    limits, has_sh, padded_sh, clipped_color = _scan(cells)
    tolerance = np.maximum(np.max(np.abs(bounds), axis=0) * 1e-6, 1e-6)
    if (np.any(limits['position'][0] < bounds[0] - tolerance)
            or np.any(limits['position'][1] > bounds[1] + tolerance)):
        raise ValueError('LCC 高斯位置超出空间分块的整体边界')

    folder = Path(os.fspath(folder))
    folder.mkdir(parents=True, exist_ok=True)
    filenames = ['index.bin', 'data.bin', 'meta.lcc', 'FORMAT_NOTICE.txt', 'shcoef.bin']
    if any((folder / name).exists() for name in filenames):
        raise FileExistsError('LCC 输出目录已含格式文件，请使用新的空目录')
    with ExitStack() as stack:
        index_file = stack.enter_context((folder / 'index.bin').open('xb'))
        data_file = stack.enter_context((folder / 'data.bin').open('xb'))
        sh_file = stack.enter_context((folder / 'shcoef.bin').open('xb')) if has_sh else None
        for cell in cells:
            index_file.write(struct.pack('<HH', int(cell['x']), int(cell['y'])))
            for node in cell['levels']:
                count = int(node['count'])
                offset = data_file.tell()
                index_file.write(struct.pack('<IQI', count, offset, count * 32))
                _write_node(node, data_file, sh_file, limits)

    metadata = {
        'version': '5.0',
        'guid': str(uuid.uuid4()),
        'name': str(dataset.get('name', 'GS Editor Gaussian Scene')),
        'description': 'Independent GS Editor export. Data Organization Format originated from XGRIDS',
        'source': 'ply',
        'dataType': 'None',
        'totalSplats': sum(lod_counts),
        'totalLevel': level_count,
        'cellLengthX': float(cell_length[0]),
        'cellLengthY': float(cell_length[1]),
        'indexDataSize': 4 + 16 * level_count,
        'offset': [0, 0, 0],
        'epsg': 0,
        'shift': [0, 0, 0],
        'scale': [1, 1, 1],
        'splats': lod_counts,
        'boundingBox': {'min': bounds[0].tolist(), 'max': bounds[1].tolist()},
        'encoding': 'COMPRESS',
        'fileType': 'Quality' if has_sh else 'Portable',
        'attributes': [{'name': name, 'min': value[0].tolist(), 'max': value[1].tolist()}
                       for name, value in limits.items()],
    }
    with (folder / 'meta.lcc').open('x', encoding='utf-8', newline='\n') as output:
        json.dump(metadata, output, ensure_ascii=False, allow_nan=False, indent=2)
        output.write('\n')
    (folder / 'FORMAT_NOTICE.txt').write_text(FORMAT_NOTICE, encoding='utf-8')
    notes = ['LCC 使用 8 位基础颜色/透明度、16 位线性尺度和压缩旋转']
    if has_sh:
        notes.append('LCC 保留 SH3 视角颜色，按 RGB 11/10/11 位量化')
    if padded_sh:
        notes.append('LCC 原生仅支持 SH0/SH3，SH1/SH2 已补零为 SH3')
    if clipped_color:
        notes.append('部分基础颜色超出 0–1，已按 LCC 的 8 位颜色范围截断')
    return notes
