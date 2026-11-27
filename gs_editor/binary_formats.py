"""Portable .splat and SPZ v2 writers for the add-on's transformed PLY output.

These are independent NumPy implementations of the published file layouts:
https://github.com/antimatter15/splat/blob/main/convert.py
https://github.com/nianticlabs/spz/blob/v1.1.0/src/cc/load-spz.cc

Coordinates are already transformed by the caller and are not flipped here.
In particular, SPZ readers conventionally interpret them as RUB (Y-up).
Both functions return (point_count, notes). They replace the destination only
after the entire write succeeds. SPZ v2 uses gzip and requires no native codec.
"""

from contextlib import contextmanager
import gzip
import os
import struct
import tempfile

import numpy as np


_CHUNK = 65536
_SH_C0 = 0.28209479177387814
_REQUIRED = ('x', 'y', 'z', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')
_SPLAT_DTYPE = np.dtype([('position', '<f4', (3,)), ('scale', '<f4', (3,)),
                         ('color', 'u1', (4,)), ('rotation', 'u1', (4,))])


@contextmanager
def _input_ply(path):
    """Memory-map our canonical float32 PLY; reuse the reference reader otherwise."""
    mapped = None
    values = None
    with open(path, 'rb') as src:
        if src.readline().strip() != b'ply':
            raise ValueError('输入文件不是 PLY')
        count, names, canonical = None, [], False
        header_size = 0
        for _ in range(10000):
            line = src.readline(65536)
            header_size += len(line)
            if not line or header_size > 1048576:
                raise ValueError('PLY 文件头不完整或过大')
            fields = line.strip().split()
            if not fields or fields[0] in (b'comment', b'obj_info'):
                continue
            if fields[0] == b'end_header':
                break
            if fields[0] == b'format':
                canonical = fields[1:] == [b'binary_little_endian', b'1.0']
            elif fields[0] == b'element':
                if len(fields) != 3 or fields[1] != b'vertex' or count is not None:
                    canonical = False
                    break
                count = int(fields[2])
            elif fields[0] == b'property':
                if count is None or len(fields) != 3 or fields[1] != b'float':
                    canonical = False
                    break
                names.append(fields[2].decode('ascii'))
            else:
                canonical = False
                break
        else:
            raise ValueError('PLY 文件头属性过多')
        if canonical:
            if count is None or not 0 < count <= 20000000:
                raise ValueError('没有可导出的高斯点，或点数超过 2000 万')
            if len(set(names)) != len(names) or not set(_REQUIRED).issubset(names):
                raise ValueError('PLY 的高斯属性缺失或重复')
            dtype = np.dtype([(name, '<f4') for name in names])
            offset = src.tell()
            if os.fstat(src.fileno()).st_size - offset < count * dtype.itemsize:
                raise ValueError('PLY 顶点数据不完整')
            mapped = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))
            values = {name: mapped[name] for name in names}
    try:
        if values is None:
            from .reference import read_ply
            values = read_ply(path)
        for name, column in values.items():
            for start in range(0, len(column), _CHUNK):
                if not np.isfinite(column[start:start + _CHUNK]).all():
                    raise ValueError('PLY 包含非有限数值：' + name)
        yield values
    finally:
        if mapped is not None:
            # Views are used only inside this context and never returned to callers.
            values.clear()
            mapped._mmap.close()


@contextmanager
def _destination(path):
    path = os.path.abspath(os.fspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.gse-export-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as out:
            yield out
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _slices(n):
    for start in range(0, n, _CHUNK):
        yield slice(start, min(start + _CHUNK, n))


def _columns(values, names, section):
    return np.column_stack([values[name][section] for name in names])


def _quaternions(values, section):
    q = _columns(values, ('rot_0', 'rot_1', 'rot_2', 'rot_3'), section).astype(np.float64)
    lengths = np.linalg.norm(q, axis=1)
    if np.any(lengths < 1e-20):
        raise ValueError('存在无效的零长度高斯旋转；请修复后再导出')
    q /= lengths[:, None]
    return q


def _alpha(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80.0, 80.0)))


def _bytes_round(values):
    # Encoded byte values are nonnegative; half-up agrees with the SPZ codec.
    return np.floor(np.clip(values, 0, 255) + 0.5).astype(np.uint8)


def _round_signed(values):
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def write_splat_from_ply(input_ply, output_path):
    """Write 32-byte records: float32 XYZ/scale, RGBA8, quaternion WXYZ8.

    Splat has no SH storage. It necessarily discards view-dependent colour.
    Point order is retained; viewers perform their own depth sorting.
    """
    notes = ['Splat 使用 8 位基础颜色与旋转，不保存高阶视角颜色（SH）']
    clipped_color = False
    with _input_ply(input_ply) as values:
        n = len(values['x'])
        with _destination(output_path) as out:
            for section in _slices(n):
                rows = np.empty(section.stop - section.start, dtype=_SPLAT_DTYPE)
                rows['position'] = _columns(values, ('x', 'y', 'z'), section)
                logs = _columns(values, ('scale_0', 'scale_1', 'scale_2'), section)
                with np.errstate(over='ignore', under='ignore'):
                    scales = np.exp(logs)
                if not np.isfinite(scales).all() or np.any(scales <= 0):
                    raise ValueError('Splat 尺寸超出 float32 范围；请调整高斯缩放')
                rows['scale'] = scales
                rgb = 0.5 + _SH_C0 * _columns(values, ('f_dc_0', 'f_dc_1', 'f_dc_2'), section)
                clipped_color |= bool(np.any((rgb < 0) | (rgb > 1)))
                rows['color'][:, :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                rows['color'][:, 3] = (_alpha(values['opacity'][section]) * 255).astype(np.uint8)
                rows['rotation'] = np.clip(_quaternions(values, section) * 128 + 128, 0, 255).astype(np.uint8)
                out.write(rows.tobytes())
    if clipped_color:
        notes.append('Splat 的基础颜色超出 0–1 的部分已截断')
    return n, notes


def write_spz_from_ply(input_ply, output_path):
    """Write legacy SPZ v2, retaining complete SH bands through degree 3.

    Positions use signed 24-bit fixed point with 12 fractional bits. All other
    streams are quantized bytes; SH uses all 8 available bits rather than the
    reference encoder's optional coarser compression buckets. XYZ quaternion
    components are stored with nonnegative W. No coordinate conversion occurs.
    """
    notes = ['SPZ v2 使用有损量化；坐标精度为 1/4096 单位，保留至三阶 SH']
    clipped = set()
    with _input_ply(input_ply) as values:
        n = len(values['x'])
        sh_names = [name for name in values if name.startswith('f_rest_')]
        if (len(sh_names) not in (0, 9, 24, 45)
                or set(sh_names) != {f'f_rest_{i}' for i in range(len(sh_names))}):
            raise ValueError('SPZ 需要完整的 0–3 阶 SH（0、9、24 或 45 个连续 f_rest 属性）')
        degree = (0, 9, 24, 45).index(len(sh_names))
        coefficients = len(sh_names) // 3
        with _destination(output_path) as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as out:
                out.write(struct.pack('<IIIBBBB', 0x5053474e, 2, n, degree, 12, 0, 0))
                for section in _slices(n):
                    xyz = _columns(values, ('x', 'y', 'z'), section).astype(np.float64)
                    fixed = _round_signed(xyz * 4096)
                    if np.any((fixed < -8388608) | (fixed > 8388607)):
                        raise ValueError('SPZ v2 坐标超出约 ±2048 单位范围；请缩小/移近原点，或改用 PLY')
                    packed = fixed.astype(np.int32).astype('<i4').view(np.uint8).reshape(-1, 4)[:, :3]
                    out.write(packed.tobytes())
                for section in _slices(n):
                    out.write(_bytes_round(_alpha(values['opacity'][section]) * 255).tobytes())
                for section in _slices(n):
                    dc = _columns(values, ('f_dc_0', 'f_dc_1', 'f_dc_2'), section)
                    encoded = dc.astype(np.float64) * (0.15 * 255) + 127.5
                    if np.any((encoded < 0) | (encoded > 255)):
                        clipped.add('基础颜色')
                    out.write(_bytes_round(encoded).tobytes())
                for section in _slices(n):
                    logs = _columns(values, ('scale_0', 'scale_1', 'scale_2'), section)
                    encoded = (logs.astype(np.float64) + 10) * 16
                    if np.any((encoded < 0) | (encoded > 255)):
                        clipped.add('尺寸')
                    out.write(_bytes_round(encoded).tobytes())
                for section in _slices(n):
                    q = _quaternions(values, section)
                    q[q[:, 0] < 0] *= -1
                    out.write(_bytes_round(q[:, 1:] * 127.5 + 127.5).tobytes())
                for section in _slices(n):
                    if not coefficients:
                        break
                    # PLY is channel-major, SPZ interleaves RGB per SH coefficient.
                    sh = np.empty((section.stop - section.start, coefficients, 3), np.uint8)
                    for channel in range(3):
                        for coefficient in range(coefficients):
                            column = values[f'f_rest_{channel * coefficients + coefficient}'][section]
                            encoded = _round_signed(column.astype(np.float64) * 128) + 128
                            if np.any((encoded < 0) | (encoded > 255)):
                                clipped.add('高阶 SH')
                            sh[:, coefficient, channel] = _bytes_round(encoded)
                    out.write(sh.tobytes())
    if clipped:
        notes.append('SPZ 表示范围外的' + '、'.join(sorted(clipped)) + '已截断；需要无损保留时请使用 PLY')
    return n, notes
