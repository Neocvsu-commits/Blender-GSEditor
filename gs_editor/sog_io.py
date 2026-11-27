"""Offline SOG v2 export using NumPy and the bundled, lossless cwebp encoder.

The input is the add-on's frozen 3DGS PLY snapshot. Coordinates and SH are already
transformed by the PLY writer; this module must not apply another axis conversion.
Specification: https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog-v2/
PNG is only an intermediate: current SuperSplat readers require real WebP data.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import subprocess
import tempfile
import zipfile
import zlib

import numpy as np

from .binary_formats import _input_ply

_BLOCK = 65536
_PALETTE_LIMIT = 65536
_TRAIN_LIMIT = 262144


def _chunks(n, size=_BLOCK):
    for start in range(0, n, size):
        yield slice(start, min(n, start + size))


def _encoder():
    machine = platform.machine().lower()
    if os.name == 'nt' and machine in ('amd64', 'x86_64'):
        relative = ('windows-x64', 'cwebp.exe')
    elif platform.system() == 'Darwin' and machine in ('arm64', 'aarch64'):
        relative = ('macos-arm64', 'cwebp')
    else:
        raise RuntimeError('SOG 编码器目前支持 Windows x64 和 Apple Silicon macOS')
    path = Path(__file__).parent.joinpath('vendor', 'webp', *relative)
    if not path.is_file():
        raise RuntimeError('缺少内置 SOG 编码器，请重新安装对应平台的完整插件包')
    # ZIP installers on Windows do not necessarily preserve Unix execute bits.
    if os.name != 'nt' and not os.access(path, os.X_OK):
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _png_chunk(file, name, data):
    file.write(struct.pack('>I', len(data)))
    file.write(name)
    file.write(data)
    file.write(struct.pack('>I', zlib.crc32(data, zlib.crc32(name)) & 0xffffffff))


def _write_png(path, rgba):
    """Write raw RGBA bytes, without gamma conversion or alpha premultiplication."""
    height, width = rgba.shape[:2]
    with open(path, 'wb') as file:
        file.write(b'\x89PNG\r\n\x1a\n')
        _png_chunk(file, b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
        compressor = zlib.compressobj(1)
        for part in _chunks(height, 32):
            pixels = rgba[part].reshape(-1, width * 4)
            scanlines = np.empty((len(pixels), width * 4 + 1), np.uint8)
            scanlines[:, 0] = 0  # PNG filter: None
            scanlines[:, 1:] = pixels
            encoded = compressor.compress(scanlines.tobytes())
            if encoded:
                _png_chunk(file, b'IDAT', encoded)
        encoded = compressor.flush()
        if encoded:
            _png_chunk(file, b'IDAT', encoded)
        _png_chunk(file, b'IEND', b'')


@contextmanager
def _image(folder, name, width, height):
    path = folder / (name + '.rgba')
    rgba = np.memmap(path, dtype=np.uint8, mode='w+', shape=(height, width, 4))
    try:
        # Truncation initializes RGB to zero; make unused alpha opaque as well.
        rgba[:, :, 3] = 255
        yield rgba
    finally:
        rgba._mmap.close()
        path.unlink(missing_ok=True)


def _encode(folder, name, rgba, executable):
    source = folder / (name + '.png')
    target = folder / (name + '.webp')
    _write_png(source, rgba)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    # Inherit the worker's process group: cancelling the worker also stops cwebp.
    result = subprocess.run(
        [str(executable), '-lossless', '-exact', '-mt', '-quiet',
         '-o', str(target), str(source)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=flags,
        check=False)
    source.unlink(missing_ok=True)
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        detail = result.stderr.decode('utf-8', errors='replace').strip()[-500:]
        raise RuntimeError('SOG 无损 WebP 编码失败' + (': ' + detail if detail else ''))
    return target.name


def _stack(values, names, part):
    return np.column_stack([values[name][part] for name in names])


def _book(columns):
    """Deterministic shared scalar codebook; includes the full data range."""
    lo = min(float(np.min(column)) for column in columns)
    hi = max(float(np.max(column)) for column in columns)
    return np.linspace(lo, hi, 256, dtype=np.float64)


def _quantize(values, book):
    if book[0] == book[-1]:
        return np.zeros(values.shape, np.uint8)
    scaled = (values.astype(np.float64) - book[0]) * (255.0 / (book[-1] - book[0]))
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _train_sh_tree(values, names, n):
    """A balanced median tree, trained on evenly spaced samples, not N x K distances.

    At each level choose the SH component with the largest range in that node.
    At most 262144 rows train the tree. All input rows subsequently contribute to
    the palette means, so sampling only affects the partition, not which points
    or SH bands are exported.
    """
    sample_n = 1 << (min(n, _TRAIN_LIMIT).bit_length() - 1)
    indices = np.linspace(0, n - 1, sample_n, dtype=np.int64)
    samples = _stack(values, names, indices)
    order = np.arange(sample_n, dtype=np.int32)
    dimensions = np.empty(_PALETTE_LIMIT - 1, np.uint8)
    thresholds = np.empty(_PALETTE_LIMIT - 1, np.float64)
    for depth in range(16):
        nodes = 1 << depth
        groups = order.reshape(nodes, -1)
        block = samples[groups]
        ranges = np.ptp(block.astype(np.float64), axis=1)
        dims = np.argmax(ranges, axis=1)
        del ranges
        scalar = np.take_along_axis(block, dims[:, None, None], axis=2)[:, :, 0]
        del block
        sort = np.argsort(scalar, axis=1, kind='stable')
        half = scalar.shape[1] // 2
        middle = np.take_along_axis(scalar, sort[:, half - 1:half + 1], axis=1)
        base = nodes - 1
        dimensions[base:base + nodes] = dims
        thresholds[base:base + nodes] = middle.astype(np.float64).mean(axis=1)
        order = np.take_along_axis(groups, sort, axis=1).reshape(-1)
    return dimensions, thresholds


def _classify_sh(block, dimensions, thresholds):
    nodes = np.zeros(len(block), np.int32)
    rows = np.arange(len(block))
    for _ in range(16):
        right = block[rows, dimensions[nodes]] > thresholds[nodes]
        nodes = nodes * 2 + 1 + right
    return (nodes - (_PALETTE_LIMIT - 1)).astype(np.uint16)


@contextmanager
def _sh_palette(values, names, n, folder):
    label_path = folder / 'sh-labels.u16'
    labels = np.memmap(label_path, dtype='<u2', mode='w+', shape=(n,))
    try:
        if n <= _PALETTE_LIMIT:
            palette = _stack(values, names, slice(None))
            labels[:] = np.arange(n, dtype=np.uint16)
        else:
            dimensions, thresholds = _train_sh_tree(values, names, n)
            # Transposed sums keep each bincount update contiguous.
            sums = np.zeros((len(names), _PALETTE_LIMIT), np.float64)
            counts = np.zeros(_PALETTE_LIMIT, np.int64)
            for part in _chunks(n, _TRAIN_LIMIT):
                block = _stack(values, names, part)
                group = _classify_sh(block, dimensions, thresholds)
                labels[part] = group
                counts += np.bincount(group, minlength=_PALETTE_LIMIT)
                for column in range(len(names)):
                    sums[column] += np.bincount(group, weights=block[:, column], minlength=_PALETTE_LIMIT)
            used = counts > 0
            sums[:, used] /= counts[used]
            # Empty leaves must not enlarge the scalar codebook's range.
            sums[:, ~used] = sums[:, np.flatnonzero(used)[0], None]
            palette = sums.T.astype(np.float32)
            del sums, counts, block, dimensions, thresholds
        yield palette, labels
    finally:
        labels._mmap.close()
        label_path.unlink(missing_ok=True)


def write_sog_from_ply(src, dst):
    """Write a bundled SOG v2. Return (point_count, user-facing loss/limit notes).

    Preserves all complete SH bands through degree 3, with explicit lossy
    quantization. Higher degrees and incomplete bands are rejected, never dropped.
    Input and working textures are memory mapped; SH training and classification
    use bounded batches. No network, global Node, GPU or external Python wheels.
    """
    executable = _encoder()
    destination = Path(dst)
    with _input_ply(src) as values, tempfile.TemporaryDirectory(
            prefix='.gs-sog-', dir=str(destination.parent)) as temporary:
        folder = Path(temporary)
        n = len(values['x'])
        sh_names = [name for name in values if name.startswith('f_rest_')]
        try:
            sh_names.sort(key=lambda name: int(name[7:]))
        except ValueError as ex:
            raise ValueError('SOG 高阶球谐属性名称无效') from ex
        degree_for_count = {0: 0, 9: 1, 24: 2, 45: 3}
        if len(sh_names) not in degree_for_count or sh_names != [f'f_rest_{i}' for i in range(len(sh_names))]:
            raise ValueError('SOG v2 需要完整的 0–3 阶球谐；当前属性不完整或超过支持阶数，请改用 PLY')
        degree = degree_for_count[len(sh_names)]
        width = min(4096, max(4, int(math.ceil(math.sqrt(n) / 4)) * 4))
        height = (n + width - 1) // width
        if height > 16383:
            raise ValueError('点数超过单个 SOG WebP 纹理上限，请分拆对象后导出')
        files = []
        meta = {'version': 2, 'asset': {'generator': 'GS Editor'}, 'count': n}

        # 16-bit signed-log positions, split across two raw RGB textures.
        position_names = ('x', 'y', 'z')
        mins, maxs = [], []
        for name in position_names:
            lo, hi = float(np.min(values[name])), float(np.max(values[name]))
            mins.append(math.copysign(math.log1p(abs(lo)), lo))
            maxs.append(math.copysign(math.log1p(abs(hi)), hi))
        span = np.asarray(maxs) - mins
        span[span == 0] = 1
        for stem, shift in (('means_l', 0), ('means_u', 8)):
            with _image(folder, stem, width, height) as image:
                flat = image.reshape(-1, 4)
                for part in _chunks(n):
                    xyz = _stack(values, position_names, part).astype(np.float64)
                    logged = np.sign(xyz) * np.log1p(np.abs(xyz))
                    codes = np.clip(np.rint((logged - mins) / span * 65535), 0, 65535).astype(np.uint16)
                    flat[part, :3] = ((codes >> shift) & 255).astype(np.uint8)
                files.append(_encode(folder, stem, image, executable))
        meta['means'] = {'mins': mins, 'maxs': maxs, 'files': files[:2]}

        zero_quats = 0
        with _image(folder, 'quats', width, height) as image:
            flat = image.reshape(-1, 4)
            for part in _chunks(n):
                quat = _stack(values, [f'rot_{i}' for i in range(4)], part).astype(np.float64)
                length = np.linalg.norm(quat, axis=1)
                invalid = length == 0
                zero_quats += int(np.count_nonzero(invalid))
                quat[invalid] = (1, 0, 0, 0)
                length[invalid] = 1
                quat /= length[:, None]
                largest = np.argmax(np.abs(quat), axis=1)
                quat *= np.where(quat[np.arange(len(quat)), largest] < 0, -1, 1)[:, None]
                kept = quat[np.arange(4)[None, :] != largest[:, None]].reshape(-1, 3)
                flat[part, :3] = np.clip(np.rint((kept * math.sqrt(2) + 1) * 127.5), 0, 255).astype(np.uint8)
                flat[part, 3] = largest + 252
            files.append(_encode(folder, 'quats', image, executable))
        meta['quats'] = {'files': ['quats.webp']}

        for stem, names in (('scales', [f'scale_{i}' for i in range(3)]),
                            ('sh0', [f'f_dc_{i}' for i in range(3)])):
            book = _book([values[name] for name in names])
            with _image(folder, stem, width, height) as image:
                flat = image.reshape(-1, 4)
                for part in _chunks(n):
                    flat[part, :3] = _quantize(_stack(values, names, part), book)
                    if stem == 'sh0':
                        logits = values['opacity'][part].astype(np.float64)
                        opacity = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
                        flat[part, 3] = np.rint(opacity * 255).astype(np.uint8)
                files.append(_encode(folder, stem, image, executable))
            meta[stem] = {'codebook': book.tolist(), 'files': [stem + '.webp']}

        notes = ['SOG v2 使用有损量化，保留完整的球谐阶数']
        if degree:
            coefficients = len(sh_names) // 3
            with _sh_palette(values, sh_names, n, folder) as (palette, labels):
                palette_count = len(palette)
                book = _book([palette])
                with _image(folder, 'shN_centroids', 64 * coefficients,
                            (palette_count + 63) // 64) as image:
                    rgb = palette.reshape(palette_count, 3, coefficients).transpose(0, 2, 1)
                    image.reshape(-1, 4)[:palette_count * coefficients, :3] = _quantize(rgb, book).reshape(-1, 3)
                    files.append(_encode(folder, 'shN_centroids', image, executable))
                with _image(folder, 'shN_labels', width, height) as image:
                    flat = image.reshape(-1, 4)
                    for part in _chunks(n):
                        flat[part, 0] = (labels[part] & 255).astype(np.uint8)
                        flat[part, 1] = (labels[part] >> 8).astype(np.uint8)
                    files.append(_encode(folder, 'shN_labels', image, executable))
                meta['shN'] = {'count': palette_count, 'bands': degree, 'codebook': book.tolist(),
                               'files': ['shN_centroids.webp', 'shN_labels.webp']}
            if n > _PALETTE_LIMIT:
                notes.append('高阶球谐已量化到 65,536 项调色盘')
        if zero_quats:
            notes.append(f'{zero_quats:,} 个零旋转四元数已替换为单位旋转')

        archive = folder / 'output.sog'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as bundle:
            bundle.writestr('meta.json', json.dumps(meta, ensure_ascii=True, separators=(',', ':'), allow_nan=False))
            for name in files:
                bundle.write(folder / name, arcname=name)
        os.replace(archive, destination)
    return n, notes
