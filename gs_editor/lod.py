"""Spatially partition a frozen, oriented PLY and build coarser Gaussian LODs.

LOD0 keeps every input primitive. Later levels merge spatial neighbours by
moment matching (covariance includes the centres' spread), with alpha/area
weighted colour and SH. This is an offline approximation, not retraining.
Only a single spatial block and bounded covariance batches are held in RAM.
"""

import math
from pathlib import Path

import numpy as np

from .binary_formats import _input_ply, _alpha

_BLOCK = 32768


def _header(file, names, count):
    text = ['ply', 'format binary_little_endian 1.0', f'element vertex {count}']
    text.extend('property float ' + name for name in names)
    file.write(('\n'.join(text) + '\nend_header\n').encode('ascii'))


def _extend_bounds(bounds, rows):
    # A sphere enclosing 3 sigma in every orientation. Conservative even for
    # highly elongated splats; used for LCC2's node-level frustum culling.
    xyz = np.column_stack([rows[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    logs = np.column_stack([rows[f'scale_{k}'] for k in range(3)]).astype(np.float64)
    radius = 3.0 * np.exp(logs.max(axis=1))
    if not np.isfinite(radius).all():
        raise ValueError('高斯尺寸过大，无法生成 LOD 边界')
    bounds[0] = np.minimum(bounds[0], (xyz - radius[:, None]).min(axis=0))
    bounds[1] = np.maximum(bounds[1], (xyz + radius[:, None]).max(axis=0))


def _empty_bounds():
    return np.array([[np.inf] * 3, [-np.inf] * 3], dtype=np.float64)


def _write_subset(path, values, indices):
    names = list(values)
    dtype = np.dtype([(name, '<f4') for name in names])
    bounds = _empty_bounds()
    with open(path, 'wb') as file:
        _header(file, names, len(indices))
        for start in range(0, len(indices), _BLOCK):
            ids = indices[start:start + _BLOCK]
            rows = np.empty(len(ids), dtype=dtype)
            for name in names:
                rows[name] = values[name][ids]
            _extend_bounds(bounds, rows)
            file.write(rows.tobytes())
    return bounds


def _spread(v):
    """Spread a 16-bit integer across every third bit of uint64."""
    v = v.astype(np.uint64)
    v = (v | (v << 32)) & np.uint64(0x1f00000000ffff)
    v = (v | (v << 16)) & np.uint64(0x1f0000ff0000ff)
    v = (v | (v << 8)) & np.uint64(0x100f00f00f00f00f)
    v = (v | (v << 4)) & np.uint64(0x10c30c30c30c30c3)
    return (v | (v << 2)) & np.uint64(0x1249249249249249)


def _morton_order(values):
    code = np.zeros(len(values['x']), dtype=np.uint64)
    for axis, name in enumerate(('x', 'y', 'z')):
        v = values[name].astype(np.float64)
        extent = float(v.max() - v.min())
        q = np.zeros(len(v), dtype=np.uint64) if extent == 0 else (
            (v - v.min()) * (65535.0 / extent)).astype(np.uint64)
        code |= _spread(q) << axis
    return np.argsort(code, kind='stable')


def _rotation(q):
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm < 1e-20):
        raise ValueError('存在无效的零长度高斯旋转；请修复后再导出')
    w, x, y, z = (q / norm[:, None]).T
    r = np.empty((len(q), 3, 3), dtype=np.float64)
    r[:, 0, :] = np.column_stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)))
    r[:, 1, :] = np.column_stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)))
    r[:, 2, :] = np.column_stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    return r


def _quaternion(r):
    """Stable matrix to WXYZ conversion, including rotations near 180 degrees."""
    candidates = np.column_stack((1+r[:, 0, 0]+r[:, 1, 1]+r[:, 2, 2],
                                  1+r[:, 0, 0]-r[:, 1, 1]-r[:, 2, 2],
                                  1-r[:, 0, 0]+r[:, 1, 1]-r[:, 2, 2],
                                  1-r[:, 0, 0]-r[:, 1, 1]+r[:, 2, 2]))
    branch = candidates.argmax(axis=1)
    q = np.empty((len(r), 4), dtype=np.float64)
    for axis in range(4):
        take = branch == axis
        m = r[take]
        s = 2 * np.sqrt(np.maximum(candidates[take, axis], 1e-30))
        if axis == 0:
            q[take] = np.column_stack((s/4, (m[:, 2, 1]-m[:, 1, 2])/s,
                                      (m[:, 0, 2]-m[:, 2, 0])/s, (m[:, 1, 0]-m[:, 0, 1])/s))
        elif axis == 1:
            q[take] = np.column_stack(((m[:, 2, 1]-m[:, 1, 2])/s, s/4,
                                      (m[:, 0, 1]+m[:, 1, 0])/s, (m[:, 0, 2]+m[:, 2, 0])/s))
        elif axis == 2:
            q[take] = np.column_stack(((m[:, 0, 2]-m[:, 2, 0])/s,
                                      (m[:, 0, 1]+m[:, 1, 0])/s, s/4, (m[:, 1, 2]+m[:, 2, 1])/s))
        else:
            q[take] = np.column_stack(((m[:, 1, 0]-m[:, 0, 1])/s,
                                      (m[:, 0, 2]+m[:, 2, 0])/s, (m[:, 1, 2]+m[:, 2, 1])/s, s/4))
    return q / np.linalg.norm(q, axis=1)[:, None]


def _merge(values, ids, names):
    starts = np.arange(0, len(ids), 4)
    sizes = np.minimum(4, len(ids) - starts)
    group = np.repeat(np.arange(len(starts)), sizes)
    xyz = np.column_stack([values[k][ids] for k in ('x', 'y', 'z')]).astype(np.float64)
    logs = np.column_stack([values[f'scale_{k}'][ids] for k in range(3)]).astype(np.float64)
    q = np.column_stack([values[f'rot_{k}'][ids] for k in range(4)]).astype(np.float64)
    # Approximate projected area, independent of the eventual camera direction.
    area = np.exp(logs.sum(axis=1) * (2.0 / 3.0))
    mass = np.maximum(_alpha(values['opacity'][ids]) * area, 1e-30)
    total = np.add.reduceat(mass, starts)
    weights = mass / total[group]
    mean = np.add.reduceat(xyz * weights[:, None], starts, axis=0)
    r = _rotation(q)
    basis = r * np.exp(logs)[:, None, :]
    delta = xyz - mean[group]
    covariance = basis @ basis.transpose(0, 2, 1) + delta[:, :, None] * delta[:, None, :]
    covariance = np.add.reduceat(covariance * weights[:, None, None], starts, axis=0)
    eig, axes = np.linalg.eigh(covariance)
    eig = np.maximum(eig, 1e-30)
    # eigh can return a reflection. Flipping one eigenvector preserves covariance.
    axes[:, :, 0] *= np.where(np.linalg.det(axes) < 0, -1, 1)[:, None]
    log_scale = np.log(eig) * 0.5
    rotation = _quaternion(axes)
    alpha = np.clip(total / np.exp(log_scale.sum(axis=1) * (2.0 / 3.0)), 1e-6, 1-1e-6)
    out = np.empty(len(starts), dtype=[(name, '<f4') for name in names])
    for name in names:
        out[name] = np.add.reduceat(values[name][ids] * weights, starts)
    for axis, name in enumerate(('x', 'y', 'z')):
        out[name] = mean[:, axis]
        out[f'scale_{axis}'] = log_scale[:, axis]
    for axis in range(4):
        out[f'rot_{axis}'] = rotation[:, axis]
    out['opacity'] = np.log(alpha / (1-alpha))
    # A final unpaired primitive must remain unchanged across levels.
    singles = sizes == 1
    if singles.any():
        for name in names:
            out[name][singles] = values[name][ids[starts[singles]]]
    if not np.isfinite(out.view('<f4')).all():
        raise ValueError('LOD 合并结果超出数值范围，请检查点云尺寸')
    return out


def _coarsen(source, target):
    bounds = _empty_bounds()
    with _input_ply(source) as values:
        names = list(values)
        order = _morton_order(values)
        count = (len(order) + 3) // 4
        with open(target, 'wb') as file:
            _header(file, names, count)
            for start in range(0, len(order), _BLOCK):
                rows = _merge(values, order[start:start + _BLOCK], names)
                _extend_bounds(bounds, rows)
                file.write(rows.tobytes())
    return count, bounds


def _partition(values, target):
    lo = np.array([values[k].min() for k in ('x', 'y')], dtype=np.float64)
    span = np.array([values[k].max() for k in ('x', 'y')], dtype=np.float64) - lo
    number = max(1, min(1024, math.ceil(len(values['x']) / target)))
    cell_budget = min(1024, max(64, number * 4))
    previous = None
    max_log_scale = max(float(values[f'scale_{axis}'].max()) for axis in range(3))
    min_length = max(1e-8, math.exp(max_log_scale) * 6.0 / 4096.0)
    if not math.isfinite(min_length):
        raise ValueError('高斯尺寸过大，无法进行空间分块')
    if min(span) <= 1e-12:
        grid = [number if span[0] > 1e-12 else 1, number if span[1] > 1e-12 else 1]
    else:
        nx = max(1, min(number, round(math.sqrt(number * span[0] / span[1]))))
        grid = [nx, max(1, math.ceil(number / nx))]
    for attempt in range(7):
        lengths = np.maximum(span / grid, min_length)
        coord = [np.clip(((values[k].astype(np.float64)-lo[a])/lengths[a]).astype(np.int64),
                         0, grid[a]-1) for a, k in enumerate(('x', 'y'))]
        keys = coord[0] + (coord[1] << 16)
        order = np.argsort(keys, kind='stable')
        sorted_keys = keys[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1, len(order)]
        current = (lo, lengths, order, sorted_keys[cuts[:-1]], cuts)
        if len(cuts) - 1 > cell_budget and previous is not None:
            return previous
        previous = current
        if np.diff(cuts).max() <= target or attempt == 6:
            break
        # Dense or collinear clouds may not admit small XY blocks. Limit grid
        # refinement rather than loop indefinitely or drop any primitives.
        refined = [v*2 if span[a] > 1e-12 and v < 4096 else v for a, v in enumerate(grid)]
        if refined == grid:
            break
        grid = refined
    return current


def build_dataset(source, folder, levels=4, chunk_points=65536):
    """Create per-cell PLY levels; callers own and remove the temporary folder."""
    if not 2 <= levels <= 6 or not 4096 <= chunk_points <= 262144:
        raise ValueError('LOD 层数或分块点数超出支持范围')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = {'cells': [], 'levels': levels, 'lod_counts': [0] * levels, 'notes': []}
    scene_bounds = _empty_bounds()
    with _input_ply(source) as values:
        result['count'] = len(values['x'])
        origin, lengths, order, keys, cuts = _partition(values, chunk_points)
        for index, key in enumerate(keys):
            ids = order[cuts[index]:cuts[index+1]]
            if len(ids) > chunk_points and not result['notes']:
                result['notes'].append('部分密集空间块超过目标点数；已保留全部有效点')
            path = folder / f'cell-{index}-lod-0.ply'
            bounds = _write_subset(path, values, ids)
            cell = {'x': int(key & 65535), 'y': int(key >> 16),
                    'levels': [{'path': str(path), 'count': len(ids)}]}
            for level in range(1, levels):
                next_path = folder / f'cell-{index}-lod-{level}.ply'
                count, level_bounds = _coarsen(path, next_path)
                bounds[0] = np.minimum(bounds[0], level_bounds[0])
                bounds[1] = np.maximum(bounds[1], level_bounds[1])
                cell['levels'].append({'path': str(next_path), 'count': count})
                path = next_path
            cell['bounds'] = bounds.tolist()
            scene_bounds[0] = np.minimum(scene_bounds[0], bounds[0])
            scene_bounds[1] = np.maximum(scene_bounds[1], bounds[1])
            for level, info in enumerate(cell['levels']):
                result['lod_counts'][level] += info['count']
            result['cells'].append(cell)
    # LCC1 derives cell positions from boundingBox.min. Expand its origin by
    # whole cells so conservative splat bounds do not shift the grid itself.
    shift = np.maximum(0, np.ceil((origin - scene_bounds[0, :2]) / lengths)).astype(np.int64)
    if np.any(shift > 32767):
        raise ValueError('高斯尺寸相对分块过大；请增大每块目标点数或清除异常大点')
    scene_bounds[0, :2] = origin - shift * lengths
    for cell in result['cells']:
        cell['x'] += int(shift[0])
        cell['y'] += int(shift[1])
        if max(cell['x'], cell['y']) > 32767:
            raise ValueError('空间索引超出 LCC 范围，请增大每块目标点数')
        scene_bounds[1, :2] = np.maximum(scene_bounds[1, :2],
            scene_bounds[0, :2] + (np.array([cell['x'], cell['y']]) + 1) * lengths)
    result['bounds'] = scene_bounds.tolist()
    result['cell_length'] = lengths.tolist()
    return result
