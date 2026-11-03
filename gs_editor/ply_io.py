"""Write a Blender splat point cloud back out as a standard 3DGS PLY.

Blender 5.3's own PLY exporter only writes mesh vertices, so a point cloud comes
out empty. This writes the INRIA layout that Brush, SuperSplat, nerfstudio and
Blender's importer all read.

A PLY has no scene graph, so the object's own place in the world (its location,
rotation and scale) has to be baked into the values it writes: otherwise a moved
object exports at its old coordinates. `write_pointcloud_ply(matrix=...)` does
that on the way out -- centres through the matrix, each splat's rotation turned
with it, its size scaled, its view-dependent colour rotated to keep facing the
way it did -- while the object itself is left untouched.

The file is filled one attribute at a time through a memory map, so exporting a
multi-million-splat scan needs only one attribute's worth of extra memory, not
several copies of the whole cloud.
"""

import numpy as np

from .core import _SH_BANDS, quat_mul, sh_rotation_matrices
from .merge import decompose, quat_from_matrix
from .splats import SplatSet, BASE_ATTR, HIDDEN_ATTR, _LAYOUT


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


class _World:
    """An object's place in the world, ready to be baked into the exported values.

    A rotation is applied to each splat's own rotation and to the higher SH bands,
    a uniform scale to its size, the translation to its centre. A non-uniform
    scale and a mirror cannot be expressed in a single splat's rotation and
    scale, so they are approximated the same way merging objects does.
    """

    def __init__(self, M):
        self.M = np.asarray(M, np.float64)
        self.R, self.s, self.t, _, _ = decompose(self.M)
        self.identity = bool(np.allclose(self.M, np.eye(4), rtol=0.0, atol=1e-7))
        self.rotating = not np.allclose(self.R, np.eye(3), rtol=0.0, atol=1e-7)
        self.q = quat_from_matrix(self.R) if self.rotating else None
        self.bands = sh_rotation_matrices(self.R) if self.rotating else None

    def position(self, p):
        if self.identity:                    # untouched object: write the values as they are
            return p
        return (p.astype(np.float64) @ (self.R * self.s).T + self.t).astype(np.float32)

    def scale(self, s):
        if self.identity:
            return s
        return (s * self.s).astype(np.float32)

    def rotation(self, q):
        return quat_mul(self.q, q)


def _write_sh_rotated(rows, put, sh_names, bands, k):
    """f_rest with each SH band turned by the object's rotation (a band mixes within itself)."""
    done = 0
    for (lo, hi), Mb in zip(_SH_BANDS, bands):
        if hi > k:
            break
        block = np.stack([rows(sh_names[i]) for i in range(lo, hi)], axis=1)
        block = np.einsum('km,nmc->nkc', Mb.astype(np.float32), block.astype(np.float32))
        for i in range(lo, hi):
            for ch in range(3):
                put(f"f_rest_{ch * k + i}", block[:, i - lo, ch])
        del block
        done = hi
    for i in range(done, k):                 # a band that was not a full set stays as it is
        sh = rows(sh_names[i])
        for ch in range(3):
            put(f"f_rest_{ch * k + i}", sh[:, ch])
        del sh


def _write(path, n_total, keep, get, sh_names, matrix=None):
    """keep: bool mask over the n_total splats (or None for all).
    get(name) -> (n_total, width) array for an attribute, or None if absent.
    matrix: the object's world matrix to bake in, or None to write object space."""
    n = n_total if keep is None else int(np.count_nonzero(keep))
    k = len(sh_names)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * k)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    col = {nm: i for i, nm in enumerate(names)}
    world = _World(matrix) if matrix is not None else None

    header = ["ply", "format binary_little_endian 1.0",
              "comment Exported from Blender by GS Editor",
              f"element vertex {n}"]
    header += [f"property float {nm}" for nm in names]
    header.append("end_header")
    head = ("\n".join(header) + "\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(head)
        f.truncate(len(head) + n * len(names) * 4)
    if n == 0:
        return 0
    out = np.memmap(path, dtype="<f4", mode="r+", offset=len(head), shape=(n, len(names)))

    def put(name, values):
        out[:, col[name]] = values

    def rows(name):
        a = get(name)
        if a is None:
            return None
        return a if keep is None else a[keep]

    pos = rows("position")
    if pos is not None and world is not None:
        pos = world.position(pos)
    for i, a in enumerate("xyz"):
        put(a, pos[:, i])
    del pos
    for a in ("nx", "ny", "nz"):
        put(a, 0.0)

    base = rows(BASE_ATTR)
    if base is not None:
        for i in range(3):
            put(f"f_dc_{i}", base[:, i])
        put("opacity", _logit(base[:, 3]))
    else:
        for i in range(3):
            put(f"f_dc_{i}", 0.0)
        put("opacity", _logit(np.ones(n, np.float32)))
    del base

    # f_rest is channel-major: all red coefficients, then green, then blue.
    if world is not None and world.rotating and sh_names:
        _write_sh_rotated(rows, put, sh_names, world.bands, k)
    else:
        for j, name in enumerate(sh_names):
            sh = rows(name)
            for ch in range(3):
                put(f"f_rest_{ch * k + j}", sh[:, ch])
            del sh

    scale = rows("scale")
    if scale is not None and world is not None:
        scale = world.scale(scale)
    for i in range(3):
        put(f"scale_{i}", np.log(np.maximum(scale[:, i], 1e-12)) if scale is not None else np.log(0.01))
    del scale

    rot = rows("rotation")
    if rot is not None and world is not None and world.rotating:
        rot = world.rotation(rot)
    for i in range(4):
        put(f"rot_{i}", rot[:, i] if rot is not None else (1.0 if i == 0 else 0.0))
    del rot

    out.flush()
    del out
    return n


def _sh_sorted(names):
    names = [k for k in names if k.startswith("radiance:sh_")]
    return sorted(names, key=lambda k: int(k.rsplit('_', 1)[1]))


def write_splat_ply(path, S: SplatSet, matrix=None):
    """Write a SplatSet (all of its rows), optionally with a world matrix baked in."""
    return _write(path, S.n, None, S.get, S.sh_names(), matrix)


def write_pointcloud_ply(path, pc, matrix=None):
    """Write a Blender point cloud straight from its attributes, leaving out erased splats.

    matrix: the object's world matrix, so the file holds where the splats are
    seen rather than where they sit in object space. None keeps object space."""
    n = len(pc.points)

    def get(name):
        attr = pc.attributes.get(name)
        if attr is None or attr.domain != 'POINT':
            return None
        width, key, dtype = _LAYOUT[attr.data_type]
        buf = np.empty(n * width, dtype)
        if n:
            attr.data.foreach_get(key, buf)
        return buf.reshape(n, width)

    hidden = get(HIDDEN_ATTR)
    keep = None if hidden is None or not hidden.any() else ~hidden[:, 0]
    return _write(path, n, keep, get, _sh_sorted([a.name for a in pc.attributes]), matrix)
