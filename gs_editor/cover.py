"""覆盖：一块吸附到表面的刚性矩形面片，用来盖住泼溅点云上的区域。

框选一次做这些事：
1. 取框内的点，先按屏幕小格去掉被挡住的背景，再用「深度众数 + 迭代内点回归」
   拟合出占多数的那个面的朝向（`surface_points` / `fit_plane_robust`）；
2. 尺寸按"框多大就多大"换算（框像素 × 该处的像素尺度），也可反过来解成盖满框；
3. 建一块居中的矩形网格，物体变换 = 表面朝向的旋转 + 中心位置 + 沿法线的偏移，
   旋转与位置都存在物体变换里（Object 面板可见）；
4. 生成后进入吸附阶段（操作符里）：跟着鼠标在表面上滑，朝向取命中点的局部表面法线；
   按住 Shift 可退回手动旋转；左键确认、右键取消。

几何都是纯函数，操作符只负责取场景和交互。
"""

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Matrix, Quaternion, Vector

COVER_MAT_NAME = "覆盖材质"


# ------------------------------------------------------------------ 选点 / 拟合
def region_splats(proj, S, rect, front=1.0, min_opacity=0.02):
    """框选区域内的泼溅点索引。

    默认取框内**所有**点：朝向交给 fit_plane_robust 去找"占多数的那个面"，
    而不是先按深度砍前层——砍前层会把凸起结构当成主体（它的侧面更靠近相机）。
    front < 1 时才只保留最近的那部分点。
    """
    (x0, y0), (x1, y1) = rect
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    m = (proj.ok & (proj.sx >= x0) & (proj.sx <= x1) & (proj.sy >= y0) & (proj.sy <= y1)
         & S.visible & (S.opacity > min_opacity))
    idx = np.nonzero(m)[0]
    if len(idx) < 3 or front >= 1.0:
        return idx
    depth = proj.depth[idx]
    near = idx[depth <= np.percentile(depth, max(1.0, front * 100.0))]
    return near if len(near) >= 3 else idx


def _pca_plane(pts):
    """一次平面 PCA，中心取中位数（比均值更抗偏）。"""
    centre = np.median(pts, axis=0)
    d = pts - centre
    cov = d.T @ d
    _, v = np.linalg.eigh(cov)
    return centre, v[:, 0]


def fit_plane(points, view_dir):
    """点集的平均平面：返回 (中心, 朝相机的单位法线)。"""
    pts = np.asarray(points, np.float64)
    if len(pts) < 3:
        return pts.mean(0) if len(pts) else np.zeros(3), -np.asarray(view_dir, np.float64)
    centre, normal = _pca_plane(pts)
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        normal = -np.asarray(view_dir, np.float64)
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, view_dir) > 0.0:    # 统一朝相机
        normal = -normal
    return centre, normal


def fit_plane_robust(points, view_dir, iterations=4):
    """抗凹凸的平面拟合：只认"占多数"的那个面。

    框选里只要有凸起结构，它的**侧面**（基本正对视线）会严重带偏普通 PCA：
    所以先按深度找最密的那条"深度带"（多数面），再迭代用内点重拟合。
    纯 numpy，无随机性，同一输入永远同一结果。
    """
    pts = np.asarray(points, np.float64)
    if len(pts) < 5:
        return fit_plane(pts, view_dir)

    centre, normal = _pca_plane(pts)
    if np.linalg.norm(normal) < 1e-12 or not np.isfinite(normal).all():
        return fit_plane(pts, view_dir)
    normal = normal / np.linalg.norm(normal)

    for _ in range(iterations):
        t = (pts - centre) @ normal
        lo, hi = np.percentile(t, 2.0), np.percentile(t, 98.0)
        if hi - lo <= 1e-12:
            break
        bins = 24
        hist, edges = np.histogram(t, bins=bins, range=(lo, hi))
        band_idx = int(np.argmax(hist))
        band_lo, band_hi = float(edges[band_idx]), float(edges[band_idx + 1])
        width = band_hi - band_lo
        band = np.abs(t - 0.5 * (band_lo + band_hi)) <= width * 1.5
        if band.sum() < max(5, int(0.2 * len(pts))):
            band = np.ones(len(pts), bool)

        tt = t[band]
        med = float(np.median(tt))
        mad = 1.4826 * float(np.median(np.abs(tt - med))) + 1e-9
        inliers = band & (np.abs(t - med) < 3.0 * mad)
        if inliers.sum() < max(5, int(0.15 * len(pts))):
            inliers = band

        new_centre, new_normal = _pca_plane(pts[inliers])
        length = np.linalg.norm(new_normal)
        if length < 1e-12 or not np.isfinite(new_normal).all():
            break
        moved = float(np.linalg.norm(new_normal / length - normal))
        centre, normal = new_centre, new_normal / length
        if moved < 1e-4:
            break

    if np.dot(normal, view_dir) > 0.0:
        normal = -normal
    return centre, normal


def blend_normal(surface_normal, view_dir, fit):
    """贴合度：1 = 完全跟表面，0 = 正对屏幕。"""
    fit = float(min(1.0, max(0.0, fit)))
    view_normal = -np.asarray(view_dir, np.float64)
    n = surface_normal * fit + view_normal * (1.0 - fit)
    length = np.linalg.norm(n)
    if length < 1e-9:
        return view_normal / np.linalg.norm(view_normal)
    return n / length


def surface_points(proj, S, idx, cell_px=6.0):
    """框内点里只留"从这个视角看得见的那个面"。

    泼溅点云没有遮挡关系：框选一块屏幕区域，除了表面本身，还会框到被它挡住的背景
    （同屏幕位置、深度远得多）。做法是按屏幕小格取最近深度，得到"每格最前面"的深度，
    再取这些深度的众数带 —— 多数格子属于当前表面，背景只在表面缺口的格子里露头。
    这样拟合的是看得见的面，而不是表面 + 背景的混合。
    """
    if len(idx) < 12:
        return idx
    d = np.asarray(proj.depth[idx], np.float64)
    ix = np.floor(np.asarray(proj.sx[idx], np.float64) / float(cell_px)).astype(np.int64)
    iy = np.floor(np.asarray(proj.sy[idx], np.float64) / float(cell_px)).astype(np.int64)
    if len(np.unique(ix)) < 2 or len(np.unique(iy)) < 2:
        return idx
    key = (ix - ix.min()) * (iy.max() - iy.min() + 1) + (iy - iy.min())
    order = np.lexsort((d, key))
    k_sorted, d_sorted = key[order], d[order]
    first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
    cell_front = d_sorted[first]
    if len(cell_front) < 6:
        return idx

    lo, hi = float(np.percentile(cell_front, 2.0)), float(np.percentile(cell_front, 98.0))
    if hi - lo <= 1e-9:
        return idx

    # 把"每格最近深度"按间隙切成若干层，取格子最多的那一层 —— 就是看得见的主表面。
    # 比固定宽度的众数带稳：表面本身有起伏时不会被切碎，远处的背景也不会混进来。
    cf = np.sort(np.asarray(cell_front, np.float64))
    if len(cf) < 8:
        return idx
    diffs = np.diff(cf)
    base = float(np.median(diffs))
    thr = max(base * 6.0, (hi - lo) * 0.02, 1e-9)
    segs = np.split(cf, np.nonzero(diffs > thr)[0] + 1)
    main = max(segs, key=len)
    lo_c, hi_c = float(main[0]), float(main[-1])
    margin = max((hi_c - lo_c) * 0.15, thr)
    keep = (d >= lo_c - margin) & (d <= hi_c + margin)
    return idx[keep] if int(keep.sum()) >= 8 else idx


COLOUR_BINS = 12          # 主色统计的量化级数（12³ 个桶）


def dominant_colour(S, idx, bins=COLOUR_BINS):
    """给定一批泼溅点，取占比最大的那个颜色（量化分桶 + 桶内加权平均）。

    直接求平均会把"白墙 + 一点蓝标签"混成脏灰蓝，所以先找主流色。
    """
    if len(idx) == 0:
        return (0.78, 0.78, 0.78)
    rgb = np.clip(S.rgb(idx), 0.0, 1.0).astype(np.float64)
    weight = np.clip(S.opacity[idx], 0.05, 1.0).astype(np.float64)
    q = np.clip((rgb * bins).astype(np.int64), 0, bins - 1)
    keys = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    sums = np.bincount(keys, weights=weight, minlength=bins ** 3)
    best = int(np.argmax(sums))
    sel = keys == best
    total = float(weight[sel].sum())
    if total <= 0.0:
        return (0.78, 0.78, 0.78)
    colour = (rgb[sel] * weight[sel, None]).sum(0) / total
    return tuple(float(c) for c in np.clip(colour, 0.0, 1.0))


def ring_colour(proj, S, rect, ring=0.35):
    """取框选**外面一圈**里占比最大的颜色，用来让面片和周围表面一致。

    取样环是"甜甜圈"：跳过紧贴框边的那一小段（那里往往还是要盖的东西本身，
    比如标签溢出来的颜色），只取更外一圈，所以不容易被框内内容带偏。
    ring 是外扩比例（相对框的长宽）。
    圈里点太少（框太贴边、周围是空的）就退回框内与紧邻带的主流色。
    """
    (x0, y0), (x1, y1) = rect
    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    w, h = xb - xa, yb - ya
    mx = max(w * float(ring), 2.0)
    my = max(h * float(ring), 2.0)
    gap_x = w * float(ring) * 0.4
    gap_y = h * float(ring) * 0.4

    base = proj.ok & S.visible & (S.opacity > 0.05)
    inner = (proj.sx >= xa) & (proj.sx <= xb) & (proj.sy >= ya) & (proj.sy <= yb)
    core = ((proj.sx >= xa - gap_x) & (proj.sx <= xb + gap_x)
            & (proj.sy >= ya - gap_y) & (proj.sy <= yb + gap_y))
    outer = ((proj.sx >= xa - mx) & (proj.sx <= xb + mx)
             & (proj.sy >= ya - my) & (proj.sy <= yb + my))
    band = np.nonzero(base & outer & ~core)[0]
    if len(band) < 20:                       # 外面不够采样就退回框内
        band = np.nonzero(base & inner)[0]
    return dominant_colour(S, band)


# ------------------------------------------------------------------ 屏幕 -> 世界
def screen_corners(rect):
    """框选矩形的四角（屏幕像素），顺序：左上、右上、右下、左下。"""
    (x0, y0), (x1, y1) = rect
    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    return (xa, ya), (xb, ya), (xb, yb), (xa, yb)


def ray_plane_hit(region, rv3d, x, y, plane_point, plane_normal):
    """屏幕点射出的视线与该平面的交点（世界空间），平行或落在身后返回 None。"""
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
    denom = direction.dot(Vector(plane_normal))
    if abs(denom) < 1e-9:
        return None
    t = (Vector(plane_point) - origin).dot(Vector(plane_normal)) / denom
    if t <= 0.0:
        return None
    return origin + direction * t


def quad_corners(region, rv3d, rect, plane_point, plane_normal):
    """把矩形四角投到平面上的世界坐标；有任何一角失败就返回 None。"""
    out = []
    for x, y in screen_corners(rect):
        hit = ray_plane_hit(region, rv3d, x, y, plane_point, plane_normal)
        if hit is None:
            return None
        out.append(hit)
    return out


def _world_per_pixel(region, rv3d, rect, world_point):
    """在给定世界点所在的深度上，屏幕上 1 像素约等于多少世界单位。"""
    (x0, y0), (x1, y1) = rect
    cx = 0.5 * (float(x0) + float(x1))
    cy = 0.5 * (float(y0) + float(y1))
    try:
        a = view3d_utils.region_2d_to_location_3d(region, rv3d, (cx, cy), world_point)
        b = view3d_utils.region_2d_to_location_3d(region, rv3d, (cx + 1.0, cy), world_point)
    except Exception:                       # noqa: BLE001
        return 0.0
    return float((a - b).length)


# ------------------------------------------------------------------ 网格
def grid_counts(rect, subdiv):
    """细分 -> (cols, rows)：扁长的框横着切，竖长的框竖着切。"""
    segs = max(1, int(subdiv) + 1)
    (x0, y0), (x1, y1) = rect
    if abs(x1 - x0) >= abs(y1 - y0):
        return segs, 1
    return 1, segs


def grid_faces(cols, rows):
    """规则网格的四边面（顶点按行优先编号）。"""
    faces = []
    for j in range(rows):
        for i in range(cols):
            a = j * (cols + 1) + i
            faces.append((a, a + 1, a + cols + 2, a + cols + 1))
    return faces


def rect_mesh(width, height, cols, rows):
    """以原点为中心的矩形网格（局部 XY 平面），法线朝 +Z。"""
    verts = []
    for j in range(rows + 1):
        y = -height * 0.5 + height * j / rows
        for i in range(cols + 1):
            x = -width * 0.5 + width * i / cols
            verts.append((x, y, 0.0))
    return verts, grid_faces(cols, rows)


def basis_from_normal(normal_world, rv3d):
    """面内两根轴：u 尽量对齐屏幕向右，v = n × u；u × v = n（法线朝相机）。

    这样面片自身的"旋转"就完整描述在物体变换里：+X 沿框的宽、+Y 沿框的高、+Z 是表面朝向。
    """
    n = Vector(normal_world).normalized()
    to_world = rv3d.view_matrix.inverted().to_3x3()
    right = (to_world @ Vector((1.0, 0.0, 0.0))).normalized()
    up = (to_world @ Vector((0.0, 1.0, 0.0))).normalized()
    u = right - n * right.dot(n)
    if u.length < 1e-6:
        u = up - n * up.dot(n)
    u.normalize()
    v = n.cross(u)
    return u, v, n


def patch_matrix(centre_world, normal_world, rv3d):
    """面片物体的变换：旋转由表面朝向 + 屏幕方向决定，平移放在面片中心。"""
    u, v, n = basis_from_normal(normal_world, rv3d)
    rot = Matrix(((u.x, v.x, n.x),
                  (u.y, v.y, n.y),
                  (u.z, v.z, n.z))).to_4x4()
    rot.translation = Vector(centre_world)
    return rot


# ------------------------------------------------------------------ 手动对位
ROTATE_SENSITIVITY = 0.010          # 每像素转多少弧度（约 0.57°/px）


def rotation_step(quat, dx, dy, view_matrix, sensitivity=ROTATE_SENSITIVITY):
    """屏幕位移 -> 一次绕屏幕横/竖轴的旋转，累加进四元数。

    横拖 = 绕屏幕竖轴转，竖拖 = 绕屏幕横轴转，手感像"捏着一张卡片拧"：
    面片还是那个面片，尺寸和中心都不变，只是朝向变了。
    """
    to_world = view_matrix.inverted().to_3x3()
    up = (to_world @ Vector((0.0, 1.0, 0.0))).normalized()
    right = (to_world @ Vector((1.0, 0.0, 0.0))).normalized()
    step = Quaternion(up, -dx * sensitivity) @ Quaternion(right, -dy * sensitivity)
    return (step @ quat).normalized()


def rotated_matrix(base_matrix, quat):
    """绕面片自身中心旋转后的变换（中心与尺寸不变）。"""
    centre = base_matrix.translation.copy()
    rot = quat.to_matrix().to_4x4()
    return Matrix.Translation(centre) @ rot @ Matrix.Translation(-centre) @ base_matrix


def shifted_matrix(base_matrix, distance):
    """沿面片自身法线平移，用来把面片往前往后挪。"""
    matrix = base_matrix.copy()
    axis = matrix.to_3x3().col[2].normalized()
    matrix.translation = matrix.translation + axis * distance
    return matrix


# ------------------------------------------------------------------ 材质 / 物体
def cover_material(colour):
    """每块面片一份自己的材质，改一块不会连带别的地块。"""
    mat = bpy.data.materials.new(COVER_MAT_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if out is None:
            out = nt.nodes.new('ShaderNodeOutputMaterial')
        nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

    def sock(name, value):
        s = bsdf.inputs.get(name)
        if s is not None:
            s.default_value = value
    sock('Base Color', (colour[0], colour[1], colour[2], 1.0))
    sock('Roughness', 1.0)
    sock('Metallic', 0.0)
    sock('Specular IOR Level', 0.15)      # 4.x / 5.x
    sock('Specular', 0.15)                # 旧版命名
    mat.diffuse_color = (colour[0], colour[1], colour[2], 1.0)
    mat.roughness = 1.0
    mat.use_backface_culling = False      # 两面都画，从背面也挡得住
    return mat


def make_patch(context, splat_obj, verts, faces, colour, matrix=None, name="覆盖面片"):
    """建一块独立的面片物体：不带父子关系，旋转与位置存在物体变换里。"""
    mesh = bpy.data.meshes.new(name + "网格")
    mesh.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    mesh.validate()
    mesh.update()

    ob = bpy.data.objects.new(name, mesh)
    collection = (splat_obj.users_collection[0] if splat_obj.users_collection
                  else context.scene.collection)
    collection.objects.link(ob)
    ob.matrix_world = matrix if matrix is not None else Matrix.Identity(4)

    mesh.materials.clear()
    mesh.materials.append(cover_material(colour))
    return ob


def project_points(region, rv3d, points):
    """世界坐标 -> 区域像素坐标（和视图里看到的一致）。"""
    P = np.array(rv3d.perspective_matrix, np.float64)
    pts = np.asarray(points, np.float64)
    clip = pts @ P[:, :3].T + P[:, 3]
    w = clip[:, 3]
    ok = w > 1e-6
    w = np.where(ok, w, 1.0)
    sx = (clip[:, 0] / w * 0.5 + 0.5) * float(region.width)
    sy = (clip[:, 1] / w * 0.5 + 0.5) * float(region.height)
    return sx, sy, ok


def fit_rect_to_frame(region, rv3d, centre, u, v, width, height, rect, iterations=6):
    """调矩形尺寸和位置，让它的**屏幕投影**正好盖上框选的矩形。

    斜着看表面时，射线打在平面上得到的是梯形，按边长算出来的矩形必然对不上；
    所以反过来迭代：投出像素范围 -> 和框选的宽高比 -> 缩放/平移 -> 再来一遍。
    返回 (width, height, centre, 参考像素尺寸)。
    """
    (x0, y0), (x1, y1) = rect
    xa, xb = sorted((float(x0), float(x1)))
    ya, yb = sorted((float(y0), float(y1)))
    target_w, target_h = xb - xa, yb - ya
    target_cx, target_cy = 0.5 * (xa + xb), 0.5 * (ya + yb)
    centre = np.asarray(centre, np.float64)
    if target_w <= 0 or target_h <= 0:
        return width, height, centre, (target_w, target_h)

    w, h = float(width), float(height)
    for _ in range(iterations):
        corners = np.array([
            centre + u * (-w * 0.5) + v * (-h * 0.5),
            centre + u * (w * 0.5) + v * (-h * 0.5),
            centre + u * (w * 0.5) + v * (h * 0.5),
            centre + u * (-w * 0.5) + v * (h * 0.5),
        ])
        sx, sy, ok = project_points(region, rv3d, corners)
        if not bool(ok.all()):
            break
        meas_w = float(sx.max() - sx.min())
        meas_h = float(sy.max() - sy.min())
        if meas_w <= 1e-6 or meas_h <= 1e-6:
            break
        cx, cy = 0.5 * (float(sx.min()) + float(sx.max())), 0.5 * (float(sy.min()) + float(sy.max()))
        ratio_w = target_w / meas_w
        ratio_h = target_h / meas_h
        if abs(ratio_w - 1.0) < 0.005 and abs(ratio_h - 1.0) < 0.005 and \
                abs(cx - target_cx) < 0.5 and abs(cy - target_cy) < 0.5:
            break
        w *= ratio_w
        h *= ratio_h
        # 把中心按像素差挪回去（用该处的像素尺度换算成世界单位）
        wpp = _world_per_pixel(region, rv3d, rect, centre)
        if wpp > 0.0:
            centre = centre + u * ((target_cx - cx) * wpp) + v * ((target_cy - cy) * wpp)
    return w, h, centre, (target_w, target_h)


# ------------------------------------------------------------------ 组装
def build_patch(context, splat_obj, S, proj, region, rv3d, rect, view_dir, fit, offset,
                subdiv, colour, ring=0.35, size_from_screen=False):
    """框选 -> 一块**刚性矩形**面片，整块旋转对齐到选中表面的朝向。

    尺寸默认"框多大就多大"：按该处的像素尺度换算成世界尺寸，斜看时投影自然变短（透视，不是错）。
    `size_from_screen=True` 时反过来解，让投影正好盖满框选矩形。
    """
    idx_all = region_splats(proj, S, rect)
    if len(idx_all) < 3:
        return None, "框选范围内没有足够的泼溅点，请对准要覆盖的表面"
    idx = surface_points(proj, S, idx_all)          # 去掉被挡住的背景点

    mw = splat_obj.matrix_world
    rot = np.array(mw.to_3x3(), np.float64)
    view_dir_world = rot @ np.asarray(view_dir, np.float64)
    view_dir_world /= max(np.linalg.norm(view_dir_world), 1e-12)

    centre_local, surface_normal_local = fit_plane_robust(S.pos[idx], view_dir)
    # 视线和表面几乎平行时，画出来的面片必然侧成一条、盖不住任何东西
    facing = abs(float(np.dot(surface_normal_local, view_dir)))
    if facing < 0.06:
        return None, "视线几乎与表面平行，换个角度再框"
    normal_local = blend_normal(surface_normal_local, view_dir, fit)
    normal_world = rot @ np.asarray(normal_local, np.float64)
    normal_world /= max(np.linalg.norm(normal_world), 1e-12)
    plane_point_world = mw @ Vector(centre_local)

    # 框选四角投到那块平面上：只用来决定矩形的尺寸和中心
    corners = quad_corners(region, rv3d, rect, plane_point_world,
                           Vector(normal_world).normalized())
    if corners is None:
        return None, "无法把框选投到表面上，请换个角度或缩小范围"
    c0, c1, c2, c3 = [np.asarray(c, np.float64) for c in corners]
    width = 0.5 * (float(np.linalg.norm(c1 - c0)) + float(np.linalg.norm(c2 - c3)))
    height = 0.5 * (float(np.linalg.norm(c3 - c0)) + float(np.linalg.norm(c2 - c1)))
    if not (np.isfinite(width) and np.isfinite(height)) or min(width, height) < 1e-6:
        return None, "面片尺寸无效，请重新框选"

    centre_world = (c0 + c1 + c2 + c3) * 0.25
    (x0, y0), (x1, y1) = rect
    px_w, px_h = abs(float(x1) - float(x0)), abs(float(y1) - float(y0))

    # 尺寸：默认"框多大就多大"——按该处的像素尺度换成世界尺寸，不为了盖满屏幕而放大。
    # 斜着看表面时，面片的屏幕投影自然会短一截，那是透视，不是尺寸错。
    hit_centre = ray_plane_hit(region, rv3d, 0.5 * (float(x0) + float(x1)),
                               0.5 * (float(y0) + float(y1)), plane_point_world,
                               Vector(normal_world).normalized())
    if hit_centre is not None:
        centre_world = np.array(hit_centre, np.float64)
    wpp = _world_per_pixel(region, rv3d, rect, centre_world)
    if wpp <= 0.0:
        return None, "无法把框选投到表面上，请换个角度或缩小范围"
    width = px_w * wpp
    height = px_h * wpp
    if not (np.isfinite(width) and np.isfinite(height)) or min(width, height) < 1e-9:
        return None, "面片尺寸无效，请重新框选"

    if size_from_screen:
        # 可选：反过来解，让面片的屏幕投影正好盖上框选矩形（斜视时面片会被放大）
        u_axis, v_axis, _n_axis = basis_from_normal(normal_world, rv3d)
        width, height, centre_world, _ = fit_rect_to_frame(
            region, rv3d, centre_world, np.array(u_axis), np.array(v_axis),
            width, height, rect)
        if not (np.isfinite(width) and np.isfinite(height)) or min(width, height) < 1e-9:
            return None, "面片尺寸无效，请重新框选"
        limit = 8.0      # 71° 左右的斜视会自然放大 3 倍上下，留出余量
        if (px_w > 0 and width > px_w * wpp * limit) or (px_h > 0 and height > px_h * wpp * limit):
            return None, "这个角度太贴近表面，面片会被拉飞；换个角度再框"

    size = max(width, height)
    shift = float(offset) if offset > 0.0 else size * 0.006      # 往相机方向让开一点，防穿插
    centre_world = centre_world + np.asarray(normal_world, np.float64) * shift

    cols, rows = grid_counts(rect, subdiv)
    verts, faces = rect_mesh(width, height, cols, rows)
    matrix = patch_matrix(centre_world, normal_world, rv3d)
    if colour is None:
        colour = ring_colour(proj, S, rect, ring)
    ob = make_patch(context, splat_obj, verts, faces, colour, matrix)
    return ob, None
