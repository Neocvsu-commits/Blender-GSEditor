# -*- coding: utf-8 -*-
"""在 Blender 里跑 GS Editor 的自检。

用法：
    set GSE_ADDON_PARENT=<gs_editor 上级目录>
    blender -b --factory-startup --python scripts/verify.py

检查：注册、各面板在三种工具下的 draw、覆盖几何（拟合/网格/选点/颜色）、
面片物体与材质创建、旧工具确实已移除、擦除路径仍可用、更新检查与覆盖安装。
交互式拖动那部分需要真实视口，这里只测它调用的纯函数。
"""

import json
import os
import pathlib
import shutil
import sys
import tempfile
import zipfile

import bpy
import numpy as np
from mathutils import Vector

PARENT = os.environ.get("GSE_ADDON_PARENT")
if not PARENT:
    raise SystemExit("缺少环境变量 GSE_ADDON_PARENT")
sys.path.insert(0, PARENT)

import gs_editor                      # noqa: E402
from gs_editor import cover, core, ops, ui, update_checker  # noqa: E402
from gs_editor.splats import SplatSet  # noqa: E402

OUT = {"tests": [], "panels": {}}


def check(name, ok, detail=""):
    OUT["tests"].append({"name": name, "ok": bool(ok), "detail": str(detail)})


# ----------------------------------------------------------------- 假界面
class Layout:
    def __init__(self, log):
        self.log = log
        self.scale_y = 1.0
        self.use_property_split = False

    def _child(self):
        return Layout(self.log)

    def label(self, text="", **kw):
        self.log.append(("label", text))

    def operator(self, idname, text=None, **kw):
        self.log.append(("operator", idname, text, kw))
        return Layout(self.log)

    def prop(self, data, name, text=None, **kw):
        self.log.append(("prop", name, text, kw))

    def box(self):
        self.log.append(("box", ""))
        return Layout(self.log)

    def column(self, **kw):
        return self._child()

    def row(self, **kw):
        return self._child()

    def split(self, **kw):
        return self._child()

    def grid_flow(self, **kw):
        return self._child()

    def separator(self, **kw):
        pass


class FakeSelf:
    def __init__(self, log, cls=None):
        self.layout = Layout(log)
        self.cls = cls

    def draw_params(self, context, col):
        """面板把参数画在 draw_params 里，测试时转给真正的类方法。"""
        return self.cls.draw_params(self, context, col)


def panel_self(cls, log):
    return FakeSelf(log, cls)


class FakeContext:
    def __init__(self, scene, obj):
        self.scene = scene
        self.active_object = obj


def make_cloud(name="测试点云", n=200):
    """合成一块竖直平面上的点云（z 约等于 0 的小抖动），左右两半颜色不同。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    rng = np.random.default_rng(7)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = rng.uniform(-1.0, 1.0, n)
    pos[:, 1] = rng.uniform(-1.0, 1.0, n)
    pos[:, 2] = rng.normal(0.0, 0.002, n)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    rgba = np.tile(np.array([0.5, 0.35, 0.3, 0.9], np.float32), (n, 1))
    rgba[pos[:, 0] < 0.0] = (0.8, 0.1, 0.1, 0.9)      # 左半红
    rgba[pos[:, 0] > 0.0] = (0.1, 0.1, 0.8, 0.9)      # 右半蓝
    base.data.foreach_set("color", rgba.ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def draw_panels(obj):
    scene = bpy.context.scene
    for tool in ('ERASE', 'RESTORE', 'COVER'):
        scene.gse.tool = tool
        for cls in (ui.GSE_PT_panel, ui.GSE_PT_brush, ui.GSE_PT_cover,
                    ui.GSE_PT_shortcuts, ui.GSE_PT_export):
            log = []
            cls.draw(panel_self(cls, log), FakeContext(scene, obj))
            OUT["panels"][f"{tool}/{cls.bl_label}"] = log
    scene.gse.tool = 'ERASE'
    log = []
    ui.GSE_PT_panel.draw(panel_self(ui.GSE_PT_panel, log), FakeContext(scene, None))
    OUT["panels"]["无对象/主面板"] = log


# ----------------------------------------------------------------- 覆盖几何
class FakeProj:
    """静态投影替身：坐标是预先算好的，update() 什么都不做。"""

    def __init__(self, sx, sy, depth, ok):
        self.sx, self.sy, self.depth, self.ok = sx, sy, depth, ok
        self.key = None

    def update(self, *args, **kwargs):
        return None

    def project(self, pts):
        pts = np.asarray(pts, np.float64)
        return pts[:, 0], pts[:, 1], np.ones(len(pts), bool)

    def pick(self, S, mx, my, radius_px):
        """屏幕上最近的几个点里，取靠前那几颗的平均（和真实 Projection.pick 同思路）。"""
        d = np.hypot(self.sx - mx, self.sy - my)
        m = self.ok & (d < max(float(radius_px), 4.0))
        if int(m.sum()) < 3:
            return None
        idx = np.nonzero(m)[0]
        idx = idx[np.argsort(self.depth[idx])][:max(3, len(idx) // 4)]
        return np.asarray(S.pos[idx], np.float64).mean(0)


def test_cover_geometry(obj):
    S = SplatSet.read(obj.data)
    # 一块 200 点的平面，前层拟合出的法线应当接近 +Z（view_dir 指向场景 -Z）
    idx = np.arange(S.n)
    centre, normal = cover.fit_plane(S.pos[idx], np.array([0.0, 0.0, -1.0]))
    check("fit_plane 法线接近 +Z", abs(abs(normal[2]) - 1.0) < 0.05,
          f"normal={np.round(normal, 3).tolist()}")
    check("fit_plane 朝相机", float(np.dot(normal, [0.0, 0.0, -1.0])) < 0.0,
          f"dot={float(np.dot(normal, [0.0, 0.0, -1.0])):.3f}")

    n_mid = cover.blend_normal(normal, np.array([0.0, 0.0, -1.0]), 0.0)
    check("贴合=0 时正对屏幕", abs(abs(n_mid[2]) - 1.0) < 1e-6, np.round(n_mid, 3).tolist())
    n_mix = cover.blend_normal(normal, np.array([0.0, 0.0, -1.0]), 0.5)
    check("贴合=0.5 时是插值", np.linalg.norm(n_mix) > 0.99, np.round(n_mix, 3).tolist())

    # 矩形网格：必须是以原点为中心的平面矩形
    verts, faces = cover.rect_mesh(2.0, 1.0, 2, 1)
    xs = [p[0] for p in verts]
    ys = [p[1] for p in verts]
    check("rect_mesh 是居中的矩形", abs(min(xs) + 1.0) < 1e-9 and abs(max(xs) - 1.0) < 1e-9
          and abs(min(ys) + 0.5) < 1e-9 and abs(max(ys) - 0.5) < 1e-9
          and all(abs(p[2]) < 1e-12 for p in verts),
          (min(xs), max(xs), min(ys), max(ys)))
    check("rect_mesh 面数与细分匹配", len(faces) == 2 and all(len(f) == 4 for f in faces), faces)

    # 面内基：u 对屏幕向右、v = n × u，且 u × v = n
    rv_probe, _ = _fake_camera()
    u, v_axis, n_axis = cover.basis_from_normal((0.0, 0.0, 1.0), rv_probe)
    check("面内基 u 沿屏幕向右", u.x > 0.9 and abs(u.z) < 1e-6,
          tuple(round(c, 3) for c in u))
    check("面内基 正交且右手系",
          abs(u.dot(v_axis)) < 1e-6 and abs((u.cross(v_axis) - n_axis).length) < 1e-6,
          (round(u.dot(v_axis), 6), round((u.cross(v_axis) - n_axis).length, 6)))

    # 选点：框住屏幕左下 12 像素，只应拿到那一块的 4 个点
    n = 9
    sx = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20], np.float32)
    sy = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20], np.float32)
    depth = np.ones(n, np.float32)
    proj = FakeProj(sx, sy, depth, np.ones(n, bool))

    class FakeS:
        visible = np.ones(9, bool)
        opacity = np.ones(9, np.float32)

    picked = cover.region_splats(proj, FakeS(), ((0, 0), (12, 12)))
    check("region_splats 只取框内点", sorted(picked.tolist()) == [0, 1, 3, 4],
          sorted(picked.tolist()))

    # 主色统计：混了 30% 蓝的白墙里，取到的必须是白（而不是平均成脏灰蓝）
    class ColourS:
        def __init__(self, rgb):
            self._rgb = np.asarray(rgb, np.float32)
            self.n = len(self._rgb)

        def rgb(self, idx):
            return self._rgb[idx]

        @property
        def opacity(self):
            return np.ones(self.n, np.float32)

    white = np.tile([0.78, 0.78, 0.78], (70, 1))
    blue = np.tile([0.53, 0.53, 0.73], (30, 1))
    mixed = ColourS(np.vstack([white, blue]))
    got = cover.dominant_colour(mixed, np.arange(100))
    check("dominant_colour 取占比最大的颜色", abs(got[2] - got[0]) < 0.04,
          f"b-r={got[2] - got[0]:.3f}（接近 0 说明是白，不是蓝）")

    # 面片物体：旋转与位置必须存在物体变换里，顶点在局部 XY 平面上
    left_col = cover.dominant_colour(S, np.nonzero(S.pos[:, 0] < -0.5)[0])
    right_col = cover.dominant_colour(S, np.nonzero(S.pos[:, 0] > 0.5)[0])
    check("左右两半取色不同", left_col[0] > left_col[2] and right_col[2] > right_col[0],
          (np.round(left_col, 3).tolist(), np.round(right_col, 3).tolist()))

    rect_v, rect_f = cover.rect_mesh(2.0, 1.0, 1, 1)
    matrix = cover.patch_matrix((0.5, -0.25, 1.0), (0.0, 0.0, 1.0), rv_probe)
    patch = cover.make_patch(bpy.context, obj, rect_v, rect_f, left_col, matrix)
    check("make_patch 生成网格物体", patch is not None and patch.type == 'MESH'
          and len(patch.data.polygons) == 1, f"{patch.name} 面数={len(patch.data.polygons)}")
    check("面片独立于点云（没有父子关系）", patch.parent is None,
          patch.parent.name if patch.parent else "无父级")
    moved = (patch.matrix_world.translation - matrix.translation).length < 1e-6
    rotated = patch.matrix_world.to_3x3().col[2].z > 0.99
    check("旋转信息在物体变换里", moved and rotated,
          f"位置={tuple(round(v, 3) for v in patch.matrix_world.translation)}，"
          f"法线 z={patch.matrix_world.to_3x3().col[2].z:.3f}")
    check("顶点在局部平面内（刚性矩形）",
          all(abs(v.co.z) < 1e-9 for v in patch.data.vertices)
          and abs(max(v.co.x for v in patch.data.vertices) - 1.0) < 1e-9,
          [tuple(round(c, 3) for c in v.co) for v in patch.data.vertices][:2])

    check("面片带材质", len(patch.data.materials) == 1 and patch.data.materials[0] is not None,
          patch.data.materials[0].name if patch.data.materials else "")
    mat = patch.data.materials[0]
    bsdf = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    base = tuple(round(v, 3) for v in bsdf.inputs['Base Color'].default_value)
    check("材质颜色取自取到的颜色",
          abs(base[0] - left_col[0]) < 0.01 and abs(base[2] - left_col[2]) < 0.01, base)

    patch2 = cover.make_patch(bpy.context, obj, rect_v, rect_f, right_col, matrix)
    mat2 = patch2.data.materials[0]
    check("每块面片有独立材质", mat2 is not mat and mat2.name != mat.name,
          f"{mat.name} / {mat2.name}")
    base2 = tuple(round(v, 3) for v in
                  next(n for n in mat2.node_tree.nodes
                       if n.type == 'BSDF_PRINCIPLED').inputs['Base Color'].default_value)
    check("第二块面片用的是另一边的颜色", base2[2] > base2[0], base2)


# --------------------------------------------------- 还原笔刷的真实代码路径
class _FakeOverlay:
    def __init__(self):
        self.points = []
        self.rings = []
        self.matrix = None

    def clear(self):
        self.points = []
        self.rings = []


class _FakeProj:
    """够还原笔刷用的投影替身：屏幕坐标直接由局部坐标缩放而来。"""

    def __init__(self, S):
        self.sx = 400.0 + S.pos[:, 0] * 100.0
        self.sy = 300.0 + S.pos[:, 1] * 100.0
        self.depth = 5.0 - S.pos[:, 2]
        self.ok = np.ones(S.n, bool)

    def disc(self, mx, my, r, extents=False):
        d = np.hypot(self.sx - mx, self.sy - my)
        idx = np.nonzero(self.ok & (d < r))[0]
        return idx, d[idx]

    def project(self, pts):
        return 400.0 + pts[:, 0] * 100.0, 300.0 + pts[:, 1] * 100.0, np.ones(len(pts), bool)


class _FakeRv3d:
    from mathutils import Matrix as _M
    # 相机在 z=+5 看向 -Z：view_matrix 是把世界平移到相机空间的矩阵
    view_matrix = _M.Translation((0.0, 0.0, -5.0))


class _FakeBrush:
    """把 GSE_OT_brush 的方法当普通函数调用，只补它读的几个属性。"""

    def __init__(self, S, obj):
        self.S = S
        self.T = None
        self.obj = obj
        self.radius = 0.5
        self.hit = np.zeros(3, np.float32)
        self.view_dir = np.array([0.0, 0.0, -1.0], np.float32)
        self.proj = _FakeProj(S)
        self.rv3d = _FakeRv3d()
        self.rs_remove = np.zeros(S.n, bool)
        self.rs_unfade = np.zeros(S.n, bool)
        self.rs_restore = np.zeros(0, bool)
        self.reports = []

    def report(self, level, message):
        self.reports.append(message)


def test_restore_path(obj):
    p = bpy.context.scene.gse
    p.tool = 'RESTORE'
    p.depth_mode = 'SURFACE'

    # 制造"已被擦除"的点，挑屏幕中心附近的一批，保证还原笔刷能命中
    S = SplatSet.read(obj.data)
    st = core.Stroke(S)
    near = np.argsort(np.linalg.norm(S.pos[:, :2], axis=1))[:60]
    st.erase(near, np.ones(len(near), np.float32), 1.0)
    st.commit_clones_and_erase(False)
    S.write(obj.data)
    S = SplatSet.read(obj.data)
    hidden_before = int(S.hidden.sum())
    check("先擦掉一批点", hidden_before >= 60, hidden_before)

    brush = _FakeBrush(S, obj)
    try:
        ops.GSE_OT_brush.restore_dab(brush, bpy.context, 400.0, 300.0)
        check("restore_dab 不再抛 AttributeError", True)
    except Exception as ex:                       # noqa: BLE001
        check("restore_dab 不再抛 AttributeError", False, repr(ex))
        return

    # 面板画覆盖层（这里也读 restore_erased / remove_added）
    try:
        ops.GSE_OT_brush.draw_restore(brush, _FakeOverlay(), p, None)
        check("还原的覆盖层能画出来", True)
    except Exception as ex:                       # noqa: BLE001
        check("还原的覆盖层能画出来", False, repr(ex))

    try:
        ops.GSE_OT_brush.end_restore(brush, bpy.context)
        S2 = SplatSet.read(obj.data)
        hidden_after = int(S2.hidden.sum())
        check("还原写回后隐藏点数减少", hidden_after < hidden_before,
              f"{hidden_before} -> {hidden_after}")
        check("还原有报告文字", any("已还原" in r for r in brush.reports), brush.reports)
    except Exception as ex:                       # noqa: BLE001
        check("还原写回后隐藏点数减少", False, repr(ex))

    p.tool = 'ERASE'


# ----------------------------------------------------------------- 其余检查
def test_props():
    """ops/cover 里读过的 scene.gse 属性必须都还在（上一版就是删属性删崩了还原笔刷）。"""
    import pathlib
    import re
    pkg_dir = pathlib.Path(ops.__file__).parent
    used = set()
    for name in ("ops.py", "cover.py"):
        src = (pkg_dir / name).read_text(encoding="utf-8")
        used |= set(re.findall(r"\bp\.([a-z_][a-z_0-9]*)", src))
    prop = bpy.context.scene.gse
    missing = sorted(n for n in used if not hasattr(prop, n))
    check("代码用到的属性都有定义", not missing, missing if missing else f"{len(used)} 个")


def test_ops():
    ids = [c.bl_idname for c in ops.classes]
    check("只注册需要的操作符",
          ids == ['gse.set_tool', 'gse.set_depth', 'gse.brush', 'gse.cover', 'gse.stop_cover',
                  'gse.export_ply', 'gse.merge_splats', 'gse.bake_edits', 'gse.stop_brush'], ids)
    removed = [n for n in ('gse.select_all', 'gse.fill_selected', 'gse.select_floaters',
                           'gse.delete_selected') if hasattr(bpy.types, n)]
    check("旧的选择/填充操作符已移除", not removed, removed)
    tools = [t[1] for t in ops.TOOL_ITEMS]
    check("工具只剩 擦除/还原/覆盖", tools == ['擦除', '还原', '覆盖'], tools)
    panels = [ui.GSE_PT_panel.bl_label, ui.GSE_PT_brush.bl_label, ui.GSE_PT_cover.bl_label,
              ui.GSE_PT_shortcuts.bl_label, ui.GSE_PT_export.bl_label]
    check("面板顺序", panels == ['GS 编辑器', '笔刷', '覆盖', '快捷键', '烘焙、合并与导出'], panels)


def test_erase(obj):
    """擦除路径：直接在 SplatSet 上跑一次笔触，确认核心仍然连通。"""
    S = SplatSet.read(obj.data)
    st = core.Stroke(S)
    idx = np.arange(0, 40)
    st.erase(idx, np.ones(len(idx), np.float32), 1.0)
    hidden, added = st.commit_clones_and_erase(False)
    S.write(obj.data)
    check("擦除写回 → 隐藏点数增加", hidden == 40, hidden)


# ------------------------------------------------- 端到端：框选 → 面片（假相机）
class _R:
    pass


class _Region:
    width = 800
    height = 600


def _fake_camera(width=800, height=600, dist=5.0):
    """手搓一台针孔相机：焦距 800px（水平 90°），在 +Z 上看向 -Z。"""
    from mathutils import Matrix

    half_w, half_h = 0.5, 0.375
    near, far = 0.1, 100.0
    cam = Matrix.Translation((0.0, 0.0, dist))
    view = cam.inverted()
    proj = Matrix.Identity(4)
    proj[0][0] = 1.0 / half_w
    proj[1][1] = 1.0 / half_h
    proj[2][2] = -(far + near) / (far - near)
    proj[2][3] = -2.0 * far * near / (far - near)
    proj[3][2] = -1.0
    proj[3][3] = 0.0

    rv = _R()
    rv.view_matrix = view
    rv.window_matrix = proj
    rv.perspective_matrix = proj @ view
    rv.is_perspective = True

    class Reg:
        pass
    reg = Reg()
    reg.width, reg.height = width, height
    return rv, reg


def _fake_proj(S, obj, rv, region):
    mw = obj.matrix_world
    P = np.array(rv.perspective_matrix @ mw, np.float64)
    V = np.array(rv.view_matrix @ mw, np.float64)
    clip = S.pos @ P[:, :3].T + P[:, 3]
    w = np.where(clip[:, 3] > 1e-6, clip[:, 3], 1.0)
    return FakeProj((clip[:, 0] / w * 0.5 + 0.5) * region.width,
                    (clip[:, 1] / w * 0.5 + 0.5) * region.height,
                    -(S.pos @ V[2, :3] + V[2, 3]),
                    clip[:, 3] > 1e-6)


RECT = ((200.0, 150.0), (600.0, 450.0))


def _patch_geometry(patch):
    """面片的世界顶点、中心、法线（法线来自物体变换的 +Z）。"""
    mw = patch.matrix_world
    verts = np.array([np.array(mw @ v.co) for v in patch.data.vertices])
    centre = np.array(mw.translation)
    normal = np.array(mw.to_3x3().col[2])
    normal /= max(np.linalg.norm(normal), 1e-12)
    return verts, centre, normal


def _world_corners(patch):
    """按局部坐标找出四个角，返回世界坐标（左上、右上、右下、左下）。"""
    mw = patch.matrix_world
    loc = np.array([list(v.co) for v in patch.data.vertices])
    idx = []
    for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
        idx.append(int(np.argmax(loc[:, 0] * sx + loc[:, 1] * sy)))
    return [np.array(mw @ patch.data.vertices[i].co) for i in idx]


def _is_rectangle(corners, tol=0.02):
    """四个角是不是真正的矩形：邻边垂直、对边等长。"""
    tl, tr, br, bl = corners
    e1, e2, e3, e4 = tr - tl, br - tr, bl - br, tl - bl
    l1, l2, l3, l4 = (np.linalg.norm(e) for e in (e1, e2, e3, e4))
    if min(l1, l2, l3, l4) < 1e-9:
        return False
    cos = abs(np.dot(e1, e2)) / (l1 * l2)
    return cos < tol and abs(l1 - l3) < tol * l1 and abs(l2 - l4) < tol * l2


def test_build_patch_flat(obj):
    """贴合=0：面片正对屏幕，是一个矩形，正好覆盖框选范围。"""
    rv, region = _fake_camera()
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, RECT,
                                   np.array([0.0, 0.0, -1.0]), 0.0, 0.0, 1, None)
    if patch is None:
        check("端到端(平板) 框选→面片", False, why)
        return
    verts, centre, normal = _patch_geometry(patch)
    check("端到端(平板) 面片正对屏幕", normal[2] > 0.999, np.round(normal, 3).tolist())
    check("端到端(平板) 覆盖框选范围",
          abs(np.abs(verts[:, 0]).max() - 1.25) < 0.02
          and abs(np.abs(verts[:, 1]).max() - 0.9375) < 0.02,
          f"x={np.abs(verts[:, 0]).max():.3f} y={np.abs(verts[:, 1]).max():.3f}")
    check("端到端(平板) 是矩形", _is_rectangle(_world_corners(patch)),
          [tuple(round(c, 2) for c in p) for p in _world_corners(patch)])
    check("端到端(平板) 顶点数按细分", len(verts) == 6 and len(patch.data.polygons) == 2,
          f"{len(verts)} 点 / {len(patch.data.polygons)} 面")


def make_curved_cloud(name="弧形点云", n=4000, radius=3.0):
    """合成一张朝相机鼓起来的弧面（绕 Y 轴弯的圆柱面片段）。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    rng = np.random.default_rng(11)
    ang = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1.0, 1.0, n)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = radius * np.sin(ang)
    pos[:, 2] = radius * (np.cos(ang) - 1.0)
    pos[:, 1] = y
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    base.data.foreach_set("color", np.tile(np.array([0.6, 0.6, 0.6, 0.95], np.float32), (n, 1)).ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_build_patch_curved():
    """曲面：面片保持刚性平面（不再逐点变成高度场），法线对齐该处表面。"""
    obj = make_curved_cloud()
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    view_dir = np.array([0.0, 0.0, -1.0])
    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, RECT,
                                   view_dir, 1.0, 0.0, 3, None)
    if patch is None:
        check("端到端(曲面) 框选→面片", False, why)
        return
    verts, centre, normal = _patch_geometry(patch)
    plane_d = np.abs((verts - centre) @ normal)
    check("端到端(曲面) 面片保持刚性平面", plane_d.max() < 1e-6, f"离面最大 {plane_d.max():.2e}")

    idx = cover.region_splats(proj, S, RECT)
    _, n_local = cover.fit_plane(S.pos[idx], view_dir)
    n_world = np.array(obj.matrix_world.to_3x3() @ Vector(n_local))
    n_world /= np.linalg.norm(n_world)
    check("端到端(曲面) 法线对齐该处表面", abs(float(np.dot(normal, n_world))) > 0.98,
          f"dot={float(np.dot(normal, n_world)):.3f}")


def make_noisy_cloud(name="噪点平面", n=20000, noise=0.06):
    """一块带前后噪点的平面（模拟真实扫描里贴面的那层高斯的深度抖动）。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    rng = np.random.default_rng(23)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = rng.uniform(-1.5, 1.5, n)
    pos[:, 1] = rng.uniform(-1.0, 1.0, n)
    pos[:, 2] = rng.normal(0.0, noise, n)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    base.data.foreach_set("color", np.tile(np.array([0.7, 0.7, 0.7, 0.95], np.float32), (n, 1)).ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_tilted_cloud(name="斜面点云", n=20000, slope=0.8, noise=0.002):
    """一块相对视线倾斜的平面：z = slope*y，模拟柜面的斜视角。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    rng = np.random.default_rng(31)
    x = rng.uniform(-1.5, 1.5, n)
    y = rng.uniform(-1.0, 1.0, n)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = x
    pos[:, 1] = y
    pos[:, 2] = slope * y + rng.normal(0.0, noise, n)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    base.data.foreach_set("color", np.tile(np.array([0.7, 0.7, 0.7, 0.95], np.float32), (n, 1)).ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_patch_hugs_tilted_surface():
    """斜着的表面：面片必须整块旋转对齐那个斜面，而不是被拉成别的形状。"""
    obj = make_tilted_cloud()
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    rect = ((300.0, 220.0), (520.0, 420.0))
    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                   np.array([0.0, 0.0, -1.0]), 1.0, 0.0, 2, None)
    if patch is None:
        check("斜面 框选→面片", False, why)
        return
    verts, centre, normal = _patch_geometry(patch)
    expected = np.array([0.0, -0.8, 1.0])
    expected /= np.linalg.norm(expected)
    check("斜面 面片法线对齐斜面", abs(float(np.dot(normal, expected))) > 0.995,
          f"dot={float(np.dot(normal, expected)):.4f}")
    dist = np.abs(verts[:, 2] - 0.8 * verts[:, 1]) / np.linalg.norm([0.0, 0.8, 1.0])
    check("斜面 面片贴在斜面上", dist.max() < 0.03, f"最大离面 {dist.max():.4f}")
    check("斜面 面片仍是矩形", _is_rectangle(_world_corners(patch)),
          [tuple(round(c, 2) for c in p) for p in _world_corners(patch)])


def test_patch_hides_splats():
    """带噪点的贴面：面片要盖在大部分被选中的点前面。"""
    obj = make_noisy_cloud(noise=0.004)
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    rect = ((300.0, 200.0), (500.0, 400.0))
    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                   np.array([0.0, 0.0, -1.0]), 1.0, 0.0, 2, None)
    if patch is None:
        check("噪点面 框选→面片", False, why)
        return
    verts, centre, normal = _patch_geometry(patch)
    idx = cover.region_splats(proj, S, rect)
    mw = obj.matrix_world
    world = np.array([np.array(mw @ Vector(p)) for p in S.pos[idx]])
    d = (world - centre) @ normal
    behind = float((d <= 0.0).mean())
    check("噪点面 面片挡在大部分点前面", behind >= 0.8, f"挡住 {behind:.1%}")


def make_bumpy_wall(name="带凸起的墙", n=24000, ribbon=6000):
    """白墙前面立着一道凸起：凸起的侧面几乎正对视线，是拉偏法线的元凶。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n + ribbon)
    rng = np.random.default_rng(53)
    pos = np.zeros((n + ribbon, 3), np.float32)
    pos[:n, 0] = rng.uniform(-1.5, 1.5, n)
    pos[:n, 1] = rng.uniform(-1.0, 1.0, n)
    pos[:n, 2] = rng.normal(0.0, 0.002, n)
    pos[n:, 0] = rng.uniform(0.45, 0.75, ribbon)          # 凸起的侧面
    pos[n:, 1] = rng.uniform(-0.7, 0.7, ribbon)
    pos[n:, 2] = rng.uniform(0.0, 0.22, ribbon)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    rgba = np.tile(np.array([1.0, 1.0, 1.0, 0.95], np.float32), (n + ribbon, 1))
    rgba[n:] = (0.45, 0.45, 0.45, 0.95)
    base.data.foreach_set("color", rgba.ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_robust_plane_fit():
    """凹凸结构：普通 PCA 会被侧面带偏，抗干扰拟合必须还认这面墙。"""
    obj = make_bumpy_wall()
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    view_dir = np.array([0.0, 0.0, -1.0])
    rect = ((280.0, 210.0), (540.0, 430.0))     # 框住墙，凸起在框内偏右
    idx = cover.region_splats(proj, S, rect)
    check("凹凸墙 框内取到点", len(idx) > 500, len(idx))

    _, n_naive = cover.fit_plane(S.pos[idx], view_dir)
    _, n_robust = cover.fit_plane_robust(S.pos[idx], view_dir)
    err_naive = np.degrees(np.arccos(min(1.0, abs(float(n_naive[2])))))
    err_robust = np.degrees(np.arccos(min(1.0, abs(float(n_robust[2])))))
    check("普通 PCA 会被带偏（对照）", err_naive > 0.5, f"偏差 {err_naive:.2f}°")
    check("抗干扰拟合认住这面墙", err_robust < 1.0, f"偏差 {err_robust:.2f}°")
    check("抗干扰拟合明显优于普通 PCA", err_robust < err_naive * 0.5,
          f"{err_robust:.2f}° vs {err_naive:.2f}°")

    # 整条链：面片法线也要贴着墙
    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                   view_dir, 1.0, 0.0, 2, None)
    if patch is None:
        check("凹凸墙 框选→面片", False, why)
        return
    _, _, normal = _patch_geometry(patch)
    ang = np.degrees(np.arccos(min(1.0, abs(float(normal[2])))))
    check("凹凸墙 面片旋转对齐墙面", ang < 1.5, f"偏差 {ang:.2f}°")


def make_wall_with_backdrop(name="墙+背景", n=20000, m=20000):
    """前面一面墙、后面 3 米处一大片背景：背景在屏幕上也落在框里，但被墙挡住。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n + m)
    rng = np.random.default_rng(67)
    pos = np.zeros((n + m, 3), np.float32)
    pos[:n, 0] = rng.uniform(-1.5, 1.5, n)
    pos[:n, 1] = rng.uniform(-1.0, 1.0, n)
    pos[:n, 2] = rng.normal(0.0, 0.002, n)
    pos[n:, 0] = rng.uniform(-6.0, 6.0, m)
    pos[n:, 1] = rng.uniform(-4.0, 4.0, m)
    pos[n:, 2] = -3.0 + rng.normal(0.0, 0.01, m)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    rgba = np.tile(np.array([1.0, 1.0, 1.0, 0.95], np.float32), (n + m, 1))
    rgba[n:] = (0.2, 0.2, 0.2, 0.95)
    base.data.foreach_set("color", rgba.ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_backdrop_ignored():
    """背景：框里混着被墙挡住的背景点，拟合与面片都必须留在墙上。"""
    obj = make_wall_with_backdrop()
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    rect = ((300.0, 220.0), (500.0, 400.0))
    idx_all = cover.region_splats(proj, S, rect)
    idx = cover.surface_points(proj, S, idx_all)
    z_kept = S.pos[idx][:, 2]
    check("背景 被遮挡的背景点被剔除", abs(float(np.median(z_kept))) < 0.05
          and int((z_kept < -1.0).sum()) == 0,
          f"中位 z={float(np.median(z_kept)):.3f}，远深度点数={int((z_kept < -1.0).sum())}/{len(idx)}")
    check("背景 保留的点仍是大部分墙面", len(idx) > 0.5 * len(idx_all),
          f"{len(idx)} / {len(idx_all)}")

    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                   np.array([0.0, 0.0, -1.0]), 1.0, 0.0, 2, None)
    if patch is None:
        check("背景 框选→面片", False, why)
        return
    verts, centre, normal = _patch_geometry(patch)
    check("背景 面片贴在前面的墙上", abs(float(centre[2])) < 0.05,
          f"面片中心 z={float(centre[2]):.3f}（墙在 0，背景在 -3）")
    check("背景 面片法线朝相机", abs(float(normal[2])) > 0.99, np.round(normal, 3).tolist())


def make_oblique_wall(name="斜视墙", n=40000, slope=3.0):
    """几乎侧看过去的一面墙：法线与视线夹角约 71°，是最容易拉飞的角度。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    rng = np.random.default_rng(71)
    x = rng.uniform(-1.2, 1.2, n)
    y = rng.uniform(-1.0, 1.0, n)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = x
    pos[:, 1] = y
    pos[:, 2] = slope * y + rng.normal(0.0, 0.002, n)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    base.data.foreach_set("color", np.tile(np.array([0.7, 0.7, 0.7, 0.95], np.float32), (n, 1)).ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_oblique_view():
    """斜视：面片仍要贴在墙上（不能被掰成朝屏幕的卡片）。"""
    obj = make_oblique_wall()
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    rect = ((330.0, 250.0), (470.0, 350.0))
    patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                   np.array([0.0, 0.0, -1.0]), 1.0, 0.0, 2, None)
    if patch is None:
        check("斜视 框选→面片", False, why)
        return
    verts, centre, normal = _patch_geometry(patch)
    expected = np.array([0.0, -3.0, 1.0])
    expected /= np.linalg.norm(expected)
    check("斜视 面片法线对齐墙面", abs(float(np.dot(normal, expected))) > 0.99,
          f"dot={float(np.dot(normal, expected)):.4f}")
    dist = np.abs(verts[:, 2] - 3.0 * verts[:, 1]) / np.linalg.norm([0.0, 3.0, 1.0])
    check("斜视 面片贴在墙上", dist.max() < 0.05, f"最大离面 {dist.max():.4f}")


def test_manual_rotation():
    """手动对位：鼠标位移 -> 旋转，中心不变、尺寸不变、法线真的转过去。"""
    from mathutils import Quaternion
    rv, region = _fake_camera(dist=5.0)
    base = cover.patch_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), rv)
    identity = Quaternion((1.0, 0.0, 0.0, 0.0))

    same = cover.rotated_matrix(base, identity)
    check("手动旋转 零旋转矩阵相等",
          max(abs(same[r][c] - base[r][c]) for r in range(4) for c in range(4)) < 1e-9, "max diff < 1e-9")

    # 横拖让法线绕屏幕竖轴转：100px * 0.01 rad ≈ 57.3°
    q = cover.rotation_step(identity, 100.0, 0.0, rv.view_matrix)
    m = cover.rotated_matrix(base, q)
    n0 = np.array(base.to_3x3().col[2])
    n1 = np.array(m.to_3x3().col[2])
    ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(n0, n1)), -1.0, 1.0))))
    check("手动旋转 横拖 100px 约转 57°", abs(ang - 57.3) < 3.0, f"转了 {ang:.1f}°")
    check("手动旋转 中心保持不动",
          (np.array(m.translation) - np.array(base.translation)).max() < 1e-9,
          np.round(np.array(m.translation), 4).tolist())
    check("手动旋转 尺寸仍是 1（正交）",
          abs(float(np.linalg.det(np.array(m.to_3x3()))) - 1.0) < 1e-6,
          f"det={float(np.linalg.det(np.array(m.to_3x3()))):.6f}")

    # 同一个轴上的连续两步，等于把角度加起来
    qa = cover.rotation_step(identity, 50.0, 0.0, rv.view_matrix)
    qb = cover.rotation_step(qa, 50.0, 0.0, rv.view_matrix)
    qc = cover.rotation_step(identity, 100.0, 0.0, rv.view_matrix)
    ma = cover.rotated_matrix(base, qb)
    mb = cover.rotated_matrix(base, qc)
    diff = max(abs(ma[r][c] - mb[r][c]) for r in range(4) for c in range(4))
    check("手动旋转 同轴分两步等于一步", diff < 1e-6, f"差 {diff:.2e}")

    shifted = cover.shifted_matrix(base, 0.25)
    check("手动旋转 滚轮只沿法线平移",
          abs(float(shifted.translation.z) - 0.25) < 1e-9
          and abs(float(shifted.translation.x)) < 1e-9
          and abs(float(shifted.translation.y)) < 1e-9,
          tuple(round(v, 3) for v in shifted.translation))


def test_snap_follows_surface():
    """吸附：面片中心跟到命中点、法线等于该处的局部表面法线（斜面上也一样）。"""
    obj = make_tilted_cloud(slope=3.0)          # 71° 斜视也照样吸
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    mw = obj.matrix_world
    view_dir = np.array([0.0, 0.0, -1.0])

    class FakeCover:
        pass

    brush = FakeCover()
    brush.proj = proj
    brush.S = S
    brush.obj = obj
    brush.region = region
    brush.rv3d = rv
    brush.rev = 0
    brush.snap_radius_px = 30.0
    brush.base_shift = 0.0
    brush.push = 0.0

    # 用一个 40x40 的面片去吸屏幕中心附近的两处位置
    verts, faces = cover.rect_mesh(0.3, 0.3, 1, 1)
    patch = cover.make_patch(bpy.context, obj, verts, faces, (0.7, 0.7, 0.7),
                             cover.patch_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), rv))
    brush.patch = patch

    from gs_editor import ops as ops_mod
    results = []
    for px, py in ((400.0, 300.0), (430.0, 330.0)):
        ok = ops_mod.GSE_OT_cover.snap_to_surface(brush, bpy.context, px, py)
        results.append(bool(ok))
        if not ok:
            continue
        m = patch.matrix_world
        normal = np.array(m.to_3x3().col[2])
        normal /= np.linalg.norm(normal)
        expected = np.array([0.0, -3.0, 1.0])
        expected /= np.linalg.norm(expected)
        # 命中点应在斜面上
        centre = np.array(m.translation)
        dist = abs(float(centre[2] - 3.0 * centre[1])) / np.linalg.norm([0.0, 3.0, 1.0])
        check(f"吸附({int(px)},{int(py)}) 中心落在表面", dist < 0.05, f"离面 {dist:.4f}")
        check(f"吸附({int(px)},{int(py)}) 法线等于局部表面法线",
              abs(float(np.dot(normal, expected))) > 0.98,
              f"dot={float(np.dot(normal, expected)):.3f}")
    check("吸附 两次都成功命中", all(results), results)

    # 鼠标移到空处：不改动面片
    before = patch.matrix_world.copy()
    box = bpy.data.objects.new("空处遮挡物", None)
    ok_empty = ops_mod.GSE_OT_cover.snap_to_surface(brush, bpy.context, 5.0, 5.0)
    after = patch.matrix_world
    same = max(abs(before[r][c] - after[r][c]) for r in range(4) for c in range(4)) < 1e-9
    check("吸附 空处不改变面片", (not ok_empty) and same, f"ok={ok_empty}")


def test_patch_size_matches_frame():
    """面片在屏幕上的投影必须和你框的那个矩形一样大（正对、斜视、曲面都算）。"""
    rv, region = _fake_camera(dist=5.0)
    rect = ((300.0, 220.0), (520.0, 420.0))
    tw = rect[1][0] - rect[0][0]
    th = rect[1][1] - rect[0][1]
    cases = (
        ("正对", lambda: make_cloud(n=8000)),
        ("斜面 39°", lambda: make_tilted_cloud(slope=0.8)),
        ("斜面 71°", lambda: make_oblique_wall()),
        ("曲面", lambda: make_curved_cloud()),
    )
    for label, maker in cases:
        obj = maker()
        S = SplatSet.read(obj.data)
        proj = _fake_proj(S, obj, rv, region)
        patch, why = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                       np.array([0.0, 0.0, -1.0]), 1.0, 0.0, 2, None)
        if patch is None:
            check(f"尺寸({label}) 生成面片", False, why)
            continue
        # 默认：框多大就多大 —— 世界尺寸应当等于 框像素 × 该处像素尺度
        centre = np.array(patch.matrix_world.translation)
        wpp = cover._world_per_pixel(region, rv, rect, centre)
        want_w, want_h = tw * wpp, th * wpp
        dims = patch.dimensions
        got = sorted([float(dims.x), float(dims.y)])
        want = sorted([want_w, want_h])
        check(f"尺寸({label}) 世界尺寸 = 框选换算值",
              abs(got[0] / want[0] - 1.0) < 0.05 and abs(got[1] / want[1] - 1.0) < 0.05,
              f"{got[0]:.3f}×{got[1]:.3f} vs {want[0]:.3f}×{want[1]:.3f}")

        # 可选：尺寸按屏幕投影时，投影应当正好盖住框
        patch2, why2 = cover.build_patch(bpy.context, obj, S, proj, region, rv, rect,
                                         np.array([0.0, 0.0, -1.0]), 1.0, 0.0, 2, None,
                                         size_from_screen=True)
        if patch2 is None:
            check(f"尺寸({label}) 投影模式生成", False, why2)
            continue
        pts = [patch2.matrix_world @ v.co for v in patch2.data.vertices]
        sx, sy, ok = cover.project_points(region, rv, pts)
        w = float(sx.max() - sx.min())
        h = float(sy.max() - sy.min())
        check(f"尺寸({label}) 投影模式盖住框选", abs(w / tw - 1.0) < 0.03 and abs(h / th - 1.0) < 0.03,
              f"{w:.1f}×{h:.1f}px vs {tw:.1f}×{th:.1f}px")


def make_island_cloud(name="蓝斑白墙", n=30000):
    """一片白色表面上有一小块蓝斑：框住蓝斑时，取色应当取外面那圈的白。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    rng = np.random.default_rng(41)
    pos = np.zeros((n, 3), np.float32)
    pos[:, 0] = rng.uniform(-1.5, 1.5, n)
    pos[:, 1] = rng.uniform(-1.0, 1.0, n)
    pos[:, 2] = rng.normal(0.0, 0.002, n)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    base = pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT')
    rgba = np.tile(np.array([1.0, 1.0, 1.0, 0.95], np.float32), (n, 1))     # 白
    island = (np.abs(pos[:, 0]) < 0.075) & (np.abs(pos[:, 1]) < 0.075)
    rgba[island] = (0.0, 0.0, 1.0, 0.95)                                    # 蓝斑
    base.data.foreach_set("color", rgba.ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_ring_colour():
    """取色：框里是蓝的、框外那圈是白的，就应该取白。"""
    obj = make_island_cloud()
    rv, region = _fake_camera(dist=5.0)
    S = SplatSet.read(obj.data)
    proj = _fake_proj(S, obj, rv, region)
    rect = ((388.0, 288.0), (412.0, 312.0))     # 正好框住中间的蓝斑（±12px ≈ 世界 ±0.075）
    inside = np.nonzero(proj.ok & (proj.sx >= rect[0][0]) & (proj.sx <= rect[1][0])
                        & (proj.sy >= rect[0][1]) & (proj.sy <= rect[1][1]))[0]
    col_in = cover.dominant_colour(S, inside)
    check("测试前提：框内确实是蓝的", col_in[2] - col_in[0] > 0.15,
          f"b-r={col_in[2] - col_in[0]:.3f}")
    col = cover.ring_colour(proj, S, rect, ring=0.35)
    check("ring_colour 取框外的白色而不是框内的蓝", abs(col[2] - col[0]) < 0.05,
          f"取到 {tuple(round(c, 3) for c in col)}，b-r={col[2] - col[0]:.3f}")


def test_build_patch_integration(obj):
    """兼容旧入口：平面端到端 + 曲面端到端 + 斜面贴合 + 噪点遮挡 + 取色。"""
    test_build_patch_flat(obj)
    test_build_patch_curved()
    test_patch_hugs_tilted_surface()
    test_patch_hides_splats()
    test_robust_plane_fit()
    test_backdrop_ignored()
    test_oblique_view()
    test_manual_rotation()
    test_snap_follows_surface()
    test_patch_size_matches_frame()
    test_ring_colour()


def test_ui_behaviour(obj):
    """界面行为：工具按钮只选不涂抹、两套参数面板同一套排版、快捷键按工具分开。"""
    scene = bpy.context.scene
    ctx = FakeContext(scene, obj)

    log = []
    scene.gse.tool = 'ERASE'
    ui.GSE_PT_panel.draw(panel_self(ui.GSE_PT_panel, log), ctx)
    tools = [e for e in log if e[0] == "operator" and e[2] in ("擦除", "还原", "覆盖")]
    check("工具按钮用 set_tool（只选不涂抹）",
          bool(tools) and all(e[1] == "gse.set_tool" for e in tools),
          [(e[1], e[2]) for e in tools])

    # 擦除：笔刷面板可见、覆盖面板隐藏
    check("擦除时只显示笔刷面板", ui.GSE_PT_brush.poll(ctx) and not ui.GSE_PT_cover.poll(ctx),
          (ui.GSE_PT_brush.poll(ctx), ui.GSE_PT_cover.poll(ctx)))
    log = []
    ui.GSE_PT_brush.draw(panel_self(ui.GSE_PT_brush, log), ctx)
    brush_props = [e[1] for e in log if e[0] == "prop"]
    check("擦除时显示笔刷参数",
          "radius_px" in brush_props and "strength" in brush_props, brush_props)
    check("笔刷面板没有嵌套方框", not any(e[0] == "box" for e in log), log[:3])

    # 覆盖：覆盖面板可见、笔刷面板隐藏，且排版同为行式
    scene.gse.tool = 'COVER'
    check("覆盖时只显示覆盖面板", ui.GSE_PT_cover.poll(ctx) and not ui.GSE_PT_brush.poll(ctx),
          (ui.GSE_PT_brush.poll(ctx), ui.GSE_PT_cover.poll(ctx)))
    log = []
    ui.GSE_PT_cover.draw(panel_self(ui.GSE_PT_cover, log), ctx)
    cover_props = [e[1] for e in log if e[0] == "prop"]
    check("覆盖时只显示覆盖参数",
          "cover_fit" in cover_props and "radius_px" not in cover_props, cover_props)
    check("覆盖面板没有嵌套方框", not any(e[0] == "box" for e in log), log[:3])

    # 还原：开关是普通复选框（不是大按钮），选择深度是并排两个按钮（不是下拉）
    scene.gse.tool = 'RESTORE'
    scene.gse.depth_mode = 'THROUGH'
    log = []
    ui.GSE_PT_brush.draw(panel_self(ui.GSE_PT_brush, log), ctx)
    restore_bits = [e for e in log if e[0] == "prop"]
    check("还原：开关用普通复选框",
          all((len(e) < 3 or not e[2]) for e in restore_bits if e[1] in
              ("restore_erased", "remove_added")), restore_bits)
    depth_buttons = [e for e in log if e[0] == "operator" and e[1] == "gse.set_depth"]
    check("选择深度是并排两个按钮，不是下拉菜单",
          [e[2] for e in depth_buttons] == ["表面", "穿透"], depth_buttons)
    check("当前深度对应的按钮是按下状态",
          [e[3].get("depress") for e in depth_buttons] == [False, True],
          [e[3].get("depress") for e in depth_buttons])
    scene.gse.depth_mode = 'SURFACE'
    bpy.ops.gse.set_depth(mode='THROUGH')
    check("点深度按钮真的切了状态", scene.gse.depth_mode == 'THROUGH', scene.gse.depth_mode)
    scene.gse.depth_mode = 'SURFACE'

    # 顶部更新条：版本号 + 检查更新按钮（版本取自 blender_manifest.toml）
    log = []
    scene.gse.tool = 'ERASE'
    ui.GSE_PT_panel.draw(panel_self(ui.GSE_PT_panel, log), ctx)
    version_label = "v" + update_checker.version_text()
    check("主面板顶部显示版本号",
          any(e[0] == "label" and e[1] == version_label for e in log),
          [e for e in log if e[0] == "label"][:3])
    check("主面板顶部有检查更新按钮",
          any(e[0] == "operator" and e[1] == "gse.check_update" for e in log), log[:5])

    # 偏好设置：更新条 + 反馈入口
    log = []
    ui.GSE_AddonPreferences.draw(panel_self(ui.GSE_AddonPreferences, log), ctx)
    check("偏好设置里也有更新条和反馈入口",
          any(e[0] == "operator" and e[1] == "gse.check_update" for e in log)
          and any(e[0] == "operator" and e[1] == "wm.url_open" for e in log), log[:8])

    # 快捷键按工具分组
    scene.gse.tool = 'COVER'
    log = []
    ui.GSE_PT_shortcuts.draw(panel_self(ui.GSE_PT_shortcuts, log), ctx)
    cover_labels = [e[1] for e in log if e[0] == "label"]
    check("覆盖时快捷键面板只列覆盖按键",
          any("覆盖时可用" in t for t in cover_labels)
          and any("吸附" in t for t in cover_labels)
          and not any("调整笔刷大小" in t for t in cover_labels), cover_labels)

    scene.gse.tool = 'ERASE'
    log = []
    ui.GSE_PT_shortcuts.draw(panel_self(ui.GSE_PT_shortcuts, log), ctx)
    brush_labels = [e[1] for e in log if e[0] == "label"]
    check("擦除时快捷键面板只列涂抹按键",
          any("涂抹时可用" in t for t in brush_labels)
          and any("调整笔刷大小" in t for t in brush_labels)
          and not any("吸附" in t for t in brush_labels), brush_labels)


def test_update_checker():
    """更新检查：版本号、两种发布包目录布局、覆盖安装（本地 zip 走完整流程）。"""
    check("版本号解析 v1.2.3", update_checker._parse_version("v1.2.3") == (1, 2, 3),
          update_checker._parse_version("v1.2.3"))
    check("版本号比较：新的才算更新",
          update_checker._version_newer((1, 2, 3), (1, 2, 0))
          and not update_checker._version_newer((1, 2, 0), (1, 2, 3)))

    manifest = os.path.join(os.path.dirname(gs_editor.__file__), "blender_manifest.toml")
    with open(manifest, "r", encoding="utf-8") as handle:
        manifest_text = handle.read()
    check("运行版本和 blender_manifest.toml 一致",
          f'version = "{update_checker.version_text()}"' in manifest_text,
          update_checker.version_text())

    tmp = tempfile.mkdtemp(prefix="gse_verify_update_")
    try:
        # 发布里的扩展包：根目录就是本体
        package = os.path.join(tmp, "package")
        module_dir = os.path.join(package, "gs_editor")
        os.makedirs(os.path.join(module_dir, "__pycache__"))
        with open(os.path.join(module_dir, "blender_manifest.toml"), "w", encoding="utf-8") as handle:
            handle.write('id = "gs_editor"\nversion = "9.9.9"\n')
        with open(os.path.join(module_dir, "__init__.py"), "w", encoding="utf-8") as handle:
            handle.write("# 9.9.9\n")
        with open(os.path.join(module_dir, "new_module.py"), "w", encoding="utf-8") as handle:
            handle.write("# new\n")
        with open(os.path.join(module_dir, "__pycache__", "new_module.pyc"), "wb") as handle:
            handle.write(b"\x00")
        found = update_checker._find_extension_root(package)
        check("认得出扩展包目录", found == module_dir, found)

        # GitHub 源码归档：repo-tag/gs_editor/...
        archive_layout = os.path.join(tmp, "archive")
        nested = os.path.join(archive_layout, "Blender-GSEditor-9.9.9", "gs_editor")
        os.makedirs(nested)
        with open(os.path.join(nested, "blender_manifest.toml"), "w", encoding="utf-8"):
            pass
        with open(os.path.join(archive_layout, "Blender-GSEditor-9.9.9", "README.md"), "w",
                  encoding="utf-8") as handle:
            handle.write("readme")
        found = update_checker._find_extension_root(archive_layout)
        check("认得出源码归档目录", found == nested, found)

        # 一键更新：本地 zip 当发布包，走「下载 → 解压 → 覆盖」全程
        zip_path = os.path.join(tmp, "gs_editor-9.9.9.zip")
        with zipfile.ZipFile(zip_path, "w") as archive:
            for folder, _dirs, files in os.walk(package):
                for name in files:
                    path = os.path.join(folder, name)
                    archive.write(path, os.path.relpath(path, package))
        dest = os.path.join(tmp, "installed")
        saved = update_checker._cache.get(update_checker.REPO)
        update_checker._cache[update_checker.REPO] = {
            "status": "has_update",
            "current_version": update_checker.version_text(),
            "latest_version": "9.9.9",
            "zip_url": pathlib.Path(zip_path).as_uri(),
            "html_url": "",
        }
        try:
            ok, message = update_checker.install_update(plugin_dir=dest)
            installed = [
                os.path.isfile(os.path.join(dest, "blender_manifest.toml")),
                os.path.isfile(os.path.join(dest, "__init__.py")),
                os.path.isfile(os.path.join(dest, "new_module.py")),
                not os.path.isdir(os.path.join(dest, "__pycache__")),
            ]
        finally:
            if saved is None:
                update_checker._cache.pop(update_checker.REPO, None)
            else:
                update_checker._cache[update_checker.REPO] = saved
        check("一键更新把新文件覆盖进安装目录", ok and all(installed), (message, installed))
        with open(os.path.join(dest, "new_module.py"), "r", encoding="utf-8") as handle:
            check("更新后的文件是新版本的", handle.read() == "# new\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    check("没有更新时不乱装", update_checker.install_update(plugin_dir=tmp)[0] is False)
    check("检查更新/一键更新按钮已注册",
          hasattr(bpy.ops.gse, "check_update") and hasattr(bpy.ops.gse, "install_update"))
    check("后台模式不会自己联网", update_checker.get_check_status()["status"] == "pending",
          update_checker.get_check_status())


def main():
    try:
        gs_editor.register()
        check("注册成功", True)
    except Exception as ex:                       # noqa: BLE001
        check("注册成功", False, repr(ex))
        raise
    obj = make_cloud()
    draw_panels(obj)
    test_cover_geometry(obj)
    try:
        test_build_patch_integration(obj)
    except Exception as ex:                       # noqa: BLE001
        check("端到端 框选→面片", False, repr(ex))
    test_ops()
    test_props()
    test_ui_behaviour(obj)
    test_update_checker()
    test_erase(obj)
    test_restore_path(obj)
    ok = all(t["ok"] for t in OUT["tests"])
    print("GSE_VERIFY_BEGIN")
    print(json.dumps(OUT, ensure_ascii=False, indent=1))
    print("GSE_VERIFY_END")
    print("PASS" if ok else "FAIL")


main()
