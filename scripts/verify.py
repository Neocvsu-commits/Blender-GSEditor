# -*- coding: utf-8 -*-
"""在 Blender 里跑 GS Editor 的自检。

用法：
    set GSE_ADDON_PARENT=<gs_editor 上级目录>
    blender -b --factory-startup --python scripts/verify.py

检查：注册、各面板在擦除和还原工具下的 draw、旧工具确实已移除、
擦除与还原路径、PLY 导出变换、更新检查与覆盖安装。
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
from mathutils import Matrix, Quaternion

PARENT = os.environ.get("GSE_ADDON_PARENT")
if not PARENT:
    raise SystemExit("缺少环境变量 GSE_ADDON_PARENT")
sys.path.insert(0, PARENT)

import gs_editor                      # noqa: E402
from gs_editor import core, ops, ply_io, ui, update_checker  # noqa: E402
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
    for tool in ('ERASE', 'RESTORE'):
        scene.gse.tool = tool
        for cls in (ui.GSE_PT_panel, ui.GSE_PT_brush,
                    ui.GSE_PT_shortcuts, ui.GSE_PT_export):
            log = []
            cls.draw(panel_self(cls, log), FakeContext(scene, obj))
            OUT["panels"][f"{tool}/{cls.bl_label}"] = log
    scene.gse.tool = 'ERASE'
    log = []
    ui.GSE_PT_panel.draw(panel_self(ui.GSE_PT_panel, log), FakeContext(scene, None))
    OUT["panels"]["无对象/主面板"] = log


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
    """ops 里读过的 scene.gse 属性必须都还在（上一版就是删属性删崩了还原笔刷）。"""
    import pathlib
    import re
    pkg_dir = pathlib.Path(ops.__file__).parent
    used = set()
    for name in ("ops.py",):
        src = (pkg_dir / name).read_text(encoding="utf-8")
        used |= set(re.findall(r"\bp\.([a-z_][a-z_0-9]*)", src))
    prop = bpy.context.scene.gse
    missing = sorted(n for n in used if not hasattr(prop, n))
    check("代码用到的属性都有定义", not missing, missing if missing else f"{len(used)} 个")


def test_ops():
    ids = [c.bl_idname for c in ops.classes]
    check("只注册需要的操作符",
          ids == ['gse.set_tool', 'gse.set_depth', 'gse.brush',
                  'gse.export_ply', 'gse.merge_splats', 'gse.bake_edits', 'gse.stop_brush'], ids)
    removed = [n for n in ('gse.select_all', 'gse.fill_selected', 'gse.select_floaters',
                           'gse.delete_selected') if hasattr(bpy.types, n)]
    check("旧的选择/填充操作符已移除", not removed, removed)
    tools = [t[1] for t in ops.TOOL_ITEMS]
    check("工具只剩 擦除/还原", tools == ['擦除', '还原'], tools)
    panels = [ui.GSE_PT_panel.bl_label, ui.GSE_PT_brush.bl_label,
              ui.GSE_PT_shortcuts.bl_label, ui.GSE_PT_export.bl_label]
    check("面板顺序", panels == ['GS 编辑器', '笔刷', '快捷键', '烘焙、合并与导出'], panels)


def test_erase(obj):
    """擦除路径：直接在 SplatSet 上跑一次笔触，确认核心仍然连通。"""
    S = SplatSet.read(obj.data)
    st = core.Stroke(S)
    idx = np.arange(0, 40)
    st.erase(idx, np.ones(len(idx), np.float32), 1.0)
    hidden, added = st.commit_clones_and_erase(False)
    S.write(obj.data)
    check("擦除写回 → 隐藏点数增加", hidden == 40, hidden)


def test_ui_behaviour(obj):
    """界面行为：工具按钮只选不涂抹、笔刷参数与快捷键。"""
    scene = bpy.context.scene
    ctx = FakeContext(scene, obj)

    log = []
    scene.gse.tool = 'ERASE'
    ui.GSE_PT_panel.draw(panel_self(ui.GSE_PT_panel, log), ctx)
    tools = [e for e in log if e[0] == "operator" and e[2] in ("擦除", "还原")]
    check("工具按钮用 set_tool（只选不涂抹）",
          bool(tools) and all(e[1] == "gse.set_tool" for e in tools),
          [(e[1], e[2]) for e in tools])

    # 擦除：笔刷面板可见
    check("擦除时显示笔刷面板", ui.GSE_PT_brush.poll(ctx), ui.GSE_PT_brush.poll(ctx))
    log = []
    ui.GSE_PT_brush.draw(panel_self(ui.GSE_PT_brush, log), ctx)
    brush_props = [e[1] for e in log if e[0] == "prop"]
    check("擦除时显示笔刷参数",
          "radius_px" in brush_props and "strength" in brush_props, brush_props)
    check("笔刷面板没有嵌套方框", not any(e[0] == "box" for e in log), log[:3])

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


# --------------------------------------------------------- 导出：物体变换烘焙进 PLY
def read_ply(path):
    """(属性名 -> 一维数组, 点数)。只读本插件写出的那个布局：全 float、表头 end_header。"""
    with open(path, "rb") as handle:
        raw = handle.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    props = [line.split()[2] for line in lines if line.startswith("property")]
    count = int(next(line for line in lines if line.startswith("element vertex")).split()[2])
    data = np.frombuffer(raw[end:], dtype="<f4").reshape(count, len(props))
    return {name: data[:, i] for i, name in enumerate(props)}, count


def make_export_cloud(name, n=3):
    """三个已知坐标、已知尺寸/朝向/颜色的泼溅点，用来核对导出写出了什么。"""
    pc = bpy.data.pointclouds.new(name)
    pc.resize(n)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], np.float32)
    pc.attributes["position"].data.foreach_set("vector", pos.ravel())
    scale = np.array([0.1, 0.2, 0.3], np.float32)
    pc.attributes.new("scale", 'FLOAT_VECTOR', 'POINT').data.foreach_set(
        "vector", np.tile(scale, (n, 1)).ravel())
    half = float(np.cos(np.pi / 4))       # 绕 X 轴 90°
    pc.attributes.new("rotation", 'QUATERNION', 'POINT').data.foreach_set(
        "value", np.tile(np.array([half, half, 0.0, 0.0], np.float32), (n, 1)).ravel())
    base = np.array([0.4, 0.5, 0.6, 0.8], np.float32)
    pc.attributes.new("radiance:base", 'FLOAT_COLOR', 'POINT').data.foreach_set(
        "color", np.tile(base, (n, 1)).ravel())
    for k, coeff in enumerate(([0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, 0.9])):
        pc.attributes.new(f"radiance:sh_{k}", 'FLOAT_VECTOR', 'POINT').data.foreach_set(
            "vector", np.tile(np.array(coeff, np.float32), (n, 1)).ravel())
    obj = bpy.data.objects.new(name, pc)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def test_export_transform():
    tmp = tempfile.mkdtemp(prefix="gse_export_")
    made = []
    try:
        obj = make_export_cloud("导出测试")
        made.append(obj)
        pc = obj.data
        pos_in = SplatSet.read(pc).pos.copy()
        M = (Matrix.Translation((10.0, 0.0, 0.0))
             @ Matrix.Rotation(np.pi / 2, 4, 'Z')
             @ Matrix.Diagonal((2.0, 2.0, 2.0, 1.0)))
        obj.matrix_world = M
        mat = np.array(obj.matrix_world, np.float64)

        plain = os.path.join(tmp, "object_space.ply")
        baked = os.path.join(tmp, "world.ply")
        same = os.path.join(tmp, "identity.ply")
        ply_io.write_pointcloud_ply(plain, pc)
        ply_io.write_pointcloud_ply(baked, pc, matrix=mat)
        ply_io.write_pointcloud_ply(same, pc, matrix=np.eye(4))
        a, _ = read_ply(plain)
        b, n = read_ply(baked)

        def xyz(table):
            return np.stack([table["x"], table["y"], table["z"]], 1)

        check("导出 物体空间就是点的原始坐标", np.abs(xyz(a) - pos_in).max() < 1e-6,
              xyz(a)[0].tolist())
        expect = pos_in.astype(np.float64) @ mat[:3, :3].T + mat[:3, 3]
        check("导出 位置带上了物体的位置/旋转/缩放", np.abs(xyz(b) - expect).max() < 1e-5,
              f"最差 {np.abs(xyz(b) - expect).max():.2e}")
        with open(plain, "rb") as one, open(same, "rb") as two:
            check("导出 单位矩阵不改动任何值", one.read() == two.read())

        got_scale = np.exp(np.stack([b["scale_0"], b["scale_1"], b["scale_2"]], 1))
        check("导出 尺寸乘上了物体缩放", np.abs(got_scale - np.array([0.2, 0.4, 0.6])).max() < 1e-5,
              got_scale[0].tolist())

        R_obj = np.array(obj.matrix_world.to_quaternion().to_matrix())
        R_local = np.array(Quaternion((np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0)).to_matrix())
        expect_R = R_obj @ R_local
        worst = 0.0
        for i in range(n):
            q = Quaternion((b["rot_0"][i], b["rot_1"][i], b["rot_2"][i], b["rot_3"][i]))
            worst = max(worst, float(np.abs(np.array(q.to_matrix()) - expect_R).max()))
        check("导出 每个泼溅点的朝向跟着物体转", worst < 1e-5, f"最差 {worst:.2e}")

        # 高阶颜色：换个方向看，转过的模型和原来的模型取到的颜色应当一致
        rng = np.random.default_rng(3)
        d = rng.normal(size=(16, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        coeffs_in = np.array([[0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [-0.7, 0.8, 0.9]], np.float64)
        k = 3                                   # 这一份数据只有 band 1（3 个系数）
        coeffs_out = np.array([[b[f"f_rest_{ch * k + j}"][0] for ch in range(3)]
                               for j in range(k)])
        lhs = core.sh_basis(d)[0] @ coeffs_out
        rhs = core.sh_basis(d @ R_obj)[0] @ coeffs_in
        check("导出 高阶颜色跟着转，看同一个方向还是同一个颜色",
              np.abs(lhs - rhs).max() < 2e-3, f"最差 {np.abs(lhs - rhs).max():.2e}")

        # 操作符本身：移动过的物体，导出文件里就该是新位置
        obj2 = make_export_cloud("导出操作符测试")
        made.append(obj2)
        obj2.location = (5.0, 0.0, 0.0)
        for other in bpy.context.view_layer.objects:
            other.select_set(False)
        obj2.select_set(True)
        bpy.context.view_layer.objects.active = obj2
        path = os.path.join(tmp, "operator.ply")
        result = bpy.ops.gse.export_ply('EXEC_DEFAULT', filepath=path)
        ok = result == {'FINISHED'} and os.path.isfile(path)
        if ok:
            c, m = read_ply(path)
            check("导出操作符 移动后的位置写进文件",
                  m == 3 and np.abs(xyz(c)[:, 0] - (pos_in[:, 0] + 5.0)).max() < 1e-5,
                  xyz(c)[:, 0].tolist())
        else:
            check("导出操作符 移动后的位置写进文件", False, (result, os.path.isfile(path)))

        # 坐标轴：默认写成 glTF / 网页端的 Y-up，选 Blender 时保持原生坐标
        props = bpy.context.scene.gse
        yup_ply = os.path.join(tmp, "operator_yup.ply")
        zup_ply = os.path.join(tmp, "operator_zup.ply")
        try:
            props.export_up_axis = 'YUP'
            bpy.ops.gse.export_ply('EXEC_DEFAULT', filepath=yup_ply)
            props.export_up_axis = 'ZUP'
            bpy.ops.gse.export_ply('EXEC_DEFAULT', filepath=zup_ply)
        finally:
            props.export_up_axis = 'YUP'

        world = pos_in.astype(np.float64) + np.array([5.0, 0.0, 0.0])
        up, _ = read_ply(yup_ply)
        flat, _ = read_ply(zup_ply)
        as_yup = world[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])   # (x, y, z) -> (x, z, -y)
        check("导出 默认按 glTF 的 Y-up 写 (x, z, -y)",
              np.abs(xyz(up) - as_yup).max() < 1e-5, f"最差 {np.abs(xyz(up) - as_yup).max():.2e}")
        check("导出 选 Blender Z-up 时坐标原样写出",
              np.abs(xyz(flat) - world).max() < 1e-5, f"最差 {np.abs(xyz(flat) - world).max():.2e}")

        R_yup = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        worst = 0.0
        for i in range(len(up["rot_0"])):
            q = Quaternion((up["rot_0"][i], up["rot_1"][i], up["rot_2"][i], up["rot_3"][i]))
            worst = max(worst, float(np.abs(np.array(q.to_matrix()) - R_yup @ R_local).max()))
        check("导出 Y-up 时每个泼溅点的朝向跟着换轴", worst < 1e-5, f"最差 {worst:.2e}")
    finally:
        for obj in made:
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    try:
        gs_editor.register()
        check("注册成功", True)
    except Exception as ex:                       # noqa: BLE001
        check("注册成功", False, repr(ex))
        raise
    obj = make_cloud()
    draw_panels(obj)
    test_ops()
    test_props()
    test_ui_behaviour(obj)
    test_update_checker()
    test_erase(obj)
    test_restore_path(obj)
    test_export_transform()
    ok = all(t["ok"] for t in OUT["tests"])
    print("GSE_VERIFY_BEGIN")
    print(json.dumps(OUT, ensure_ascii=False, indent=1))
    print("GSE_VERIFY_END")
    print("PASS" if ok else "FAIL")


main()
