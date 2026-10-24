"""GS Editor：在 Blender 5.3+ 里编辑高斯泼溅点云 —— 擦除、还原，并用吸附到表面的矩形面片覆盖区域。"""

import bpy

if "bpy" in locals():
    import importlib
    if "update_checker" in locals():
        importlib.reload(update_checker)
    if "ops" in locals():
        importlib.reload(ops)
    if "ui" in locals():
        importlib.reload(ui)

from . import ops, ui, update_checker

_classes = ui.classes[:1] + ops.classes + update_checker.classes + ui.classes[1:]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gse = bpy.props.PointerProperty(type=ui.GSE_Props)
    # 后台查一次 GitHub Releases：不挡启动，不联网也照常能用
    if not bpy.app.background:
        update_checker.check_for_updates()


def unregister():
    update_checker.stop()
    del bpy.types.Scene.gse
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
