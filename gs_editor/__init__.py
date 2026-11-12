"""GS Editor：4.2+ 高斯建模参考，5.3+ 原生高斯擦除、还原与覆盖。"""

import bpy

if "bpy" in locals():
    import importlib
    if "reference" in locals():
        reference.unregister()
        importlib.reload(reference)
    if "update_checker" in locals():
        importlib.reload(update_checker)
    if "ops" in locals():
        importlib.reload(ops)
    if "ui" in locals():
        importlib.reload(ui)

from . import ops, ui, update_checker, reference

_classes = ui.classes[:1] + ops.classes + update_checker.classes + (reference.GSE_OT_import_reference,) + ui.classes[1:] + (reference.GSE_PT_reference,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gse = bpy.props.PointerProperty(type=ui.GSE_Props)
    reference.register()
    # 后台查一次 GitHub Releases：不挡启动，不联网也照常能用
    if not bpy.app.background:
        update_checker.check_for_updates()


def unregister():
    reference.unregister()
    update_checker.stop()
    del bpy.types.Scene.gse
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
