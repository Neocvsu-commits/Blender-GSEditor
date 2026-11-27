"""GS Editor：4.2+ 高斯建模参考，5.3+ 原生高斯选择、擦除与还原。"""

import bpy

if "bpy" in locals():
    import importlib
    if "selection" in locals():
        selection.unregister()
        importlib.reload(selection)
    if "reference" in locals():
        reference.unregister()
        importlib.reload(reference)
    if "update_checker" in locals():
        importlib.reload(update_checker)
    if "ops" in locals():
        importlib.reload(ops)
    if "ui" in locals():
        importlib.reload(ui)
    if "mode" in locals():
        importlib.reload(mode)
    if "export_formats" in locals():
        export_formats.shutdown()
        export_formats.atexit.unregister(export_formats.exit_workers)
        if export_formats.shutdown in bpy.app.handlers.load_pre:
            bpy.app.handlers.load_pre.remove(export_formats.shutdown)
        importlib.reload(export_formats)

from . import ops, ui, update_checker, reference, selection, mode
from . import export_formats

_classes = ui.classes[:1] + ops.classes + mode.classes + export_formats.classes + update_checker.classes + (reference.GSE_OT_import_reference, selection.GSE_OT_select, selection.GSE_OT_selection_action, selection.GSE_OT_selection_tool, selection.GSE_OT_selection_depth, selection.GSE_OT_brush_action) + ui.classes[1:] + (reference.GSE_PT_reference,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gse = bpy.props.PointerProperty(type=ui.GSE_Props)
    reference.register()
    selection.register()
    bpy.app.handlers.load_pre.append(export_formats.shutdown)
    # 后台查一次 GitHub Releases：不挡启动，不联网也照常能用
    if not bpy.app.background:
        update_checker.check_for_updates()


def unregister():
    export_formats.shutdown()
    if export_formats.shutdown in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(export_formats.shutdown)
    selection.unregister()
    reference.unregister()
    update_checker.stop()
    del bpy.types.Scene.gse
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
