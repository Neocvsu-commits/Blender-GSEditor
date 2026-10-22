"""GS Editor：在 Blender 5.3+ 里编辑高斯泼溅点云 —— 擦除、还原，并用吸附到表面的矩形面片覆盖区域。"""

import bpy

from . import ops, ui

_classes = ui.classes[:1] + ops.classes + ui.classes[1:]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gse = bpy.props.PointerProperty(type=ui.GSE_Props)


def unregister():
    ops.STATE["running"] = False
    ops.STATE["cover"] = False
    del bpy.types.Scene.gse
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
