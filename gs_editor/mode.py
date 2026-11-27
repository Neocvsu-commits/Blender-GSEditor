"""Explicit native mode switching after safely ending GS modal tools."""

import bpy

from .splats import is_splat_object


class GSE_OT_set_mode(bpy.types.Operator):
    bl_idname = 'gse.set_mode'
    bl_label = '切换高斯编辑模式'
    bl_description = '结束当前选择或笔刷，再切换 Blender 原生模式；保留已完成的编辑'

    mode: bpy.props.EnumProperty(
        name='模式', items=(('OBJECT', '物体模式', '结束当前工具并返回物体模式'),
                           ('EDIT', '编辑模式', '进入高斯点云编辑模式')),
        default='OBJECT')

    @classmethod
    def poll(cls, context):
        return (is_splat_object(context.active_object)
                and context.mode in {'OBJECT', 'EDIT_POINTCLOUD'}
                and bpy.ops.object.mode_set.poll())

    def execute(self, context):
        from . import ops, selection
        obj = context.active_object
        if self.mode == 'EDIT' and (obj.data.library or obj.data.users > 1):
            self.report({'ERROR'}, '请先将高斯数据设为本地单用户数据')
            return {'CANCELLED'}
        try:
            # The sidebar is a UI region. Use the viewport's window region for
            # native mode_set while preserving the user's active object.
            region = next((r for r in context.area.regions if r.type == 'WINDOW'), None) \
                if context.area and context.area.type == 'VIEW_3D' else None
            override = {'region': region} if region else {}
            with context.temp_override(**override):
                selection.finish_for_mode_switch(context)
                ops.finish_for_mode_switch(context)
                result = bpy.ops.object.mode_set(mode=self.mode)
            if result != {'FINISHED'}:
                self.report({'ERROR'}, 'Blender 未能切换模式')
                return {'CANCELLED'}
            if context.area:
                context.area.tag_redraw()
            return {'FINISHED'}
        except (RuntimeError, ReferenceError, ValueError) as ex:
            self.report({'ERROR'}, f'切换模式失败：{ex}')
            return {'CANCELLED'}


def draw_mode_status(layout, context):
    """Derive status from Blender, never from an independent saved toggle."""
    row = layout.row(align=True)
    if context.mode == 'EDIT_POINTCLOUD':
        row.label(text='编辑模式', icon='EDITMODE_HLT')
        row.operator('gse.set_mode', text='返回物体模式', icon='OBJECT_DATAMODE').mode = 'OBJECT'
    else:
        row.label(text='物体模式', icon='OBJECT_DATAMODE')
        row.operator('gse.set_mode', text='编辑高斯', icon='EDITMODE_HLT').mode = 'EDIT'


classes = (GSE_OT_set_mode,)
