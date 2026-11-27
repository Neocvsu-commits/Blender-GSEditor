import bpy

from . import update_checker, reference, selection
from .ops import TOOL_ITEMS, DEPTH_ITEMS, STATE, SHORTCUTS_BRUSH
from .splats import is_splat_object


class GSE_Props(bpy.types.PropertyGroup):
    select_shape: bpy.props.EnumProperty(name="选择方式", items=selection.TOOLS, default='BOX')
    brush_action: bpy.props.EnumProperty(name='笔刷功能', items=selection.BRUSH_ACTIONS, default='SELECT')
    select_depth: bpy.props.EnumProperty(name="选择深度", items=[('SURFACE', '表面', '按当前高斯对象的屏幕深度筛选前层点'), ('THROUGH', '穿透', '选择范围内所有深度的点')], default='SURFACE')
    select_operation: bpy.props.EnumProperty(name="选区操作", items=[('REPLACE', '替换', '替换当前选区', 'SELECT_SET', 0), ('ADD', '加选', '追加到当前选区，也可按 Shift', 'SELECT_EXTEND', 1), ('SUBTRACT', '减选', '从当前选区移除，也可按 Ctrl', 'SELECT_SUBTRACT', 2)], default='REPLACE')
    select_radius: bpy.props.IntProperty(name="选择半径", default=40, min=1, max=1000, subtype='PIXEL')
    select_thickness: bpy.props.FloatProperty(name="表面厚度", default=0.02, min=0.00001, soft_max=1, unit='LENGTH', description="按 4 像素屏幕格的最近高斯点筛选，保留此深度差内的点；不检查其他网格的遮挡")
    reference_show: bpy.props.BoolProperty(name="显示高斯参考", default=True, update=reference.redraw)
    reference_limit: bpy.props.IntProperty(name="显示点数上限", default=200000, min=1000, max=5000000,
                                          description="超过上限时抽样显示，完整数据仍保存在 blend 中；提高上限会增加内存与视口开销", update=reference.redraw)
    tool: bpy.props.EnumProperty(name="工具", items=TOOL_ITEMS, default='ERASE')

    # 笔刷
    radius_px: bpy.props.IntProperty(name="画笔大小", subtype='PIXEL', default=40, min=3, max=1000,
                                     description="屏幕像素半径；涂抹时用 F 或 [ ] 调整")
    feather: bpy.props.FloatProperty(name="边缘羽化", default=0.35, min=0.0, max=1.0, subtype='FACTOR',
                                     description="羽化区越宽，边缘删掉的点越少；Shift+F 调整")
    strength: bpy.props.FloatProperty(name="擦除强度", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                                      description="每一笔删除命中点的比例；松开再画可继续累积")
    spacing: bpy.props.FloatProperty(name="笔触点间距", default=0.25, min=0.05, max=2.0,
                                     description="相邻笔触点之间的距离，按半径的比例计算")
    depth_mode: bpy.props.EnumProperty(name="选择深度", items=DEPTH_ITEMS, default='SURFACE',
                                       description="表面：只作用于鼠标命中处的球状区域；"
                                                   "穿透：作用于屏幕区域内所有深度的泼溅点")

    # 还原
    restore_erased: bpy.props.BoolProperty(
        name="还原被擦除点", default=True,
        description="恢复被编辑擦除或淡化的泼溅点（红色、黄色）")
    remove_added: bpy.props.BoolProperty(
        name="移除新增点", default=True,
        description="移除编辑过程中新增的泼溅点（绿色）")

    # 导出
    export_up_axis: bpy.props.EnumProperty(
        name="坐标轴",
        items=(('YUP', "Y 轴朝上（网页兼容 / glTF）",
                "将 Blender 世界坐标转为 Y-up，适用于直接按 Y 朝上显示的网页；"
                "LCC 查看器若自行转轴，应改选 Z 轴朝上"),
               ('ZUP', "Z 轴朝上（LCC 标准 / Blender）",
                "保留 Blender 世界坐标；符合 LCC 标准 Z-up 约定，也适合导回 Blender")),
        default='YUP',
        description="所有导出格式都遵从此设置；位置、旋转和高阶球谐统一转换。"
                    "LCC 标准为 Z-up，不自动转轴的网页可使用 Y-up 兼容模式")



def draw_update_banner(layout, compact=False):
    """版本 + 检查状态 + 刷新；有新版再多一条「一键更新」。主面板和偏好设置共用。"""
    status = update_checker.get_check_status()
    state = status["status"]
    row = layout.row(align=True)
    row.label(text="v" + update_checker.version_text(), icon='PACKAGE')
    if state == 'checking':
        row.label(text="检查中…", icon='SORTTIME')
    elif state == 'no_update' and not compact:
        row.label(text="已是最新", icon='CHECKMARK')
    elif state == 'installed':
        row.label(text="已下载，重启生效", icon='IMPORT')
    elif state == 'no_release':
        row.label(text="暂无发布版本", icon='INFO')
    elif state == 'no_access':
        row.label(text="在线访问已关闭", icon='INFO')
    elif state == 'error':
        row.label(text="检查失败", icon='ERROR')
    row.operator("gse.check_update", text="", icon='FILE_REFRESH')

    info = update_checker.get_update_info()
    if not info:
        return
    box = layout.box()
    box.alert = True
    col = box.column(align=True)
    col.label(text=f"有新版本 v{info['latest_version']}（当前 v{info['current_version']}）",
              icon='IMPORT')
    row = col.row(align=True)
    row.operator("gse.install_update", text="一键更新", icon='IMPORT')
    row.operator("wm.url_open", text="更新说明", icon='URL').url = info["html_url"]


class GSE_PT_panel(bpy.types.Panel):
    bl_label = "GS 编辑器"
    bl_idname = "GSE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS 编辑器"

    def draw(self, context):
        layout = self.layout
        draw_update_banner(layout, compact=True)
        obj = context.active_object
        if reference.is_reference(obj):
            layout.label(text=f'{len(obj.data.vertices):,} 个高斯点', icon='POINTCLOUD_DATA')
            selection.draw_controls(layout, context)
            return
        if bpy.app.version < (5, 3, 0):
            layout.operator('gse.import_reference', icon='IMPORT')
            return
        if not is_splat_object(obj):
            layout.label(text="请选择一个高斯泼溅对象（点云）", icon='INFO')
            layout.operator("wm.ply_import", text="导入 PLY", icon='IMPORT')
            layout.operator("wm.spz_import", text="导入 SPZ", icon='IMPORT')
            layout.operator('gse.import_reference', text='导入建模参考 PLY', icon='IMPORT')
            return
        layout.label(text=f"{len(obj.data.points):,} 个高斯点", icon='POINTCLOUD_DATA')
        from .mode import draw_mode_status
        draw_mode_status(layout, context)
        if STATE["running"]:
            label = '结束还原' if context.scene.gse.tool == 'RESTORE' else '结束涂抹'
            draw_brush_params(context, layout.box(), context.scene.gse.tool)
            row = layout.row()
            row.scale_y = 1.2
            row.operator('gse.stop_brush', text=label, icon='CHECKMARK')
        else:
            selection.draw_controls(layout, context)


class _SubPanel:
    bl_parent_id = "GSE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS 编辑器"

    @classmethod
    def poll(cls, context):
        return bpy.app.version >= (5, 3, 0) and is_splat_object(context.active_object)


def draw_brush_params(context, layout, tool):
    col = layout.column()
    col.use_property_split = True
    col.use_property_decorate = False
    p = context.scene.gse
    selection.parameter_row(col, '作用范围').prop(p, 'depth_mode', text='', expand=True)
    col.prop(p, 'radius_px', text='笔刷半径')
    col.prop(p, 'feather', slider=True)
    if tool == 'ERASE':
        col.prop(p, 'strength', slider=True)
    if tool == 'RESTORE':
        col.prop(p, 'restore_erased')
        col.prop(p, 'remove_added')
    else:
        col.prop(p, 'spacing')


class GSE_PT_advanced(_SubPanel, bpy.types.Panel):
    bl_label = '高级工具'
    bl_idname = 'GSE_PT_advanced'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 20

    @classmethod
    def poll(cls, context):
        return _SubPanel.poll(context) and not any(STATE.values())

    def draw(self, context):
        layout = self.layout
        layout.enabled = selection._active is None
        row = layout.row(align=True)
        row.operator('gse.bake_edits', icon='CHECKMARK')
        row.operator('gse.merge_splats', text='合并点云', icon='AUTOMERGE_ON')
        layout.operator('gse.import_reference', text='导入建模参考 PLY', icon='IMPORT')


class GSE_PT_shortcuts(_SubPanel, bpy.types.Panel):
    bl_label = "快捷键"
    bl_idname = "GSE_PT_shortcuts"
    bl_options = {'DEFAULT_CLOSED'}
    bl_parent_id = 'GSE_PT_advanced'

    @classmethod
    def poll(cls, context):
        return GSE_PT_advanced.poll(context)

    def draw(self, context):
        col = self.layout.column(align=True)
        for keys, what in SHORTCUTS_BRUSH:
            row = col.row()
            split = row.split(factor=0.40)
            split.label(text=keys)
            split.label(text=what)


class GSE_PT_export(_SubPanel, bpy.types.Panel):
    bl_label = "导出"
    bl_idname = "GSE_PT_export"
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        from . import export_formats
        if export_formats._job is not None:
            row = layout.row(align=True)
            row.label(text='正在导出 ' + export_formats._job.format, icon='TIME')
            row.operator('gse.cancel_export', text='', icon='CANCEL')
            return
        layout.enabled = selection._active is None and not any(STATE.values())
        row = layout.row(align=True)
        row.menu('GSE_MT_export', text='导出高斯', icon='EXPORT')
        row.popover(panel='GSE_PT_export_options', text='', icon='PREFERENCES')


class GSE_PT_export_options(bpy.types.Panel):
    bl_label = '导出设置'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(context.scene.gse, 'export_up_axis')


class GSE_AddonPreferences(bpy.types.AddonPreferences):
    """偏好设置：更新检查和反馈入口，和面板顶部共用同一条更新栏。"""
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        draw_update_banner(layout)
        layout.separator()
        box = layout.box()
        box.label(text="反馈 & 支持", icon='HELP')
        feedback = box.row(align=True)
        feedback.operator(
            "wm.url_open", text="Bug / 功能建议", icon='GHOST_ENABLED'
        ).url = "https://github.com/Neocvsu-commits/Blender-GSEditor/issues/new"
        feedback.operator(
            "wm.url_open", text="匿名反馈", icon='COMMUNITY'
        ).url = "https://docs.qq.com/form/page/DTnV6S25STkxJR0Zy"
        box.row().operator(
            "wm.url_open", text="⭐ 作者主页（了解更多工具）", icon='URL'
        ).url = "https://github.com/Neocvsu-commits"


classes = (GSE_Props, GSE_PT_panel, GSE_PT_export_options,
           GSE_PT_export, GSE_PT_advanced, GSE_PT_shortcuts, GSE_AddonPreferences)
