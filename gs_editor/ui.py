import bpy

from . import update_checker, reference, selection
from .ops import (TOOL_ITEMS, DEPTH_ITEMS, STATE, SHORTCUTS_BRUSH, SHORTCUTS_COVER)
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

    # 覆盖
    cover_fit: bpy.props.FloatProperty(
        name="贴合表面", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
        description="1 = 面片整块旋转对齐表面朝向；0 = 正对屏幕；中间为插值")
    cover_offset: bpy.props.FloatProperty(
        name="面片偏移", default=0.0, min=0.0, unit='LENGTH',
        description="面片沿表面法线往相机方向让开多少（0 = 按面片尺寸自动取 0.6%），防和点云穿插")
    cover_size_from_screen: bpy.props.BoolProperty(
        name="尺寸按屏幕投影", default=False,
        description="关（默认）：框多大就生成多大，斜看时面片投影会短一截（透视）"
                    "；开：反过来放大面片，让它在屏幕上正好盖住框选矩形")
    cover_subdiv: bpy.props.IntProperty(
        name="面片细分", default=2, min=0, max=8,
        description="面片的网格段数：越大越贴合曲面起伏，也越吃性能")
    cover_sample_color: bpy.props.BoolProperty(
        name="取周围颜色", default=True,
        description="取框选外面一圈里占比最大的颜色作为面片颜色，这样能和周围表面接上；"
                    "关掉改用下面的固定颜色")
    cover_ring: bpy.props.FloatProperty(
        name="取样环宽", default=0.35, min=0.05, max=2.0, subtype='FACTOR',
        description="往外扩多少（相对框的长宽）作为取样环；周围颜色混杂时调小一点更准")
    cover_color: bpy.props.FloatVectorProperty(
        name="面片颜色", subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.8, 0.8, 0.8, 1.0),
        description="关闭采样时使用的固定颜色")

    # 导出
    export_up_axis: bpy.props.EnumProperty(
        name="坐标轴",
        items=(('YUP', "Y 轴朝上（glTF / 网页）",
                "转成 glTF、three.js 和网页端泼溅查看器用的 Y-up，与同名的 GLB 同坐标系"),
               ('ZUP', "Z 轴朝上（Blender）",
                "保持 Blender 原生的 Z-up：再导回 Blender 时用它，网页端会躺着")),
        default='YUP',
        description="PLY 和 GLB 按同一个坐标系导出。网页端是 Y-up，选错会让泼溅点整个躺倒、"
                    "并相对场景偏移十几米")
    export_patches_glb: bpy.props.BoolProperty(
        name="同时导出覆盖面片 GLB", default=True,
        description="把选中的网格合并成一个网格，和导出的 PLY 同名存成 .glb。"
                    "覆盖面片的判定只有一条：选中的网格（点云本身除外）；没选中就不生成文件")


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
        if STATE["running"]:
            label = '结束还原' if context.scene.gse.tool == 'RESTORE' else '结束涂抹'
            draw_brush_params(context, layout.box(), context.scene.gse.tool)
            row = layout.row()
            row.scale_y = 1.2
            row.operator('gse.stop_brush', text=label, icon='CHECKMARK')
        elif STATE["cover"]:
            draw_cover_params(context, layout.box())
            row = layout.row()
            row.scale_y = 1.2
            row.operator('gse.stop_cover', text='结束覆盖', icon='CHECKMARK')
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


def draw_cover_params(context, layout):
    col = layout.column()
    col.use_property_split = True
    col.use_property_decorate = False
    p = context.scene.gse
    col.prop(p, 'cover_fit', slider=True)
    col.prop(p, 'cover_offset')
    col.prop(p, 'cover_subdiv')
    col.prop(p, 'cover_size_from_screen')
    col.prop(p, 'cover_sample_color')
    col.prop(p, 'cover_ring' if p.cover_sample_color else 'cover_color')


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
        p = context.scene.gse
        cover_mode = p.tool == 'COVER'
        col = self.layout.column(align=True)
        for keys, what in (SHORTCUTS_COVER if cover_mode else SHORTCUTS_BRUSH):
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
        layout.enabled = selection._active is None and not any(STATE.values())
        row = layout.row(align=True)
        row.operator('gse.export_ply', text='导出 PLY', icon='EXPORT')
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
        layout.prop(context.scene.gse, 'export_patches_glb')


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
