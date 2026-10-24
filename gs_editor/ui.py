import bpy

from . import update_checker
from .ops import (TOOL_ITEMS, DEPTH_ITEMS, STATE, SHORTCUTS_BRUSH, SHORTCUTS_COVER)
from .splats import is_splat_object


class GSE_Props(bpy.types.PropertyGroup):
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


def draw_update_banner(layout):
    """版本 + 检查状态 + 刷新；有新版再多一条「一键更新」。主面板和偏好设置共用。"""
    status = update_checker.get_check_status()
    state = status["status"]
    row = layout.row(align=True)
    row.label(text="v" + update_checker.version_text(), icon='PACKAGE')
    if state == 'checking':
        row.label(text="检查中…", icon='SORTTIME')
    elif state == 'no_update':
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
        draw_update_banner(layout)
        p = context.scene.gse
        obj = context.active_object
        if not is_splat_object(obj):
            layout.label(text="请选择一个高斯泼溅对象（点云）", icon='INFO')
            layout.operator("wm.ply_import", text="导入 PLY", icon='IMPORT')
            layout.operator("wm.spz_import", text="导入 SPZ", icon='IMPORT')
            return
        layout.label(text=f"{obj.name}：{len(obj.data.points):,} 个泼溅点", icon='POINTCLOUD_DATA')

        grid = layout.grid_flow(columns=3, align=True, even_columns=True)
        for ident, name, _desc, icon, _order in TOOL_ITEMS:
            # 只切工具，不开始涂抹；真正开画要点下面的「开始涂抹 / 开始覆盖」
            op = grid.operator("gse.set_tool", text=name, icon=icon, depress=(p.tool == ident))
            op.tool = ident

        row = layout.row()
        row.scale_y = 1.4
        if STATE["running"]:
            row.operator("gse.stop_brush", text="停止笔刷 (Esc)", icon='CANCEL')
        elif STATE["cover"]:
            row.operator("gse.stop_cover", text="停止覆盖 (Esc)", icon='CANCEL')
        elif p.tool == 'COVER':
            row.operator("gse.cover", text="开始覆盖", icon='MESH_PLANE')
        else:
            row.operator("gse.brush", text="开始涂抹", icon='BRUSH_DATA').tool = 'KEEP'


class _SubPanel:
    bl_parent_id = "GSE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "GS 编辑器"

    @classmethod
    def poll(cls, context):
        return is_splat_object(context.active_object)


class _ToolPanel(_SubPanel):
    """当前工具的参数面板：同一套行式排版，标题跟着工具换。"""

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False       # 不要右侧的动画圆点，两套面板都干净
        self.draw_params(context, layout.column())

    def draw_params(self, context, col):
        raise NotImplementedError


class GSE_PT_brush(_ToolPanel, bpy.types.Panel):
    bl_label = "笔刷"
    bl_idname = "GSE_PT_brush"

    @classmethod
    def poll(cls, context):
        return (is_splat_object(context.active_object)
                and context.scene.gse.tool != 'COVER')

    def draw_params(self, context, col):
        p = context.scene.gse
        col.prop(p, "radius_px")
        col.prop(p, "feather", slider=True)
        if p.tool == 'ERASE':
            col.prop(p, "strength", slider=True)
        # 高频切换：表面 / 穿透并排两个按钮，不藏在下拉菜单里
        row = col.row(align=True)
        split = row.split(factor=getattr(col, "property_split_factor", 0.4), align=True)
        label = split.column()
        label.alignment = 'RIGHT'
        label.label(text="选择深度")
        buttons = split.row(align=True)
        for ident, name, _desc in DEPTH_ITEMS:
            op = buttons.operator("gse.set_depth", text=name, depress=(p.depth_mode == ident))
            op.mode = ident
        if p.tool == 'RESTORE':
            col.prop(p, 'restore_erased')
            col.prop(p, 'remove_added')
        col.prop(p, "spacing")


class GSE_PT_cover(_ToolPanel, bpy.types.Panel):
    bl_label = "覆盖"
    bl_idname = "GSE_PT_cover"

    @classmethod
    def poll(cls, context):
        return (is_splat_object(context.active_object)
                and context.scene.gse.tool == 'COVER')

    def draw_params(self, context, col):
        p = context.scene.gse
        col.prop(p, 'cover_fit', slider=True)
        col.prop(p, 'cover_offset')
        col.prop(p, 'cover_subdiv')
        col.prop(p, 'cover_size_from_screen')
        col.prop(p, 'cover_sample_color')
        if p.cover_sample_color:
            col.prop(p, 'cover_ring')
        else:
            col.prop(p, 'cover_color')


class GSE_PT_shortcuts(_SubPanel, bpy.types.Panel):
    bl_label = "快捷键"
    bl_idname = "GSE_PT_shortcuts"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        p = context.scene.gse
        cover_mode = p.tool == 'COVER'
        col = self.layout.column(align=True)
        col.label(text="覆盖时可用：" if cover_mode else "涂抹时可用：")
        for keys, what in (SHORTCUTS_COVER if cover_mode else SHORTCUTS_BRUSH):
            row = col.row()
            split = row.split(factor=0.40)
            split.label(text=keys)
            split.label(text=what)


class GSE_PT_export(_SubPanel, bpy.types.Panel):
    bl_label = "烘焙、合并与导出"
    bl_idname = "GSE_PT_export"

    def draw(self, context):
        layout = self.layout
        layout.operator("gse.bake_edits", icon='CHECKMARK')
        layout.operator("gse.merge_splats", icon='AUTOMERGE_ON')
        layout.operator("gse.export_ply", icon='EXPORT')
        layout.label(text="覆盖生成的是网格面片，不写入 PLY", icon='INFO')


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


classes = (GSE_Props, GSE_PT_panel, GSE_PT_brush, GSE_PT_cover,
           GSE_PT_shortcuts, GSE_PT_export, GSE_AddonPreferences)
