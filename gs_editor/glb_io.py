"""把覆盖面片合并成一个网格，导出 GLB。

覆盖生成的是各自独立的网格物体，每块一份材质，位置和朝向存在物体变换里。这里按
世界坐标把它们并成一份网格，再交给 Blender 自带的 glTF 导出器写 GLB：合并用的是
临时物体，写完就删，场景里的面片一块都不动。
"""

import bpy
import numpy as np


def merged_mesh(context, patches, name="覆盖面片合集"):
    """把若干面片按世界坐标并成一份网格（顶点已经是世界坐标，材质按块保留）。"""
    depsgraph = context.evaluated_depsgraph_get()
    verts, faces, face_mat = [], [], []
    mats, slot_of = [], {}

    for ob in patches:
        ob_eval = ob.evaluated_get(depsgraph)      # 带修改器、约束的最终形状
        mesh = ob_eval.to_mesh()
        try:
            n = len(mesh.vertices)
            base = len(verts)
            if n:
                co = np.empty(n * 3, np.float64)
                mesh.vertices.foreach_get("co", co)
                M = np.array(ob_eval.matrix_world, np.float64)
                co = co.reshape(n, 3) @ M[:3, :3].T + M[:3, 3]
                verts.extend(co.tolist())
            remap = []
            for mat in mesh.materials:
                if mat is None:
                    remap.append(0)
                elif mat.name in slot_of:
                    remap.append(slot_of[mat.name])
                else:
                    slot_of[mat.name] = len(mats)
                    remap.append(len(mats))
                    mats.append(mat)
            for poly in mesh.polygons:
                faces.append(tuple(base + i for i in poly.vertices))
                face_mat.append(remap[poly.material_index] if remap else 0)
        finally:
            ob_eval.to_mesh_clear()

    out = bpy.data.meshes.new(name)
    out.from_pydata(verts, [], faces)
    out.validate()
    for mat in mats:
        out.materials.append(mat)
    if face_mat:
        out.polygons.foreach_set("material_index", np.array(face_mat, np.int32))
    out.update()
    return out


def write_patches_glb(context, path, patches, yup=True):
    """并成一份网格写 GLB，返回 (块数, 顶点数)。临时物体用完就删，选择状态照原样还回去。

    yup: 按 glTF 的 Y-up 写（网页端用）；False 保持 Blender 的 Z-up，跟 PLY 的坐标系一致。"""
    view_layer = context.view_layer
    active = view_layer.objects.active
    active_name = active.name if active is not None else None
    selected = [o.name for o in view_layer.objects if o.select_get()]

    mesh = merged_mesh(context, patches)
    ob = bpy.data.objects.new(mesh.name, mesh)
    context.scene.collection.objects.link(ob)
    try:
        for other in view_layer.objects:
            other.select_set(False)
        ob.select_set(True)
        view_layer.objects.active = ob
        bpy.ops.export_scene.gltf(
            filepath=path, export_format='GLB', use_selection=True,
            export_apply=True, export_materials='EXPORT', export_yup=yup)
        return len(patches), len(mesh.vertices)
    finally:
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(mesh)
        for name in selected:
            other = bpy.data.objects.get(name)
            if other is None:
                continue
            try:
                other.select_set(True)
            except RuntimeError:              # 不在当前视图层，跳过
                pass
        restored = bpy.data.objects.get(active_name) if active_name else None
        if restored is not None:
            try:
                view_layer.objects.active = restored
            except RuntimeError:
                pass
