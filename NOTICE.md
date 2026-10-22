# GS Editor —— 来源与许可

GS Editor 是 [Gaussian Splat Patch](https://github.com/timfennell/blender-splat-patch) 0.1.15
（作者 Tim Fennell，基线提交 `78c2890`）的衍生作品，二者同为 GPL-3.0-or-later。

## 来自上游的部分

- 高斯泼溅点云的数据结构、读写与 PLY 编码：`splats.py` `ply_io.py` `stash.py` `merge.py`；
- 擦除 / 还原 / 选择 / 烘焙 / 合并 / 导出的算法层与屏幕空间拾取：`ops.py` `core.py` `view.py`；
- `Scene` 属性的组织方式与操作符骨架。

上游代码按 GPL-3.0-or-later 使用，上游署名保留。

## GS Editor 的改动

- 工具精简为 **擦除 / 还原 / 覆盖** 三个；界面重做为中文面板（主面板 / 笔刷·覆盖 / 快捷键 / 烘焙导出）；
- 新增**覆盖**（`cover.py`）：贴合表面的刚性矩形面片，自动吸附到局部表面、跟着鼠标滑动、左键确认，
  手动旋转作为兜底；颜色取框外一圈的主流色；
- 交互细节：工具按钮只切换工具不自动开画、快捷键面板按工具分组、两套参数面板同构；
- 全中文文案。

## 数据兼容性说明

点云上的内部数据属性沿用上游的 `gsp_` 前缀（`gsp_selected`、`gsp_added`、`gsp_faded`、
`gsp_hidden`、`gsp_orig_opacity`、`gsp_revision`、`gsp_stash`）。这样同一个 `.blend`
在上游版本与 GS Editor 之间来回打开时，擦除记录与还原历史不会丢。

## 许可

GPL-3.0-or-later，完整文本见 `LICENSE`。
