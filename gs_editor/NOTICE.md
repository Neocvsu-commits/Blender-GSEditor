# GS Editor —— 来源与许可

GS Editor 是 [Gaussian Splat Patch](https://github.com/timfennell/blender-splat-patch) 0.1.15
（作者 Tim Fennell，基线提交 `78c2890`）的衍生作品，二者同为 GPL-3.0-or-later。

## 来自上游的部分

- 高斯泼溅点云的数据结构、读写与 PLY 编码：`splats.py` `ply_io.py` `stash.py` `merge.py`；
- 擦除 / 还原 / 选择 / 烘焙 / 合并 / 导出的算法层与屏幕空间拾取：`ops.py` `core.py` `view.py`；
- `Scene` 属性的组织方式与操作符骨架。

上游代码按 GPL-3.0-or-later 使用，上游署名保留。

## GS Editor 的改动

- 选择、擦除和还原的中文界面与原生选区联动；多格式及多级 LOD 导出；
- 早期版本新增的覆盖面片功能及附带 GLB 导出已在 v1.8.0 移除，已有场景网格保持原样；
- 交互细节：工具按钮只切换工具不自动开画、快捷键面板按工具分组、两套参数面板同构；
- 全中文文案。

## 数据兼容性说明

点云上的内部数据属性沿用上游的 `gsp_` 前缀（`gsp_selected`、`gsp_added`、`gsp_faded`、
`gsp_hidden`、`gsp_orig_opacity`、`gsp_revision`、`gsp_stash`）。这样同一个 `.blend`
在上游版本与 GS Editor 之间来回打开时，擦除记录与还原历史不会丢。

## 许可

GPL-3.0-or-later，完整文本见 `LICENSE`。
# Additional component (v1.6.0)

Google WebP 1.6.0 `cwebp` is redistributed unmodified under its BSD license,
with the license and patent grant in `gs_editor/vendor/webp/COPYING` and `PATENTS`.
Distribution URLs and archive checksums are in `gs_editor/vendor/webp/README.md`.

## LCC / LCC2 data organization (v1.7.0)

Data Organization Format originated from XGRIDS

GS Editor independently writes the published LCC and LCC2 organizations. This
does not represent an official XGRIDS implementation or endorsement. The
organization itself is not extended; spatial partitioning and approximate
Gaussian LOD generation are performed by GS Editor.

The data organization formats are used and distributed under authorization
in their respective White Papers. XGRIDS and its affiliates retain their
intellectual property and other rights in those formats:

- LCC specification, license and restrictions: https://github.com/xgrids/LCCWhitepaper
- LCC2 specification, license and restrictions: https://github.com/xgrids/LCC2Whitepaper

Those format conditions apply in addition to this implementation's GPL source
license; they are not an MIT grant. Exported packages include FORMAT_NOTICE.txt
with attribution, the White Paper link and redistribution/use conditions.
Retain that notice when sharing and display the required attribution in the
consuming application. In particular, without prior written consent, these
formats must not be used for training or fine-tuning an AI model competing
directly or indirectly with XGRIDS products or services; inform recipients
in writing and incorporate this restriction into applicable agreements.
The White Papers also specify public-availability obligations for derivative
formats or content and further restrictions; the linked full terms apply.
