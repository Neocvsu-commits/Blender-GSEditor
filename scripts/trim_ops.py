# -*- coding: utf-8 -*-
"""精简 ops.py：按 AST 行号整块删除不再使用的类和画笔方法。

删的都是整段定义（含装饰器），不重新排版其余代码，其余部分逐字节保持原样。
分支级别的删减在文件里手工做，这里只管"整块拿掉"。
"""

import ast
import pathlib
import sys

OPS = pathlib.Path(__file__).resolve().parents[1] / "gs_editor" / "ops.py"

DROP_CLASSES = [
    "GSE_OT_select_all",
    "GSE_OT_select_faint",
    "GSE_OT_select_floaters",
    "GSE_OT_select_cursor_sphere",
    "GSE_OT_delete_selected",
    "GSE_OT_fill_selected",
    "GSE_OT_source_from_cursor",
    "_SplatOp",
]

DROP_METHODS = {
    "GSE_OT_brush": [
        "end_bridge",        # 桥接
        "view_fill",
        "visible_hole",
        "end_spot",          # 污点修复
        "source_point",      # 克隆/愈合采样点
        "delete_selection",  # 删除选区内的高斯
    ],
}

DROP_FUNCTIONS = ["draw_selection"]


def node_span(node):
    decos = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(decos + [node.lineno]), node.end_lineno


def main():
    text = OPS.read_text(encoding="utf-8")
    tree = ast.parse(text)
    spans = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in DROP_CLASSES:
            spans.append((node_span(node), "class " + node.name))
        elif isinstance(node, ast.FunctionDef) and node.name in DROP_FUNCTIONS:
            spans.append((node_span(node), "def " + node.name))
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_SEL_CACHE"
                                                  for t in node.targets):
            spans.append((node_span(node), "_SEL_CACHE"))
        if isinstance(node, ast.ClassDef) and node.name in DROP_METHODS:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in DROP_METHODS[node.name]:
                    spans.append((node_span(sub), "%s.%s" % (node.name, sub.name)))

    if not spans:
        raise SystemExit("没有匹配到要删除的定义，检查 ops.py 是否已改动")

    lines = text.splitlines(keepends=True)
    drop = set()
    for (start, end), label in spans:
        for ln in range(start, end + 1):
            drop.add(ln)
        print("删除 %-40s 行 %d-%d" % (label, start, end))

    out = [line for i, line in enumerate(lines, 1) if i not in drop]
    OPS.write_text("".join(out), encoding="utf-8")
    print("剩余行数: %d -> %d" % (len(lines), len(out)))


if __name__ == "__main__":
    sys.exit(main())
