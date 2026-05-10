#!/usr/bin/env python3
"""
读取主文件夹下各子目录中的 metrics.txt（由 image_quality_assess 等生成），
提取各子文件夹的平均 UIQM / UCIQE / Luminance，再在主文件夹尺度上汇总，
将结果写入主目录下的 metrics_summary.txt。

默认：各子文件夹的平均值再取算术平均（每个子文件夹权重相同）。
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

FLOAT_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"

AVG_BLOCK_RE = re.compile(
    rf"(?:^|\n)===\s*Average\s*===\s*\n"
    rf"UIQM:\s*({FLOAT_RE})\s*\n"
    rf"UCIQE:\s*({FLOAT_RE})\s*\n"
    rf"(?:Luminance:\s*({FLOAT_RE})\s*)?",
    re.IGNORECASE | re.MULTILINE,
)

PER_IMAGE_RE = re.compile(
    rf"uiqm=({FLOAT_RE})\s+uciqe=({FLOAT_RE})"
    rf"(?:\s+luminance=({FLOAT_RE}))?",
    re.IGNORECASE,
)


def _parse_float(s: str) -> float:
    return float(s.strip())


def parse_metrics_txt(path: Path) -> tuple[float, float, float | None, str]:
    """
    返回 (uiqm, uciqe, luminance_or_None, source说明)。
    优先解析 === Average ===；否则对文件中所有 per-image 行求平均。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    m = AVG_BLOCK_RE.search(text)
    if m:
        uiqm = _parse_float(m.group(1))
        uciqe = _parse_float(m.group(2))
        lum_s = m.group(3)
        lum = _parse_float(lum_s) if lum_s else None
        return uiqm, uciqe, lum, "=== Average ==="

    rows = []
    for pm in PER_IMAGE_RE.finditer(text):
        u, c = _parse_float(pm.group(1)), _parse_float(pm.group(2))
        l_raw = pm.group(3)
        lum = _parse_float(l_raw) if l_raw else None
        rows.append((u, c, lum))
    if not rows:
        raise ValueError("未找到 === Average === 且无任何 uiqm= / uciqe= 行")

    uiqms = [r[0] for r in rows]
    uciqes = [r[1] for r in rows]
    lums = [r[2] for r in rows if r[2] is not None]
    mean_lum = statistics.mean(lums) if lums else None
    return (
        statistics.mean(uiqms),
        statistics.mean(uciqes),
        mean_lum,
        f"由 {len(rows)} 条逐图记录回退计算",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="汇总各子文件夹 metrics.txt 中的平均指标，写入主目录 metrics_summary.txt。"
    )
    p.add_argument(
        "root",
        type=Path,
        help="主文件夹，例如 Dataset（其下含 Camera1、Camera2…各含 metrics.txt）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（默认：<主文件夹>/metrics_summary.txt）",
    )
    args = p.parse_args()
    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"不是有效文件夹: {root}", file=sys.stderr)
        return 1

    out_path = (
        args.output.expanduser().resolve()
        if args.output
        else root / "metrics_summary.txt"
    )

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"主文件夹下没有子目录: {root}", file=sys.stderr)
        return 1

    records: list[tuple[str, float, float, float | None, str]] = []
    errors: list[str] = []

    for d in subdirs:
        mt = d / "metrics.txt"
        if not mt.is_file():
            errors.append(f"[跳过] {d.name}: 无 metrics.txt")
            continue
        try:
            u, c, l, src = parse_metrics_txt(mt)
            records.append((d.name, u, c, l, src))
        except Exception as e:
            errors.append(f"[失败] {d.name}: {e}")

    for msg in errors:
        print(msg, file=sys.stderr)

    if not records:
        print("没有可用的子文件夹指标，未生成汇总文件。", file=sys.stderr)
        return 1

    uiqms = [r[1] for r in records]
    uciqes = [r[2] for r in records]
    lums = [r[3] for r in records if r[3] is not None]

    mean_uiqm = statistics.mean(uiqms)
    mean_uciqe = statistics.mean(uciqes)
    mean_lum = statistics.mean(lums) if lums else None

    lines = [
        "# 主文件夹指标汇总（各子文件夹平均后再整体平均）",
        f"# 主路径: {root}",
        f"# 参与子文件夹数: {len(records)}",
        "",
        "## 各子文件夹",
        "",
    ]
    for name, u, c, l, src in records:
        lines.append(f"- {name}")
        lines.append(f"  UIQM: {u:.6f}")
        lines.append(f"  UCIQE: {c:.6f}")
        if l is not None:
            lines.append(f"  Luminance: {l:.6f}")
        else:
            lines.append("  Luminance: (无)")
        lines.append(f"  来源: {src}")
        lines.append("")

    lines.extend(
        [
            "## 全数据集（子文件夹平均的算术平均）",
            "",
            f"UIQM: {mean_uiqm:.6f}",
            f"UCIQE: {mean_uciqe:.6f}",
        ]
    )
    if mean_lum is not None:
        lines.append(f"Luminance: {mean_lum:.6f}")
    else:
        lines.append("Luminance: (各子目录均无此项，未汇总)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
