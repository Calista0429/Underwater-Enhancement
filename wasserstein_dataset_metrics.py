#!/usr/bin/env python3
"""
比较两个数据集（各自为「主文件夹 / 多个子文件夹 / metrics.txt」结构）在
UIQM、UCIQE、Luminance 三个一维指标上的 Wasserstein-1 距离。

从各 metrics.txt 中读取逐图行（跳过 === Average === 等汇总行）。
若两个指标向量长度不同：较短侧使用全部样本，较长侧无放回随机抽取与较短侧
相同数量，重复 --trials 次（默认 3）取距离平均；等长则直接在全量上算一次。

一维、等权经验分布、等样本量 n 时，W1 等于对样本排序后逐对绝对差的均值
（最优传输将第 i 个次序统计量配对）。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

FLOAT_TOKEN = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
PER_IMAGE_RE = re.compile(
    rf"uiqm=({FLOAT_TOKEN})\s+uciqe=({FLOAT_TOKEN})(?:\s+luminance=({FLOAT_TOKEN}))?",
    re.IGNORECASE,
)


def _find_metrics_files(dataset_root: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(dataset_root, "*", "metrics.txt")))
    root_file = os.path.join(dataset_root, "metrics.txt")
    if os.path.isfile(root_file):
        paths = [root_file] + [p for p in paths if p != root_file]
    return paths


def _parse_metrics_file(path: str) -> tuple[list[float], list[float], list[float]]:
    uiqm: list[float] = []
    uciqe: list[float] = []
    lum: list[float] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or "===" in s or s.lower().startswith("uiqm:"):
                continue
            m = PER_IMAGE_RE.search(s)
            if not m:
                continue
            uiqm.append(float(m.group(1)))
            uciqe.append(float(m.group(2)))
            if m.group(3) is not None:
                lum.append(float(m.group(3)))
    return uiqm, uciqe, lum


def collect_dataset(dataset_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = _find_metrics_files(dataset_root)
    if not files:
        raise FileNotFoundError(
            f"未找到 metrics.txt（已查 {dataset_root} 下子目录 */metrics.txt 及根目录 metrics.txt）"
        )
    u_all: list[float] = []
    c_all: list[float] = []
    l_all: list[float] = []
    for p in files:
        u, c, l = _parse_metrics_file(p)
        u_all.extend(u)
        c_all.extend(c)
        l_all.extend(l)
    if not u_all:
        raise ValueError(f"在 {dataset_root} 的 metrics.txt 中未解析到任何 uiqm=/uciqe= 行")
    if len(u_all) != len(c_all):
        raise RuntimeError("内部错误：UIQM 与 UCIQE 样本数不一致")
    return (
        np.asarray(u_all, dtype=np.float64),
        np.asarray(c_all, dtype=np.float64),
        np.asarray(l_all, dtype=np.float64),
    )


def wasserstein_1d_equal_weights(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    *,
    n_trials: int,
) -> tuple[float, str]:
    """
    一维 W1，等权经验测度。等长：一次计算；不等长：短侧全量，长侧无放回抽样
    n_trials 次取 W1 再平均。
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), "empty"

    if a.size == b.size:
        return float(np.mean(np.abs(np.sort(a) - np.sort(b)))), "equal_n_full"

    if a.size < b.size:
        small, large = a, b
    else:
        small, large = b, a
    n = small.size
    if large.size < n:
        return float("nan"), "empty"
    small_s = np.sort(small)
    trials = []
    for _ in range(n_trials):
        idx = rng.choice(large.size, size=n, replace=False)
        trials.append(np.mean(np.abs(small_s - np.sort(large[idx]))))
    return float(np.mean(trials)), f"subsample_n={n}_trials={n_trials}"


def main() -> int:
    p = argparse.ArgumentParser(
        description="基于各子文件夹 metrics.txt 中的逐图指标，计算两个数据集之间的 W1 距离。"
    )
    p.add_argument("dataset_a", help="数据集 A 根目录（含 Camera*/metrics.txt）")
    p.add_argument("dataset_b", help="数据集 B 根目录")
    p.add_argument("--seed", type=int, default=0, help="随机抽样种子（默认 0）")
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="样本量不等时，较大集随机抽样的重复次数（默认 3）",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="将结果追加写入该文本文件；不设则只打印",
    )
    args = p.parse_args()

    if args.trials < 1:
        print("--trials 须 >= 1", file=sys.stderr)
        return 1

    for name, path in ("A", args.dataset_a), ("B", args.dataset_b):
        if not os.path.isdir(path):
            print(f"数据集{name} 路径不是文件夹: {path}", file=sys.stderr)
            return 1

    try:
        ua, ca, la = collect_dataset(args.dataset_a)
        ub, cb, lb = collect_dataset(args.dataset_b)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    w_u, m_u = wasserstein_1d_equal_weights(ua, ub, rng, n_trials=args.trials)
    w_c, m_c = wasserstein_1d_equal_weights(ca, cb, rng, n_trials=args.trials)
    if la.size == 0 or lb.size == 0:
        w_l, m_l = float("nan"), "no_luminance_in_one_or_both"
    else:
        w_l, m_l = wasserstein_1d_equal_weights(la, lb, rng, n_trials=args.trials)

    lines = [
        "Wasserstein-1 (一维、等权经验分布)",
        f"  A: {os.path.abspath(args.dataset_a)}",
        f"  B: {os.path.abspath(args.dataset_b)}",
        f"  计数  UIQM: |A|={ua.size}  |B|={ub.size}",
        f"        UCIQE: |A|={ca.size}  |B|={cb.size}",
        f"        Luminance: |A|={la.size}  |B|={lb.size}（仅含带 luminance= 的行）",
        f"  随机种子: {args.seed}  不等长时抽样次数: {args.trials}",
        "",
        f"  W1(UIQM):       {w_u:.6f}   [{m_u}]",
        f"  W1(UCIQE):      {w_c:.6f}   [{m_c}]",
        f"  W1(Luminance):  {w_l:.6f}   [{m_l}]" if np.isfinite(w_l) else f"  W1(Luminance):  nan   [{m_l}]",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.output:
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
