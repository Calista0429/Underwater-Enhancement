"""
analyze_lines.py
================
以 Line 为尺度，批量读取 ROOT_DIR 下所有子目录
（Left_Line*, Right_Line*, 以及任何同时含有
 camera_trails.csv + metrics.csv 的子目录）
对 Luminance、UCIQE、UIQM 三个指标进行多维度对比分析。

目录结构示例:
    <ROOT_DIR>/
        Left_Line1/   camera_trails.csv  metrics.csv
        Left_Line2/   camera_trails.csv  metrics.csv
        Right_Line1/  camera_trails.csv  metrics.csv
        Right_Line2/  camera_trails.csv  metrics.csv
        ...

使用方法:
    1. 修改下方 ROOT_DIR
    2. pip install pandas matplotlib scipy numpy
    3. python analyze_lines.py
    4. 结果保存在 <ROOT_DIR>/analysis_output/
"""

import os
import re
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ══════════════════════════════════════════════════════════════
# ★  修改这里：你的数据根目录
# ══════════════════════════════════════════════════════════════
ROOT_DIR = r"F:\Dataset\2026.2.15\Dataset\Underwater"
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(ROOT_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ['Luminance', 'UCIQE', 'UIQM']
METRIC_LABELS = {
    'Luminance': 'Luminance (brightness)',
    'UCIQE':     'UCIQE (color quality)',
    'UIQM':      'UIQM (overall quality)',
}

BG = '#F8F8F6'

PALETTE_LEFT = [
    '#3B8BD4','#1D9E75','#534AB7','#0C447C','#1ABC9C',
    '#2980B9','#16A085','#7B68EE','#5DADE2','#48C9B0',
]
PALETTE_RIGHT = [
    '#D85A30','#BA7517','#C0392B','#D35400','#8E44AD',
    '#E74C3C','#F39C12','#CB4335','#DC7633','#A93226',
]
PALETTE_OTHER = ['#7F8C8D','#95A5A6','#BDC3C7','#566573','#717D7E']


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def parse_filename(fname):
    m = re.match(r'.+_(\d{2}-\d{2}-\d{2})_(Camera\d+)_(\d+)\.jpg', str(fname))
    return (m.group(2), int(m.group(3))) if m else (None, None)


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def side_of(line_id):
    low = line_id.lower()
    if 'right' in low: return 'Right'
    if 'left'  in low: return 'Left'
    return 'Other'


def load_line(line_dir, line_id):
    trails_path  = os.path.join(line_dir, 'camera_trails.csv')
    metrics_path = os.path.join(line_dir, 'metrics.csv')

    try:
        trails  = pd.read_csv(trails_path)
        metrics = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"  [skip] {line_id}: 读取失败 — {e}")
        return None

    metrics = metrics[metrics['File Name'] != 'AVERAGE'].copy()

    needed = {'File Name', 'Luminance', 'UCIQE', 'UIQM'}
    missing = needed - set(metrics.columns)
    if missing:
        print(f"  [skip] {line_id}: metrics.csv 缺少列 {missing}")
        return None

    metrics['camera']    = metrics['File Name'].apply(lambda x: parse_filename(x)[0])
    metrics['frame_idx'] = metrics['File Name'].apply(lambda x: parse_filename(x)[1])

    df = trails[['Number', 'X', 'Y']].merge(
        metrics[['File Name', 'Luminance', 'UCIQE', 'UIQM', 'camera', 'frame_idx']]
               .rename(columns={'File Name': 'Number'}),
        on='Number', how='inner'
    ).sort_values(['camera', 'frame_idx']).reset_index(drop=True)

    if df.empty:
        print(f"  [skip] {line_id}: 合并后无数据")
        return None

    # 累计距离（每个 camera 独立计算，避免 groupby.apply 丢列问题）
    dist_parts = []
    for cam in df['camera'].unique():
        mask = df['camera'] == cam
        sub  = df[mask].copy()
        dx, dy = sub['X'].diff().fillna(0), sub['Y'].diff().fillna(0)
        sub['dist_m'] = np.sqrt(dx**2 + dy**2).cumsum()
        dist_parts.append(sub[['Number', 'dist_m']])
    df = df.merge(pd.concat(dist_parts), on='Number', how='left')

    df['line_id'] = line_id
    df['side']    = side_of(line_id)
    return df


# ─────────────────────────────────────────────────────────────
# 1. 扫描并加载所有子目录
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  扫描目录:", ROOT_DIR)
print("=" * 60)

candidate_dirs = sorted(
    [d for d in glob.glob(os.path.join(ROOT_DIR, '*'))
     if os.path.isdir(d)
     and os.path.exists(os.path.join(d, 'camera_trails.csv'))
     and os.path.exists(os.path.join(d, 'metrics.csv'))],
    key=lambda p: natural_sort_key(os.path.basename(p))
)

if not candidate_dirs:
    raise FileNotFoundError(
        f"在 {ROOT_DIR} 的子目录中未找到同时包含 camera_trails.csv 和 "
        f"metrics.csv 的文件夹。\n请确认 ROOT_DIR 路径正确。"
    )

# 分配颜色：Left 用蓝系，Right 用橙系，Other 用灰系
left_ctr = right_ctr = other_ctr = 0
color_map = {}
all_dfs   = []

for d in candidate_dirs:
    lid = os.path.basename(d)
    print(f"  读取 {lid} ...")
    sub = load_line(d, lid)
    if sub is None:
        continue
    all_dfs.append(sub)
    side = side_of(lid)
    if side == 'Left':
        color_map[lid] = PALETTE_LEFT[left_ctr % len(PALETTE_LEFT)];   left_ctr  += 1
    elif side == 'Right':
        color_map[lid] = PALETTE_RIGHT[right_ctr % len(PALETTE_RIGHT)]; right_ctr += 1
    else:
        color_map[lid] = PALETTE_OTHER[other_ctr % len(PALETTE_OTHER)]; other_ctr += 1

if not all_dfs:
    raise RuntimeError("没有成功读取任何 Line 数据，请检查 csv 文件格式。")

df_all  = pd.concat(all_dfs, ignore_index=True)
lines   = list(dict.fromkeys(df_all['line_id']))
n_lines = len(lines)

left_lines  = [l for l in lines if side_of(l) == 'Left']
right_lines = [l for l in lines if side_of(l) == 'Right']
other_lines = [l for l in lines if side_of(l) == 'Other']

print(f"\n共加载: {n_lines} 条 Line  "
      f"(Left={len(left_lines)}, Right={len(right_lines)}, Other={len(other_lines)})")
print(f"总图像数: {len(df_all)}\n")


# ─────────────────────────────────────────────────────────────
# 2. 统计表
# ─────────────────────────────────────────────────────────────

def compute_stats(series, prefix=''):
    s = series.dropna()
    return {
        f'{prefix}N':      len(s),
        f'{prefix}Mean':   round(s.mean(),   3),
        f'{prefix}Std':    round(s.std(),    3),
        f'{prefix}Min':    round(s.min(),    3),
        f'{prefix}Q25':    round(s.quantile(.25), 3),
        f'{prefix}Median': round(s.median(), 3),
        f'{prefix}Q75':    round(s.quantile(.75), 3),
        f'{prefix}Max':    round(s.max(),    3),
        f'{prefix}IQR':    round(s.quantile(.75) - s.quantile(.25), 3),
        f'{prefix}CV%':    round(s.std() / s.mean() * 100, 1) if s.mean() != 0 else np.nan,
    }

stats_rows = []
for lid in lines:
    sub = df_all[df_all['line_id'] == lid]
    row = {'Line': lid, 'Side': side_of(lid)}
    for m in METRICS:
        row.update(compute_stats(sub[m], prefix=f'{m}_'))
    stats_rows.append(row)

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(os.path.join(OUTPUT_DIR, 'line_stats.csv'), index=False)
print(f"统计表已保存: line_stats.csv\n")

def smean(lid, metric):
    return float(stats_df.loc[stats_df['Line'] == lid, f'{metric}_Mean'].values[0])

def sstd(lid, metric):
    return float(stats_df.loc[stats_df['Line'] == lid, f'{metric}_Std'].values[0])


def grid_shape(n, max_cols=4):
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols
    return rows, cols


# ══════════════════════════════════════════════════════════════
# 图1 — 三指标箱线图（每个指标一行，所有 Line 并排）
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(max(14, n_lines * 0.9), 14))
fig.patch.set_facecolor(BG)

for ax_idx, metric in enumerate(METRICS):
    ax = axes[ax_idx]; ax.set_facecolor(BG)
    bp_data = [df_all[df_all['line_id'] == lid][metric].dropna().values for lid in lines]

    bps = ax.boxplot(bp_data, patch_artist=True, widths=0.55,
                     medianprops=dict(color='white', linewidth=2.5),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.35))

    for i, (patch, lid) in enumerate(zip(bps['boxes'], lines)):
        c = color_map[lid]
        patch.set_facecolor(c + '55'); patch.set_edgecolor(c); patch.set_linewidth(1.8)
        for j in range(2):
            bps['whiskers'][i*2+j].set_color(c)
            bps['caps'][i*2+j].set_color(c)

    means_pts = [np.mean(d) if len(d) > 0 else np.nan for d in bp_data]
    ax.scatter(range(1, n_lines+1), means_pts, marker='D', s=35,
               color=[color_map[l] for l in lines], zorder=5)

    ax.set_xticks(range(1, n_lines+1))
    ax.set_xticklabels(
        lines if ax_idx == 2 else [''] * n_lines,
        rotation=45, ha='right', fontsize=8
    )
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_title(f'{METRIC_LABELS[metric]} — distribution per Line  (◆ = mean)',
                 fontsize=11, fontweight='bold', pad=6)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    legend_items = [
        Patch(facecolor=PALETTE_LEFT[0]+'88',  edgecolor=PALETTE_LEFT[0],  label='Left lines'),
        Patch(facecolor=PALETTE_RIGHT[0]+'88', edgecolor=PALETTE_RIGHT[0], label='Right lines'),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc='upper right')

plt.suptitle('All metrics — boxplot per Line',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_boxplot_all_metrics.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ 图1 (三指标箱线图)")


# ══════════════════════════════════════════════════════════════
# 图2 — 均值±std 条形图（三个指标各一行 + 样本量）
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(4, 1, figsize=(max(14, n_lines * 0.9), 16),
                         gridspec_kw={'height_ratios': [3, 3, 3, 1.2]})
fig.patch.set_facecolor(BG)

x = np.arange(n_lines)
colors_arr = [color_map[l] for l in lines]

for ax_idx, metric in enumerate(METRICS):
    ax = axes[ax_idx]; ax.set_facecolor(BG)
    means_arr = np.array([smean(l, metric) for l in lines])
    stds_arr  = np.array([sstd(l,  metric) for l in lines])

    bars = ax.bar(x, means_arr, yerr=stds_arr, capsize=4,
                  color=[c + '55' for c in colors_arr],
                  edgecolor=colors_arr, linewidth=1.5,
                  error_kw=dict(elinewidth=1.3, ecolor='#777', capthick=1.3))
    for bar, mean, std in zip(bars, means_arr, stds_arr):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
                f'{mean:.2f}', ha='center', va='bottom', fontsize=7, color='#444')

    ax.set_xticks(x)
    ax.set_xticklabels([''] * n_lines)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'{METRIC_LABELS[metric]}  (mean ± std)',
                 fontsize=11, fontweight='bold', pad=6)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

ax_n = axes[3]; ax_n.set_facecolor(BG)
ns = stats_df.set_index('Line').loc[lines, 'Luminance_N'].values
ax_n.bar(x, ns, color=[c + '66' for c in colors_arr],
         edgecolor=colors_arr, linewidth=1.2)
for xi, n in zip(x, ns):
    ax_n.text(xi, n + 0.5, str(n), ha='center', va='bottom', fontsize=7)
ax_n.set_xticks(x)
ax_n.set_xticklabels(lines, rotation=45, ha='right', fontsize=8)
ax_n.set_ylabel('Image count', fontsize=9)
ax_n.grid(axis='y', alpha=0.3, linewidth=0.5)
ax_n.spines[['top', 'right']].set_visible(False)

plt.suptitle('Mean ± Std per Line — all metrics',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_mean_std_bar.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ 图2 (均值±std 条形图)")




# ══════════════════════════════════════════════════════════════
# 图7 — 空间轨迹地图（三指标三个子图，颜色=Line均值）
# ══════════════════════════════════════════════════════════════

SPATIAL_CMAPS = {'Luminance': 'Blues', 'UCIQE': 'Greens', 'UIQM': 'RdYlGn'}

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.patch.set_facecolor('#111122')

for ax_idx, metric in enumerate(METRICS):
    ax = axes[ax_idx]; ax.set_facecolor('#1a1a2e')
    mean_vals = np.array([smean(l, metric) for l in lines])
    vmin_g, vmax_g = mean_vals.min(), mean_vals.max()
    cmap_g = plt.get_cmap(SPATIAL_CMAPS[metric])
    norm_g = Normalize(vmin=vmin_g, vmax=vmax_g)

    for lid in lines:
        g = df_all[df_all['line_id'] == lid]
        c = cmap_g(norm_g(smean(lid, metric)))
        lw = 2.0 if side_of(lid) == 'Left' else 1.2
        ax.plot(g['X'], g['Y'], color=c, linewidth=lw, alpha=0.85)
        mid   = len(g) // 2
        short = lid.replace('Left_', 'L').replace('Right_', 'R')
        ax.text(g['X'].iloc[mid], g['Y'].iloc[mid], short,
                fontsize=5, color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='#00000099', edgecolor='none'))

    sm = ScalarMappable(cmap=cmap_g, norm=norm_g); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(f'Mean {metric}', fontsize=9, color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight='bold', color='white')
    ax.tick_params(colors='white', labelsize=6)
    ax.set_xlabel('X (m)', fontsize=8, color='white')
    ax.set_ylabel('Y (m)', fontsize=8, color='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#444')

plt.suptitle('Spatial trajectory map  (color = mean metric)  thick=Left, thin=Right',
             fontsize=12, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_spatial_map.png'),
            dpi=150, bbox_inches='tight', facecolor='#111122')
plt.close()
print("✓ 图7 (空间轨迹地图)")



# ─────────────────────────────────────────────────────────────
# 打印汇总
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  分析完成！输出文件：")
print(f"  {OUTPUT_DIR}/")