"""Generate the figures embedded in YOLO-INT8-PTQ-CALIBRATION-RECIPE.md.

Reads result JSONs from the experimental directories and writes PNG
plots into this assets directory. Run once; commit the PNGs.

Usage:
    python make_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = Path('/data/yolov8/tmp_embed')

# ---------- shared style ----------
plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

OK = '#2a9d8f'
BAD = '#e76f51'
WARN = '#f4a261'
NEUTRAL = '#5b80a5'
GOLD = '#e9c46a'


# ============================================================
# Figure 1 — In-house 9 sampling x 3 calibrators heatmap
# ============================================================
def fig1_inhouse_heatmap():
    """Show how calibrator choice rearranges the sampling-strategy ranking."""
    # Pulled from logs/val_results.json + val_results_lit.json (MinMax)
    # logs/val_results_entropy_real.json (Entropy nb=2048)
    # logs/val_results_calibrator.json (Percentile 99.999, only 4 lists)
    REF = 0.9568  # FP32 ONNX mAP50

    samples = ['easy128', 'fps128', 'mixed128', 'selectq128',
               'hard128', 'stratified128', 'kmeans128',
               'random128', 'highnorm128']
    cals = ['MinMax', 'Entropy nb=2048', 'Percentile 99.999']

    # mAP50 numbers (None where not run)
    data = {
        ('easy128', 'MinMax'): 0.9265,
        ('fps128', 'MinMax'): 0.9000,
        ('mixed128', 'MinMax'): 0.8789,
        ('selectq128', 'MinMax'): 0.4835,
        ('hard128', 'MinMax'): 0.2827,
        ('stratified128', 'MinMax'): 0.4721,
        ('kmeans128', 'MinMax'): 0.6507,
        ('random128', 'MinMax'): 0.6166,
        ('highnorm128', 'MinMax'): 0.5829,

        # Asymmetric flavour from 13_quantize_entropy_real.py (original)
        ('easy128', 'Entropy nb=2048'): 0.9402,
        ('fps128', 'Entropy nb=2048'): 0.9359,
        ('mixed128', 'Entropy nb=2048'): 0.9363,
        ('selectq128', 'Entropy nb=2048'): 0.9357,
        ('hard128', 'Entropy nb=2048'): 0.9203,
        ('stratified128', 'Entropy nb=2048'): 0.9023,
        ('kmeans128', 'Entropy nb=2048'): 0.8865,
        ('random128', 'Entropy nb=2048'): 0.8842,
        ('highnorm128', 'Entropy nb=2048'): 0.7756,

        # Asymmetric Percentile from 09_quantize_calibrator.py
        ('easy128', 'Percentile 99.999'): 0.7256,
        ('fps128', 'Percentile 99.999'): 0.6460,
        ('hard128', 'Percentile 99.999'): 0.6202,
        ('random128', 'Percentile 99.999'): 0.9185,
    }

    grid = np.full((len(samples), len(cals)), np.nan)
    for i, s in enumerate(samples):
        for j, c in enumerate(cals):
            v = data.get((s, c))
            if v is not None:
                grid[i, j] = v / REF * 100  # retention %

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=20, vmax=100)
    ax.set_xticks(range(len(cals)))
    ax.set_xticklabels(cals)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(samples)
    ax.set_title('Fig 1. In-house yolo11n 4-class — mAP50 retention (% vs FP32 ONNX)\n'
                 'Cells reveal sampling × calibrator interaction', loc='left')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center',
                        color='gray', fontsize=10)
            else:
                col = 'white' if v < 50 or v > 95 else 'black'
                ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                        color=col, fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('mAP50 retention (%)', rotation=270, labelpad=15)
    fig.tight_layout()
    out = HERE / 'fig01_inhouse_sampling_calibrator_heatmap.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 2 — COCO yolov8n recipe sweep bar chart (R0..R8)
# ============================================================
def fig2_coco_recipe_sweep():
    REF = 0.5206
    rows = [
        ('R0_baseline\n(MinMax asym)', 0.4974, NEUTRAL),
        ('R1_symmetric', 0.4836, BAD),
        ('R2_moving_avg\n+ sym', 0.4984, OK),
        ('R3_no_reduce_range', 0.2585, BAD),
        ('R4_pct_99_9', 0.4705, BAD),
        ('R5_pct_99_999\n+ sym (Tier S)', 0.5025, GOLD),
        ('R6_entropy_nb2048\n+ sym', 0.4928, OK),
        ('R7_dfl_only_exclude', 0.4835, BAD),
        ('R8_mse_custom\n+ sym', 0.4983, OK),
    ]
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows]
    rets = [r[1] / REF * 100 for r in rows]
    cols = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(rows)), rets, color=cols, edgecolor='black',
                  linewidth=0.5)

    for i, (b, r) in enumerate(zip(bars, rets)):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.5,
                f'{r:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')

    ax.axhline(95, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax.axhline(50, color='red', linewidth=1, linestyle='--', alpha=0.4)
    ax.text(len(rows) - 0.5, 95.4, '95% threshold', color='gray', fontsize=8)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('mAP50 retention (% vs FP32 ONNX)')
    ax.set_title('Fig 2. COCO yolov8n + random500 — recipe-knob impact (R0–R8)\n'
                 'Tier S = R5 Percentile 99.999 + symmetric. R3 = catastrophic without '
                 'reduce_range', loc='left')
    ax.set_ylim(40, 100)
    ax.grid(axis='y', alpha=0.3)

    handles = [
        mpatches.Patch(color=GOLD,    label='Tier S winner'),
        mpatches.Patch(color=OK,      label='Strong alternative'),
        mpatches.Patch(color=NEUTRAL, label='Default baseline'),
        mpatches.Patch(color=BAD,     label='Worse than baseline / catastrophic'),
    ]
    ax.legend(handles=handles, loc='lower left', framealpha=0.9)

    fig.tight_layout()
    out = HERE / 'fig02_coco_recipe_sweep.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 3 — yolov8 size scaling
# ============================================================
def fig3_size_scaling():
    sizes = [(3.2, 'yolov8n', 0.5206, 0.5025),
             (11.2, 'yolov8s', 0.6124, 0.6023),
             (25.9, 'yolov8m', 0.6662, 0.6563)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    params = [r[0] for r in sizes]
    fps = [r[2] for r in sizes]
    int8 = [r[3] for r in sizes]
    ret = [r[3] / r[2] * 100 for r in sizes]
    names = [r[1] for r in sizes]

    ax.plot(params, fps, 'o-', color=NEUTRAL, label='FP32 ONNX mAP50',
            linewidth=2, markersize=10)
    ax.plot(params, int8, 's-', color=GOLD,
            label='INT8 (Tier S R5) mAP50', linewidth=2, markersize=10)

    for p, f, i, n in zip(params, fps, int8, names):
        ax.annotate(n, (p, f), textcoords='offset points',
                    xytext=(8, 6), fontsize=9)
        ax.annotate(f'{f:.4f}', (p, f), textcoords='offset points',
                    xytext=(8, -12), fontsize=8, color=NEUTRAL)
        ax.annotate(f'{i:.4f}', (p, i), textcoords='offset points',
                    xytext=(8, -12), fontsize=8, color='#b07c2a')

    ax.set_xlabel('Model parameters (millions)')
    ax.set_ylabel('mAP50 (COCO val2017)')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')

    ax2.plot(params, ret, '^--', color=OK, label='Retention (%)',
             linewidth=1.5, markersize=8, alpha=0.8)
    for p, r in zip(params, ret):
        ax2.annotate(f'{r:.1f}%', (p, r), textcoords='offset points',
                     xytext=(-6, 6), fontsize=8, color=OK,
                     fontweight='bold')
    ax2.set_ylabel('INT8 retention (% vs FP32)', color=OK)
    ax2.tick_params(axis='y', labelcolor=OK)
    ax2.set_ylim(94, 100)

    ax.set_title('Fig 3. yolov8 size scaling — bigger models retain more '
                 'mAP after INT8 PTQ\n'
                 'Tier S R5 recipe; n→s→m: 96.5% → 98.3% → 98.5%',
                 loc='left')
    fig.tight_layout()
    out = HERE / 'fig03_size_scaling.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 4 — Cross-version retention (5 models)
# ============================================================
def fig4_cross_version():
    rows = [
        ('yolov8n', 'v8',  3.2, 0.5206, 0.5025, 'tail-24'),
        ('yolo11n', 'v11', 2.6, 0.5471, 0.5261, 'tail-24'),
        ('yolov10n*', 'v10', 2.8, 0.5313, 0.5083, 'tail-60'),
        ('yolov8s', 'v8',  11.2, 0.6124, 0.6023, 'tail-24'),
        ('yolov8m', 'v8',  25.9, 0.6662, 0.6563, 'tail-24'),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    x = np.arange(len(rows))
    width = 0.32

    fps = [r[3] for r in rows]
    int8 = [r[4] for r in rows]
    ret = [r[4] / r[3] * 100 for r in rows]

    b1 = ax.bar(x - width / 2, fps, width, label='FP32 ONNX',
                color=NEUTRAL, edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x + width / 2, int8, width, label='INT8 (Tier S R5)',
                color=GOLD, edgecolor='black', linewidth=0.4)

    for b, v in zip(b1, fps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f'{v:.4f}', ha='center', fontsize=8)
    for b, v, r in zip(b2, int8, ret):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f'{v:.4f}\n({r:.1f}%)', ha='center', fontsize=8,
                color='#b07c2a', fontweight='bold')

    labels = [f'{r[0]}\n{r[2]}M params' for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('mAP50 (COCO val2017)')
    ax.set_title('Fig 4. Cross-version validation — Tier S R5 recipe consistent across '
                 '4 architectures × 3 sizes\n'
                 'v10n* requires extended head_exclude (tail-60 + extra ops)',
                 loc='left')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.78)
    fig.tight_layout()
    out = HERE / 'fig04_cross_version.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 5 — Two-regime comparison
# ============================================================
def fig5_two_regime():
    """Spread across sampling strategies, COCO vs in-house, by calibrator."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={'width_ratios': [1, 1.2]})

    # Left: spread bar chart
    regimes = ['COCO\n(rich)', 'in-house\n(narrow)']
    spreads = {
        'MinMax':              [None, 67.3],
        'Entropy nb=2048':     [None, 17.2],
        'Percentile 99.999':   [None, 31.2],
        'COCO (any cal/sample)': [2.0, None],
    }
    # Combine into one grouped bar chart
    width = 0.22
    x = np.arange(len(regimes))

    spread_data = [
        ('MinMax', 67.3, BAD),
        ('Entropy nb=2048', 17.2, OK),
        ('Percentile 99.999', 31.2, WARN),
    ]
    for i, (name, val, col) in enumerate(spread_data):
        ax1.bar(i, val, color=col, edgecolor='black', linewidth=0.5,
                label=name)
        ax1.text(i, val + 1, f'{val:.1f}pp', ha='center', fontsize=10,
                 fontweight='bold')

    ax1.bar(len(spread_data), 2.0, color=NEUTRAL,
            edgecolor='black', linewidth=0.5, label='COCO any')
    ax1.text(len(spread_data), 2.0 + 1, '~2pp', ha='center', fontsize=10,
             fontweight='bold')

    ax1.set_xticks(range(len(spread_data) + 1))
    ax1.set_xticklabels(['MinMax', 'Entropy\nnb=2048', 'Percentile\n99.999',
                         'any cal\n(COCO)'], fontsize=9)
    ax1.set_ylabel('Spread of mAP50 retention across sampling strategies (pp)')
    ax1.set_title('Fig 5a. Sampling-strategy spread\n'
                  'Narrow data: spread depends heavily on calibrator',
                  loc='left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 80)

    # Right: COCO vs in-house under (random vs easy) × (MinMax vs Percentile)
    cells = [
        ('COCO\nMinMax', 'random128',  95.5, NEUTRAL),
        ('COCO\nPercentile', 'random128',  96.5, GOLD),
        ('Narrow\nMinMax', 'random128',  64.4, BAD),
        ('Narrow\nMinMax', 'easy128',    96.8, OK),
        ('Narrow\nPercentile', 'random128',  96.0, GOLD),
        ('Narrow\nPercentile', 'easy128',    75.8, BAD),
    ]
    bx = np.arange(len(cells))
    vals = [c[2] for c in cells]
    cols = [c[3] for c in cells]
    bars = ax2.bar(bx, vals, color=cols, edgecolor='black', linewidth=0.5)
    for b, c, v in zip(bars, cells, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.5,
                 f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax2.text(b.get_x() + b.get_width() / 2, 5, c[1],
                 ha='center', fontsize=8, color='white',
                 fontweight='bold', rotation=0)
    ax2.set_xticks(bx)
    ax2.set_xticklabels([c[0] for c in cells], fontsize=8.5)
    ax2.set_ylabel('mAP50 retention (%)')
    ax2.set_title('Fig 5b. Sampling–calibrator interaction differs by regime\n'
                  'In narrow data, "easy" sampling reverses winner under Percentile',
                  loc='left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(95, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax2.set_ylim(0, 110)

    fig.tight_layout()
    out = HERE / 'fig05_two_regime.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 6 — ONNX RT default Entropy ≡ MinMax (illustrate fix)
# ============================================================
def fig6_entropy_bug():
    """Illustrate that with default num_bins=128, Entropy degenerates."""
    cases = [
        ('easy128',   0.9265, 0.9265, 0.9402),
        ('fps128',    0.9000, 0.9000, 0.9359),
        ('random128', 0.6166, 0.6166, 0.8842),
        ('hard128',   0.2827, 0.2827, 0.9203),
    ]
    REF = 0.9568
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(cases))
    width = 0.27
    mm = [c[1] / REF * 100 for c in cases]
    ed = [c[2] / REF * 100 for c in cases]
    er = [c[3] / REF * 100 for c in cases]

    b1 = ax.bar(x - width, mm, width, label='MinMax',
                color=NEUTRAL, edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x,         ed, width, label='Entropy default (nb=128)',
                color='#a3a3a3', edgecolor='black', linewidth=0.4,
                hatch='///')
    b3 = ax.bar(x + width, er, width, label='Entropy effective (nb=2048)',
                color=OK, edgecolor='black', linewidth=0.4)

    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0,
                    f'{v:.1f}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel('mAP50 retention (% vs FP32 ONNX)')
    ax.set_title('Fig 6. Effective Entropy (num_bins=2048) vs the broken default\n'
                 'Default Entropy in ORT 1.23 is byte-identical to MinMax. '
                 'Bypass to nb=2048 rescues catastrophic cases.',
                 loc='left')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)
    fig.tight_layout()
    out = HERE / 'fig06_entropy_default_vs_effective.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 7 — Seed stability
# ============================================================
def fig7_seed_stability():
    seeds = [(0, 0.5025), (1, 0.5021), (2, 0.5033)]
    REF = 0.5206
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [s[0] for s in seeds]
    y = [s[1] / REF * 100 for s in seeds]
    ax.plot(x, y, 'o-', color=GOLD, markersize=12, linewidth=2)
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.2f}%', (xi, yi),
                    textcoords='offset points', xytext=(0, 12),
                    fontsize=10, ha='center', fontweight='bold')
    mean = np.mean(y)
    ax.axhline(mean, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax.axhspan(mean - 0.15, mean + 0.15, color=GOLD, alpha=0.18,
               label='±0.15 pp band')
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel('Random seed (calibration sampling)')
    ax.set_ylabel('mAP50 retention (%)')
    ax.set_title('Fig 7. Tier S seed stability on yolov8n + COCO + random128\n'
                 'Spread ±0.15 pp across 3 seeds — recipe is not a seed artifact',
                 loc='left')
    ax.set_ylim(96.0, 97.0)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = HERE / 'fig07_seed_stability.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


# ============================================================
# Figure 8 — Decision flowchart
# ============================================================
def fig8_decision_flow():
    """Hand-drawn-ish decision flowchart."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def box(x, y, w, h, text, fc='white', ec='black'):
        rect = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                       boxstyle='round,pad=0.05',
                                       linewidth=1.2, edgecolor=ec,
                                       facecolor=fc)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9.5)

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', linewidth=1.2))
        if label:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2,
                    label, fontsize=8.5, color='#444')

    # Root
    box(5, 7.3, 4, 0.8, 'YOLO INT8 PTQ on ORT', fc='#eef')

    # Layer 1 — YOLO version
    box(2.0, 5.7, 3.0, 0.8, 'v8 / v9 / v11', fc='#dfe7f3')
    box(5.0, 5.7, 2.0, 0.8, 'v10 / NMS-free', fc='#fff2d9')
    box(7.7, 5.7, 1.6, 0.8, 'v6', fc='#fde2dc')
    box(9.3, 5.7, 1.4, 0.8, 'v5 + TRT', fc='#fde2dc')
    arrow(4, 7.0, 2.5, 6.1, 'standard')
    arrow(5, 7.0, 5, 6.1, 'NMS-free')
    arrow(5.7, 7.0, 7.5, 6.1, 'v6 RepVGG')
    arrow(6, 7.0, 9.2, 6.1, 'v5 + TRT')

    # Layer 2 — config
    box(2.0, 4.3, 3.0, 0.8, 'Tier S as-is', fc='#d8efd2')
    box(5.0, 4.3, 2.0, 0.9,
        'Tier S\n+ skip_symbolic_shape\n+ tail-60 exclude',
        fc='#fff2d9')
    box(7.7, 4.3, 1.6, 0.8,
        'Try CalibPercentile\n=99.9 first', fc='#fde2dc')
    box(9.3, 4.3, 1.4, 0.8, 'SiLU plugin', fc='#fde2dc')
    arrow(2, 5.3, 2, 4.7)
    arrow(5, 5.3, 5, 4.7)
    arrow(7.7, 5.3, 7.7, 4.7)
    arrow(9.3, 5.3, 9.3, 4.7)

    # Layer 3 — narrow vs rich
    box(3, 2.7, 4, 0.8, 'Narrow / finetuned dataset?', fc='#eef')
    arrow(2, 3.9, 2.5, 3.1)
    arrow(5, 3.85, 4, 3.1)

    box(1.2, 1.3, 2.2, 1.2,
        'Tier S works\n(96% on 4-class)', fc='#d8efd2')
    box(4.8, 1.3, 3.0, 1.2,
        'If <90% retention,\ntry effective Entropy\nnb=2048 (Tier A)',
        fc='#fff2d9')
    arrow(2.5, 2.3, 1.6, 1.95, 'yes')
    arrow(3.5, 2.3, 4.6, 1.95, 'still <90%?')

    ax.set_title('Fig 8. Tier S deployment flow — pick the right special-case '
                 'handling and you\'re done',
                 fontsize=11, loc='left', pad=10)

    out = HERE / 'fig08_decision_flow.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


def fig9_apples_to_apples():
    """Apples-to-apples comparison: same recipes (symmetric Tier S
    flavour) on both regimes. Settles 'random+Pct vs easy+Entropy'."""
    REF_COCO = 0.5206
    REF_NARROW = 0.9568
    rows = [
        ('random + Pct 99.999\n(Tier S)',  0.5025, 0.9245, GOLD),
        ('easy + Pct 99.999',              0.5020, 0.6485, BAD),
        ('fps + Pct 99.999',               0.5019, 0.7660, BAD),
        ('hard + Pct 99.999',              0.5021, 0.8771, WARN),
        ('random + Entropy nb=2048',       0.4928, 0.9015, NEUTRAL),
        ('easy + Entropy nb=2048',         0.4915, 0.9280, OK),
        ('fps + Entropy nb=2048',          0.4974, 0.9407, OK),
        ('hard + Entropy nb=2048',         0.4906, 0.9113, OK),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(rows))
    width = 0.36
    coco_ret = [r[1] / REF_COCO * 100 for r in rows]
    narrow_ret = [r[2] / REF_NARROW * 100 for r in rows]

    b1 = ax.bar(x - width / 2, coco_ret, width,
                label='COCO yolov8n (rich data)', color=NEUTRAL,
                edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x + width / 2, narrow_ret, width,
                label='in-house yolo11n 4-class (narrow data)',
                color=GOLD, edgecolor='black', linewidth=0.4)

    for b, v in zip(b1, coco_ret):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6,
                f'{v:.1f}%', ha='center', fontsize=8.5)
    for b, v in zip(b2, narrow_ret):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6,
                f'{v:.1f}%', ha='center', fontsize=8.5,
                color='#7a5a14', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5,
                        rotation=15, ha='right')
    ax.set_ylabel('mAP50 retention (% vs FP32 ONNX of the same model)')
    ax.set_title('Fig 9. Apples-to-apples: random+Percentile vs '
                 'easy+Entropy across both regimes\n'
                 '(All recipes use symmetric activations + per-channel + '
                 'reduce_range=True + tail-60 exclude)',
                 loc='left')
    ax.axhline(95, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    ax.set_ylim(60, 102)
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = HERE / 'fig09_apples_to_apples.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


def fig10_coco_full_matrix():
    """The COCO 4-sampling × 3-calibrator matrix as a heatmap, mirror of
    Fig 1 for the rich-data regime."""
    REF = 0.5206
    samples = ['random128', 'easy128', 'fps128', 'hard128']
    cals = ['MinMax (R0 asym)', 'Percentile 99.999\n(R5 sym)',
            'Entropy nb=2048\n(R6 sym)']
    data = {
        ('random128', 'MinMax (R0 asym)'): 0.4974,
        ('easy128',   'MinMax (R0 asym)'): 0.4968,
        ('fps128',    'MinMax (R0 asym)'): 0.4984,
        ('hard128',   'MinMax (R0 asym)'): 0.4786,
        ('random128', 'Percentile 99.999\n(R5 sym)'): 0.5025,
        ('easy128',   'Percentile 99.999\n(R5 sym)'): 0.5020,
        ('fps128',    'Percentile 99.999\n(R5 sym)'): 0.5019,
        ('hard128',   'Percentile 99.999\n(R5 sym)'): 0.5021,
        ('random128', 'Entropy nb=2048\n(R6 sym)'):   0.4928,
        ('easy128',   'Entropy nb=2048\n(R6 sym)'):   0.4915,
        ('fps128',    'Entropy nb=2048\n(R6 sym)'):   0.4974,
        ('hard128',   'Entropy nb=2048\n(R6 sym)'):   0.4906,
    }
    grid = np.full((len(samples), len(cals)), np.nan)
    for i, s in enumerate(samples):
        for j, c in enumerate(cals):
            grid[i, j] = data[(s, c)] / REF * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=85, vmax=100)
    ax.set_xticks(range(len(cals)))
    ax.set_xticklabels(cals, fontsize=9)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(samples)
    ax.set_title('Fig 10. COCO yolov8n — sampling × calibrator matrix\n'
                 'Spread is much smaller than narrow regime (max 4.6 pp '
                 'vs 67 pp); R5 column is essentially flat at 96.4-96.5%',
                 loc='left')

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            col = 'white' if v < 92 else 'black'
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center',
                    color=col, fontsize=9.5, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('mAP50 retention (%)', rotation=270, labelpad=15)
    fig.tight_layout()
    out = HERE / 'fig10_coco_full_matrix.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'-> {out.name}')


def main():
    fig1_inhouse_heatmap()
    fig2_coco_recipe_sweep()
    fig3_size_scaling()
    fig4_cross_version()
    fig5_two_regime()
    fig6_entropy_bug()
    fig7_seed_stability()
    fig8_decision_flow()
    fig9_apples_to_apples()
    fig10_coco_full_matrix()
    print('all figures generated.')


if __name__ == '__main__':
    main()
