"""Alternative displays: reference-indexed bars and annotated metric heatmaps.

Run with matplotlib and numpy; source measurements remain in ../ablation_data.csv.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ['mse', 'mae', 'fd', 'wind', 'lmd']
TICKS = ['MSE\n×10⁴', 'MAE\n×10³', 'FD', 'WInD', 'LMD']
MULTIPLIERS = np.array([1e4, 1e3, 1, 1, 1])
COLORS = dict(audio_only='#CE842C', no_guidance='#8964AA', sft='#4B6FA9',
              no_reweight='#C5534B', no_slmd='#B78324', grpo='#238B87')
GROUPS = {
    'conditioning': ('Linguistic Conditioning',
                     ['audio_only', 'no_guidance', 'sft'],
                     ['Audio Only', 'Audio + Text + Phonemes',
                      'Audio + Text + Phonemes + Guidance']),
    'reweighting': ('Coefficient Reweighting', ['no_reweight', 'sft'],
                   ['SFT without Reweighting', 'SFT with Reweighting']),
    'motion': ('Motion Refinement', ['sft', 'no_slmd', 'grpo'],
               ['SFT', 'GRPO without SLMD', 'GRPO with SLMD']),
}


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A label outside the exported canvas would hide a reported result.
    for ax in fig.axes:
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, name
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(HERE / f'{name}.{ext}', dpi=350, facecolor='white')
    plt.close(fig)


def bars(name, data):
    title, variants, labels = GROUPS[name]
    raw = np.array([[float(data[v][m]) for m in METRICS] for v in variants])
    relative = raw / raw[0] * 100
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.23, top=0.69)
    x = np.arange(5)
    spacing = 0.28
    ax.axhline(100, color='#68717D', linestyle=(0, (4, 3)), linewidth=0.85, zorder=2)
    for i, (variant, label) in enumerate(zip(variants, labels)):
        pos = x + (i - (len(variants) - 1) / 2) * spacing
        color = COLORS[variant]
        ax.bar(pos, relative[i], width=0.20, bottom=0, color=color, zorder=3,
               label=label + (' (reference = 100)' if i == 0 else ''))
        for xx, height, value in zip(pos, relative[i], raw[i] * MULTIPLIERS):
            ax.annotate(f'{value:.1f}', (xx, height), xytext=(0, 4),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=8, color=color, zorder=5)
    ax.set_ylim(0, 145)
    ax.set_xlim(-0.6, 4.6)
    ax.set_yticks([0, 25, 50, 75, 100, 125])
    ax.set_ylabel('Error index (reference = 100) ↓', fontsize=10)
    ax.set_xticks(x, TICKS, fontsize=10)
    ax.tick_params(axis='x', length=0, pad=7)
    ax.tick_params(axis='y', labelsize=9, length=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='#E8EBEF', linewidth=0.7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#AAB1BA')
        ax.spines[side].set_linewidth(0.7)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.03), frameon=False,
              fontsize=9, handlelength=1.4, labelspacing=0.4, borderaxespad=0)
    fig.suptitle(title, x=0.54, y=0.98, fontsize=11)
    fig.text(0.54, 0.070, 'Bar height: error / reference × 100. Labels: measured values in the units shown.',
             ha='center', fontsize=8, color='#4C5561')
    fig.text(0.54, 0.024, 'Linear axis; lower is better. FD, WInD and LMD are script means.',
             ha='center', fontsize=8, color='#4C5561')
    save(fig, f'{name}_reference100')


def heatmap(data):
    name = 'conditioning'
    title, variants, _ = GROUPS[name]
    raw = np.array([[float(data[v][m]) for m in METRICS] for v in variants])
    score = (raw.max(axis=0) - raw) / np.ptp(raw, axis=0)
    cmap = LinearSegmentedColormap.from_list('error_quality', ['#F3F5F8', '#AFC7E0', '#2F5687'])
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    fig.subplots_adjust(left=0.37, right=0.98, top=0.74, bottom=0.27)
    ax.imshow(score, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(3):
        for j in range(5):
            ax.text(j, i, f'{raw[i,j] * MULTIPLIERS[j]:.1f}', ha='center', va='center',
                    fontsize=11, color='white' if score[i,j] > 0.68 else '#273747')
    ax.set_xticks(range(5), TICKS, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(3), ['Audio Only', 'Audio + Text + Phonemes',
                            'Audio + Text + Phonemes\n+ Guidance'], fontsize=9)
    for label, variant in zip(ax.get_yticklabels(), variants):
        label.set_color(COLORS[variant])
    ax.tick_params(axis='both', length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title + ' · Annotated Heatmap', y=0.99, fontsize=11)
    fig.text(0.52, 0.16, 'Numbers: measured values in the units shown; lower is better.', ha='center', fontsize=8)
    fig.text(0.52, 0.08, 'Color: per-metric min–max scaling within this group; darker = lower error.',
             ha='center', fontsize=8, color='#4C5561')
    save(fig, 'conditioning_annotated_heatmap')


def combined_bars(data):
    """Three ablation panels in one horizontal figure with shared explanatory text."""
    # At 5.5-inch manuscript width, 24--27 pt here becomes about 7--8 pt.
    # Keep measured-value labels unchanged; enlarge the explanatory typography.
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 7.6), sharey=True)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.30, top=0.61, wspace=0.13)
    for panel, (ax, (name, (title, variants, labels))) in enumerate(zip(axes, GROUPS.items())):
        raw = np.array([[float(data[v][m]) for m in METRICS] for v in variants])
        relative = raw / raw[0] * 100
        x = np.arange(5)
        spacing = 0.33
        ax.axhline(100, color='#68717D', linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        for i, (variant, label) in enumerate(zip(variants, labels)):
            if variant == 'sft' and name == 'conditioning':
                label = 'Audio + Text + Phonemes\n+ Guidance'
            pos = x + (i - (len(variants) - 1) / 2) * spacing
            color = COLORS[variant]
            ax.bar(pos, relative[i], width=0.21, bottom=0, color=color, zorder=3,
                   label=label + (' (ref.)' if i == 0 else ''))
            for xx, height, value in zip(pos, relative[i], raw[i] * MULTIPLIERS):
                ax.annotate(f'{value:.1f}', (xx, height), xytext=(0, 5),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=10.5, color=color, zorder=5)
        ax.set_ylim(0, 145)
        ax.set_xlim(-0.6, 4.6)
        ax.set_yticks([0, 25, 50, 75, 100, 125])
        ax.set_xticks(x, TICKS, fontsize=23)
        ax.tick_params(axis='x', length=0, pad=10)
        ax.tick_params(axis='y', labelsize=12, length=3)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#E8EBEF', linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#AAB1BA')
            ax.spines[side].set_linewidth(0.8)
        ax.legend(loc='upper left', bbox_to_anchor=(ax.get_position().x0, 0.87),
                  bbox_transform=fig.transFigure, frameon=False,
                  fontsize=24, handlelength=1.0, labelspacing=0.4, borderaxespad=0)
        center = (ax.get_position().x0 + ax.get_position().x1) / 2
        fig.text(center, 0.98, f'({chr(97 + panel)}) {title}',
                 ha='center', va='top', fontsize=27)
    axes[0].set_ylabel('Error index (reference = 100) ↓', fontsize=23)
    fig.text(0.52, 0.10,
             'Bar height = error / reference × 100; dashed line = reference. Lower is better.',
             ha='center', fontsize=27, color='#4C5561')
    fig.text(0.52, 0.035,
             'Labels show measured values: MSE ×10⁴, MAE ×10³; FD, WInD and LMD are script means.',
             ha='center', fontsize=27, color='#4C5561')
    # At this size neighboring labels must remain distinct after exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        boxes = [label.get_window_extent(renderer) for label in ax.texts]
        assert not any(a.overlaps(b) for i,a in enumerate(boxes) for b in boxes[i+1:]), 'Overlapping bar labels'
    save(fig, 'ablations_reference100_combined')


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'})
    with (HERE.parent / 'ablation_data.csv').open(newline='') as f:
        data = {row['variant']: row for row in csv.DictReader(f)}
    for name in GROUPS:
        bars(name, data)
    heatmap(data)
    combined_bars(data)


if __name__ == '__main__':
    main()
