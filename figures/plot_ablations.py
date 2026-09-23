"""Plot paper ablations as five-metric horizontal rows from ablation_data.csv."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = [('MSE', 'mse'), ('MAE', 'mae'), ('FD', 'fd'), ('WInD', 'wind'), ('LMD', 'lmd')]
# A method keeps its color across all ablation groups.
SFT_COLOR, GRPO_COLOR = '#4B6FA9', '#238B87'
VARIANT_COLORS = {
    'sft': SFT_COLOR, 'grpo': GRPO_COLOR,
    'audio_only': '#CE842C', 'no_guidance': '#8964AA',
    'no_reweight': '#C5534B', 'no_slmd': '#B78324',
    'sft_no_smooth': SFT_COLOR, 'grpo_no_smooth': GRPO_COLOR,
}
VARIANT_HATCHES = {'sft_no_smooth': '///', 'grpo_no_smooth': '///'}
GROUPS = {
    'conditioning': {
        'title': 'Speech Conditioning Ablation',
        'variants': ['audio_only', 'no_guidance', 'sft'],
        'labels': ['Audio Only', 'w/o Guidance', 'SFT'],
        'ticks': ['Audio\nOnly', 'w/o\nGuidance', 'SFT']},
    'reweighting': {
        'title': 'SFT Ablation: Coefficient Reweighting',
        'variants': ['no_reweight', 'sft'],
        'labels': ['w/o Reweighting', 'SFT'],
        'ticks': ['w/o\nReweighting', 'SFT']},
    'motion': {
        'title': 'GRPO Ablation: Motion Refinement',
        'variants': ['sft', 'no_slmd', 'grpo'],
        'labels': ['SFT', 'w/o SLMD', 'GRPO'],
        'ticks': ['SFT', 'w/o\nSLMD', 'GRPO']},
    'smoothing': {
        'title': 'Post-Processing Ablation',
        'variants': ['sft_no_smooth', 'sft', 'grpo_no_smooth', 'grpo'],
        'labels': ['SFT (no smooth)', 'SFT', 'GRPO (no smooth)', 'GRPO'],
        'ticks': ['SFT\n(off)', 'SFT\n(on)', 'GRPO\n(off)', 'GRPO\n(on)']},
}

def load_data():
    with (HERE / 'ablation_data.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {r['variant']: r for r in rows}

def plot_group(name, group, data):
    count = len(group['variants'])
    colors = [VARIANT_COLORS[v] for v in group['variants']]
    hatches = [VARIANT_HATCHES.get(v, '') for v in group['variants']]
    fig, axes = plt.subplots(1, 5, figsize=(11.8, 3.05))
    fig.subplots_adjust(left=0.047, right=0.99, bottom=0.265, top=0.745, wspace=0.44)
    for j, (ax, (title, metric)) in enumerate(zip(axes, METRICS)):
        values = np.array([float(data[k][metric]) for k in group['variants']])
        sds = np.array([float(data[k][metric + '_sd']) if data[k].get(metric + '_sd') else np.nan for k in group['variants']])
        upper = values + np.nan_to_num(sds)
        raw_limit = upper.max() * 1.23
        magnitude = 10 ** np.floor(np.log10(raw_limit))
        ymax = np.ceil(raw_limit / magnitude * 5) / 5 * magnitude
        x = np.arange(count)
        highlight = '#F0F4FA' if group['variants'][-1] == 'sft' else '#EFF6F5'
        ax.axvspan(count - 1.44, count - 0.56, color=highlight, zorder=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#E6E9EE', linewidth=0.7)
        bars = ax.bar(x, values, width=0.62, color=colors, zorder=3)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor('white')
                bar.set_linewidth(0.0)
        for i in range(count):
            if np.isfinite(sds[i]):
                # Same convention as the referenced conditioning figure: upper SD only.
                ax.vlines(i, values[i], upper[i], color='#46515D', linewidth=0.85, zorder=4)
                ax.hlines(upper[i], i - 0.065, i + 0.065, color='#46515D', linewidth=0.85, zorder=4)
            value_label = f'{values[i]:#.4g}'  # Four significant digits, retaining trailing zeros.
            label_offset = 0.025 + (0.085 * (i % 2) if count == 4 else 0.0)
            ax.text(i, upper[i] + ymax * label_offset, value_label,
                    ha='center', va='bottom', fontsize=6.4 if count == 4 else 7.3,
                    color=colors[i], fontweight='bold' if i == count - 1 else 'normal')
        ax.set_xlim(-0.55, count - 0.45)
        ax.set_ylim(0, ymax)
        ax.set_xticks(x, group['ticks'], fontsize=7.1 if count == 4 else 7.8)
        ax.get_xticklabels()[-1].set_fontweight('bold')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}' if j == 0 else '{x:.2f}' if j == 1 else '{x:.0f}'))
        ax.tick_params(axis='x', length=0, pad=5)
        ax.tick_params(axis='y', labelsize=7.5, length=3, width=0.6, color='#7B818A')
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#9299A3')
            ax.spines[side].set_linewidth(0.7)
        ax.set_xlabel(f'({chr(97+j)}) {title} ↓', fontsize=9.5, labelpad=7)
    fig.suptitle(group['title'], x=0.52, y=0.99, fontsize=10.5)
    handles = [Patch(facecolor=c, edgecolor='white' if h else c, hatch=h, label=l)
               for c,h,l in zip(colors, hatches, group['labels'])]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.915),
               ncol=count, frameon=False, fontsize=9.4, handlelength=1.2,
               handletextpad=0.55, columnspacing=1.8)
    note = 'Lower is better   |   Upper error bars: +1 sample SD (FD, WInD, LMD)'
    if name == 'smoothing':
        note += '   |   on/off: smoothing'
    fig.text(0.52, 0.020, note,
             ha='center', va='bottom', fontsize=7.5, color='#606773')
    # Check that exported labels remain within the canvas and do not overlap adjacent values.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        bounds = [t.get_window_extent(renderer) for t in ax.texts]
        for box in bounds:
            assert box.x0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, name
        assert not any(a.overlaps(b) for a,b in zip(bounds, bounds[1:])), f'Overlapping values: {name}'
    stem = HERE / f'ablation_{name}_wide'
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.' + ext), dpi=400, facecolor='white')
    plt.close(fig)
    print(stem.name)

def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'text.color': '#2D333B', 'axes.labelcolor': '#2D333B',
        'xtick.color': '#3C4149', 'ytick.color': '#3C4149',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
    data = load_data()
    for name, group in GROUPS.items():
        plot_group(name, group, data)

if __name__ == '__main__':
    main()
