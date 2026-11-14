"""Publication figures drawn only from observed, completed training records."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def plot():
    summary = json.loads((HERE / 'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed' or len(summary['trials']) != 9:
        raise ValueError('Complete nine-trial result required')
    history = pd.DataFrame(json.loads((HERE / 'epoch_history.json').read_text(encoding='utf-8')))
    out = ROOT / 'output/figures/november_training'
    out.mkdir(parents=True, exist_ok=True)
    colors = {42: '#20639b', 123: '#dd8452', 2026: '#2a8c70'}
    names = {'b0_standard': 'EfficientNet-B0 / standard', 'b0_regularized': 'EfficientNet-B0 / regularized',
             'v2s_regularized': 'EfficientNet-V2-S / regularized'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9), sharex=True, constrained_layout=True)
    for row, recipe in enumerate(names):
        for seed in [42, 123, 2026]:
            trial = f'{recipe}_seed_{seed}'
            rec = next(r for r in summary['trials'] if r['trial'] == trial)
            h = history[(history.trial == trial) & history.complete_epoch].sort_values('epoch')
            if len(h) != rec['epochs_completed']:
                raise ValueError('Incomplete epoch capture: ' + trial)
            best = h[h.epoch == rec['best_epoch']]
            if len(best) != 1:
                raise ValueError('Selected epoch absent from history')
            axes[row, 0].plot(h.epoch, 100 * h.early_stop_accuracy, color=colors[seed], label=f'Seed {seed}', lw=1.5)
            axes[row, 0].scatter(best.epoch, 100 * best.early_stop_accuracy, color=colors[seed], edgecolors='white', s=44, zorder=5)
            axes[row, 1].plot(h.epoch, h.train_loss, color=colors[seed], label=f'Seed {seed}', lw=1.5)
        axes[row, 0].set_title(names[recipe], loc='left', fontweight='bold')
        axes[row, 0].set_ylabel('Internal holdout accuracy (%)')
        axes[row, 1].set_ylabel('Training objective value')
        for ax in axes[row]:
            ax.grid(alpha=.2); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # Set shared limits after every recipe is drawn; setting a bound in the
    # first row disables autoscaling and can clip later, longer trials.
    for ax in axes.flat: ax.set_xlim(1, int(history.loc[history.complete_epoch, 'epoch'].max()) + .5)
    axes[0, 0].legend(loc='lower right', frameon=False)
    axes[-1, 0].set_xlabel('Complete training epoch')
    axes[-1, 1].set_xlabel('Complete training epoch')
    fig.suptitle('Nine fresh training runs\nDots mark saved checkpoints; internal holdout n = 781', fontsize=13)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / ('learning_curves.' + suffix), dpi=180)
    plt.close(fig)

    comparison = summary['uncertainty']['paired_comparison']
    metric_names = [('accuracy', 'Accuracy'), ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity')]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), constrained_layout=True)
    y = np.arange(3)
    axes[0].barh(y + .17, [100 * comparison['baseline'][k] for k, _ in metric_names], height=.32, color='#61788a', label='Original 320 ensemble')
    axes[0].barh(y - .17, [100 * comparison['followup'][k] for k, _ in metric_names], height=.32, color='#207f79', label='November candidate')
    axes[0].set(yticks=y, yticklabels=[n for _, n in metric_names], xlim=(0, 105), xlabel='Observed source-test percentage')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.01), frameon=False, fontsize=10)
    for i, (key, _) in enumerate(metric_names):
        delta = 100 * comparison['difference'][key]
        low, high = np.array(comparison['paired_cluster_bootstrap_95ci'][key]) * 100
        # Plot percentile endpoints directly; they need not straddle the point estimate.
        axes[1].plot([low, high], [i, i], color='#207f79', lw=2)
        axes[1].scatter([delta], [i], color='#207f79', zorder=4)
    axes[1].axvline(0, color='#7a7a7a', ls='--', lw=1)
    axes[1].set(yticks=y, yticklabels=[n for _, n in metric_names], xlabel='Candidate minus original (percentage points)',
                title='Paired 95% group-bootstrap intervals')
    for ax in axes: ax.grid(axis='x', alpha=.2)
    fig.suptitle('Exploratory source test: 618 images / 426 filename proxy groups', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / ('test_comparison.' + suffix), dpi=180)
    plt.close(fig)
    verification_path = HERE / 'cpu_verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        diagnostics = verification['descriptive_score_diagnostics']
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        palette = {'calibration': '#20639b', 'test': '#dd8452'}
        axes[0].plot([0, 1], [0, 1], ls='--', color='gray', lw=1)
        for split, color in palette.items():
            bins = [b for b in diagnostics[split]['fixed_equal_width_bins'] if b['count']]
            axes[0].scatter([b['mean_score'] for b in bins], [b['observed_fraction'] for b in bins],
                            s=[20 + 4*np.sqrt(b['count']) for b in bins], color=color, alpha=.8,
                            label=f"{split.title()} (n={diagnostics[split]['images']})")
        axes[0].set(xlabel='Mean raw ensemble score in bin', ylabel='Observed pneumonia-label fraction',
                    title='Ten fixed equal-width bins', xlim=(-.03, 1.03), ylim=(-.03, 1.03))
        axes[0].legend(frameon=False, fontsize=10, loc='lower right')
        positions, labels = [], []
        for i, (split, label) in enumerate([('calibration', 'normal'), ('test', 'normal'),
                                           ('calibration', 'pneumonia'), ('test', 'pneumonia')]):
            q = diagnostics[split]['class_scores'][label]['quantiles_5_25_50_75_95']
            axes[1].plot([q[0], q[4]], [i, i], color=palette[split], lw=1.5)
            axes[1].plot([q[1], q[3]], [i, i], color=palette[split], lw=6, solid_capstyle='butt')
            axes[1].scatter(q[2], i, color='white', edgecolor=palette[split], s=35, zorder=4)
            positions.append(i); labels.append(f'{split.title()} / {label}')
        axes[1].set(yticks=positions, yticklabels=labels, xlabel='Raw ensemble score',
                    title='Class-conditional score distributions', xlim=(-.03, 1.03))
        axes[1].invert_yaxis()
        for ax in axes: ax.grid(alpha=.2)
        fig.suptitle('Descriptive score transfer; fixed trained model and cutoff', fontsize=12)
        for suffix in ['png', 'pdf']: fig.savefig(out / ('score_transfer.' + suffix), dpi=180)
        plt.close(fig)
    return out


if __name__ == '__main__': print(plot().relative_to(ROOT).as_posix())
