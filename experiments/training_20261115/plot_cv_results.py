"""Show all same-fold internal CV results without independence-based error bars."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def plot():
    result = json.loads((HERE / 'cv_summary.json').read_text(encoding='utf-8'))
    if result['status'] != 'completed' or len(result['training_records']) != 27 or len(result['folds']) != 9:
        raise ValueError('All 27 fits and all nine fold ensembles required')
    names = ['b0_standard', 'b0_regularized', 'v2s_regularized']
    labels = ['B0 standard', 'B0 regularized', 'V2-S regularized']
    series = {name: result['recipes'][name] for name in names}
    control_path = HERE / 'cv_cpu_verification.json'
    if control_path.exists():
        control = json.loads(control_path.read_text(encoding='utf-8'))['geometry_control']
        names.append('geometry'); labels.append('Image dimensions'); series['geometry'] = control
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10.5, 4.7), constrained_layout=True)
    for fold, color in enumerate(['#20639b', '#dd8452', '#2a8c70']):
        ax.plot(np.arange(len(names)), [100*series[name]['fold_accuracies'][fold] for name in names],
                marker='o', color=color, label=f'Fold {fold+1}', lw=1.2, alpha=.8)
    ax.scatter(np.arange(len(names)), [100*series[name]['pooled_metrics']['accuracy'] for name in names],
               color='#243746', marker='D', s=75, label='Pooled out-of-fold', zorder=6)
    ax.set(xticks=np.arange(len(names)), xticklabels=labels, ylabel='Outer-fold accuracy (%)', ylim=(0, 101),
           title='Grouped internal assessment: 3,815 images / 2,153 filename proxies')
    ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, ncol=4, loc='lower left')
    out = ROOT / 'output/figures/november_training'; out.mkdir(exist_ok=True, parents=True)
    for suffix in ['png', 'pdf']: fig.savefig(out / ('grouped_cv.' + suffix), dpi=180)
    plt.close(fig)
    print((out / 'grouped_cv.pdf').relative_to(ROOT).as_posix())


if __name__ == '__main__': plot()
