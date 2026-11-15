"""Publication diagram of the frozen primary and secondary analysis paths."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/figures/november_training'


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    def box(x, y, w, h, title, detail, color='#eaf1f4'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.08',
                     linewidth=.9, edgecolor='#7b96a1', facecolor=color))
        ax.text(x+w/2, y+h-.23, title, ha='center', va='top', weight='bold', color='#173b4b', fontsize=10)
        ax.text(x+w/2, y+h-.62, detail, ha='center', va='top', fontsize=9.2, linespacing=1.35)
    def arrow(a, b, label=None):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color='#526e79', lw=1.2))
        if label: ax.text((a[0]+b[0])/2+.1, (a[1]+b[1])/2, label, fontsize=8, va='center')
    ax.text(.2, 9.8, 'Two fixed analysis paths', fontsize=17, weight='bold', color='#173b4b')
    ax.text(.2, 9.42, '5,371 retained images; filename groups are unverified patient proxies', fontsize=10, color='#526e79')
    box(.2, 7.8, 4.1, 1.25, 'Original training cohort', '3,815 images / 2,153 proxy groups')
    box(.2, 5.65, 4.1, 1.55, 'Primary training: nine fresh models', 'Fit: 3,034 images / 1,721 groups\nCheckpoint holdout: 781 / 432\nThree recipes × three seeds')
    arrow((2.25, 7.8), (2.25, 7.2))
    box(.2, 3.55, 4.1, 1.55, 'Select one fixed ensemble', 'Selection: 456 images / 270 groups\nFive-fold cutoff cross-fitting\nFour candidates; no test-based choice')
    arrow((2.25, 5.65), (2.25, 5.1))
    box(.2, 1.75, 4.1, 1.25, 'Fit cutoff and freeze export', 'Calibration: 482 images / 268 groups\nSave cutoff, members and weight hashes')
    arrow((2.25, 3.55), (2.25, 3.0))
    box(.2, .05, 4.1, 1.15, 'Exploratory source test', '618 images / 426 groups\nNo feedback into training or selection', '#f6eddf')
    arrow((2.25, 1.75), (2.25, 1.2))
    box(5.35, 7.45, 4.45, 1.6, 'Secondary grouped cross-validation', 'Same original training cohort only\nThree outer folds, seed 20261115\nOriginal selection/calibration/test excluded')
    arrow((4.3, 8.42), (5.35, 8.42))
    box(5.35, 5.05, 4.45, 1.85, 'Within each outer training fold', 'Separate inner fit and holdout groups\nThree recipes × three seeds\nInner holdout selects checkpoints/cutoff\n27 fresh neural models in total')
    arrow((7.58, 7.45), (7.58, 6.9))
    box(5.35, 2.85, 4.45, 1.65, 'Freeze, then score outer groups', 'Average three seeds per recipe/fold\nEach image gets one held-out-fold score\nReport all recipes without model promotion')
    arrow((7.58, 5.05), (7.58, 4.5))
    box(5.35, .55, 4.45, 1.75, 'Independent reproduction and control', 'Fresh CPU predictions and artifact checks\nSame-fold image-dimension logistic control\nNo clinical or external validation claim', '#edf2e7')
    arrow((7.58, 2.85), (7.58, 2.3))
    fig.subplots_adjust(left=.025, right=.99, top=.99, bottom=.025)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT / ('protocol_flow.' + suffix), dpi=180, facecolor='white')
    plt.close(fig)
    print((OUT / 'protocol_flow.pdf').relative_to(ROOT).as_posix())


if __name__ == '__main__': build()
