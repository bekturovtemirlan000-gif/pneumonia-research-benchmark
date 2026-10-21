"""Draw the recorded image flow without inventing patient or exclusion-cause counts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot(root=Path('.')):
    root = Path(root)
    original = json.loads((root / 'results/data_audit.json').read_text(encoding='utf-8'))
    audit = json.loads((root / 'results/benchmark224_data_audit.json').read_text(encoding='utf-8'))
    counts = audit['counts']
    development = sum(counts[s]['images'] for s in ('train', 'selection', 'calibration'))
    held = counts['selection']['images'] + counts['calibration']['images']
    raw_dev = development + original['development_removed']
    raw_test = counts['test']['images'] + original['test_duplicates_removed']
    if raw_dev + raw_test != original['raw_images'] or development + counts['test']['images'] != audit['retained_images']:
        raise ValueError('Cohort flow does not reconcile')
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.set(xlim=(0, 10), ylim=(0, 7.2)); ax.axis('off')
    def box(x, y, width, height, label, fill='#edf2f7'):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
            boxstyle='round,pad=0.05,rounding_size=0.08', linewidth=.8,
            edgecolor='#40556b', facecolor=fill))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, linespacing=1.35)
    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#40556b', lw=1.1))
    box(5, 6.65, 3.3, .65, f"Decoded source images\n{original['raw_images']:,}")
    box(3.1, 5.5, 3.1, .7, f'Original train + val\n{raw_dev:,} images')
    box(7.7, 5.5, 2.8, .7, f'Original test\n{raw_test:,} images')
    arrow((4.1, 6.27), (3.1, 5.9)); arrow((5.9, 6.27), (7.7, 5.9))
    box(3.1, 3.95, 3.1, .7, f'Development retained\n{development:,} images')
    box(7.7, 3.95, 2.8, .9, f"Source test retained\n{counts['test']['images']:,} images\n{counts['test']['groups']:,} proxy groups", '#e8f1eb')
    arrow((3.1, 5.08), (3.1, 4.38)); arrow((7.7, 5.08), (7.7, 4.48))
    ax.text(.1, 4.75, f"{original['development_removed']:,} development images removed*", fontsize=9, va='center')
    ax.text(6.25, 4.75, f"{original['test_duplicates_removed']} exact test\nduplicates removed", fontsize=9, ha='center', va='center')
    box(1.6, 2.65, 2.7, .85, f"Training\n{counts['train']['images']:,} images\n{counts['train']['groups']:,} proxy groups", '#e8f1eb')
    box(4.9, 2.65, 2.7, .85, f'Frozen held-out development\n{held:,} images')
    arrow((2.5, 3.53), (1.6, 3.12)); arrow((3.7, 3.53), (4.9, 3.12))
    box(3.45, 1.25, 2.6, .85, f"Model selection\n{counts['selection']['images']:,} images\n{counts['selection']['groups']:,} proxy groups", '#e8f1eb')
    box(6.35, 1.25, 2.6, .85, f"Calibration\n{counts['calibration']['images']:,} images\n{counts['calibration']['groups']:,} proxy groups", '#e8f1eb')
    arrow((4.4, 2.16), (3.45, 1.72)); arrow((5.4, 2.16), (6.35, 1.72))
    ax.text(7.7, 2.7, 'Same frozen cohorts\nfor both resolutions\nand all three architectures',
            fontsize=9, ha='center', va='center', color='#40556b')
    fig.text(.06, .045, '* Combined development exclusions: test-overlapping pixel hashes or filename groups, and development duplicates.\n'
             'Reason-specific counts are not separated. Groups are filename proxies, not verified patient identities.',
             fontsize=9, va='bottom')
    fig.subplots_adjust(left=.04, right=.98, top=.98, bottom=.13)
    target = root / 'output/figures/cohort_flow'; target.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(target / ('cohort_flow.' + suffix), dpi=180, facecolor='white')
    plt.close(fig)
    print('Cohort flow verified and rendered')


if __name__ == '__main__':
    plot()
