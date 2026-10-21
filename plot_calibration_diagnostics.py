"""Plot captured, reconciled Kaggle reliability bins without fitting calibration."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root=Path('.')):
    root=Path(root)
    fig,axes=plt.subplots(2,2,figsize=(11,7),gridspec_kw={'height_ratios':[2.2,1]})
    colors={'calibration':'#1565A6','test':'#C34B13'}
    labels={'calibration':'Calibration (fitted partition)','test':'Test (exploratory)'}
    for column,(size,name) in enumerate(((224,'ResNet18'),(320,'EfficientNet-B0'))):
        record=json.loads((root/f'results/benchmark{size}_prediction_diagnostics.json').read_text(encoding='utf-8'))
        top,bottom=axes[:,column]
        top.plot([0,1],[0,1],color='#777777',linestyle='--',linewidth=1,label='Score = observed frequency')
        for j,split in enumerate(('calibration','test')):
            bins=np.array(record[split]['calibration_bins'])
            if int(bins[:,2].sum())!=record[split]['n']:
                raise ValueError('Incomplete diagnostic bin counts')
            top.scatter(bins[:,0],bins[:,1],s=45,color=colors[split],marker=['o','s'][j],label=labels[split],zorder=3)
            counts=np.zeros(10,dtype=int)
            for mean,fraction,count in bins:
                index=min(int(mean*10),9)
                if counts[index]:
                    raise ValueError('Duplicate equal-width bin')
                counts[index]=int(count)
            x=np.arange(10)+(j-.5)*.36
            bars=bottom.bar(x,counts,width=.34,color=colors[split])
            bottom.bar_label(bars,labels=[str(n) if n else '' for n in counts],fontsize=7,padding=2)
        top.set(title=f'{size}-pixel {name}',xlabel='Mean score within bin',ylabel='Observed pneumonia-label fraction',
                xlim=(-.035,1.035),ylim=(-.035,1.035))
        top.grid(alpha=.18)
        bottom.set(xticks=np.arange(10),xticklabels=[f'{i/10:.1f}' for i in range(10)],
                   xlabel='Score bin lower bound (width 0.1)',ylabel='Images',ylim=(0,520))
        bottom.grid(axis='y',alpha=.18)
    fig.suptitle('Frozen-score reliability and bin populations',fontsize=13)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,fontsize=9)
    fig.text(.5,.015,'Descriptive source-partition comparison. Sparse bins are unstable; fitted-partition curves are optimistic.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.91))
    target=root/'output/figures/calibration_diagnostics';target.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf'):
        fig.savefig(target/f'frozen_reliability.{extension}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    plot(Path(__file__).resolve().parent)
