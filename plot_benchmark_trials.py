"""Plot every prespecified trial; the test set never determines which trial is shown."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source,output,ensemble_metrics=None):
    records=json.loads(Path(source).read_text())
    architectures=['efficientnet_b0','densenet121','resnet18'];seeds=[42,123,2026]
    expected={(a,s) for a in architectures for s in seeds}
    if len(records)!=9 or {(r['architecture'],r['seed']) for r in records}!=expected:
        raise ValueError('Require all nine trials, without selection by test outcome')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    colors=['#2166ac','#b35806','#53822b'];markers=['o','s','^']
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for i,architecture in enumerate(architectures):
        group=sorted([r for r in records if r['architecture']==architecture],key=lambda r:seeds.index(r['seed']))
        for j,r in enumerate(group):
            x=i+(j-1)*.13
            axes[0].scatter(x,r['selection_auroc'],marker=markers[j],color=colors[i],s=45)
            axes[1].scatter(x,r['calibrated_test']['auroc'],marker=markers[j],color=colors[i],s=45)
            axes[2].plot([0,1],[r['raw_test']['brier_score'],r['calibrated_test']['brier_score']],
                         marker=markers[j],color=colors[i],alpha=.8,linewidth=1)
        for axis,key in [(axes[0],'selection'),(axes[1],'test')]:
            values=[r['selection_auroc'] if key=='selection' else r['calibrated_test']['auroc'] for r in group]
            axis.hlines(np.mean(values),i-.25,i+.25,color=colors[i],linewidth=2)
    for axis in axes[:2]:
        axis.set_xticks(range(3),['EfficientNet\nB0','DenseNet\n121','ResNet\n18'])
        axis.set_ylim(.93,1.005);axis.set_ylabel('AUROC');axis.grid(axis='y',alpha=.2)
    axes[0].set_title('A  Model-selection partition',loc='left',fontsize=11)
    axes[1].set_title('B  Source test (exploratory)',loc='left',fontsize=11)
    axes[2].set_title('C  Test calibration by trial',loc='left',fontsize=11)
    axes[2].set_xticks([0,1],['Raw','Temperature-scaled']);axes[2].set_ylabel('Brier score (lower is better)')
    axes[2].set_xlim(-.2,1.2);axes[2].grid(axis='y',alpha=.2)
    if ensemble_metrics:
        ensemble=json.loads(Path(ensemble_metrics).read_text())
        axes[2].plot([0,1],[ensemble['raw_test_at_0_5']['brier_score'],ensemble['test']['brier_score']],
                     'D--',color='black',linewidth=1.5,label='Selected ensemble')
        axes[2].legend(frameon=False,fontsize=8,loc='lower center')
    handles=[plt.Line2D([],[],marker=markers[j],color='black',linestyle='',label=f'Seed {seed}') for j,seed in enumerate(seeds)]
    axes[0].legend(handles=handles,loc='lower left',frameon=False,fontsize=9)
    fig.savefig(output/'trial_discrimination_and_calibration.png',dpi=240)
    fig.savefig(output/'trial_discrimination_and_calibration.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ensemble-metrics',type=Path)
    args=parser.parse_args();plot(args.source,args.output,args.ensemble_metrics)
