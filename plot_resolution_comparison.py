"""Show all matched trials across both resolutions without outcome-based filtering."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(low,high,output,title='Matched architecture and seed comparisons'):
    architectures=['efficientnet_b0','densenet121','resnet18']
    labels=['EfficientNet-B0','DenseNet121','ResNet18']
    seeds=[42,123,2026];expected={(a,s) for a in architectures for s in seeds}
    arms=[]
    for source in [low,high]:
        records=json.loads(Path(source).read_text())
        if len(records)!=9 or {(r['architecture'],r['seed']) for r in records}!=expected:
            raise ValueError('Each arm must contain every planned architecture/seed pair')
        arms.append({(r['architecture'],r['seed']):r for r in records})
    colors=['#2166ac','#b35806','#53822b'];markers=['o','s','^']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    fig.subplots_adjust(top=.75,bottom=.17,wspace=.31,left=.06,right=.99)
    aucs=[];briers=[]
    for i,architecture in enumerate(architectures):
        for j,seed in enumerate(seeds):
            pair=[arm[(architecture,seed)] for arm in arms]
            values=[[r['selection_auroc'] for r in pair],
                    [r['calibrated_test']['auroc'] for r in pair],
                    [r['calibrated_test']['brier_score'] for r in pair]]
            for axis,ys in zip(axes,values):
                if not np.isfinite(ys).all() or not all(0<=v<=1 for v in ys):
                    raise ValueError('Metric outside its valid range')
                x=np.array([i*3,i*3+1])+(j-1)*.07
                axis.plot(x,ys,marker=markers[j],color=colors[i],linewidth=1,alpha=.85,markersize=5)
            aucs.extend(values[0]+values[1]);briers.extend(values[2])
    positions=[0,1,3,4,6,7]
    for axis in axes:
        axis.set_xticks(positions,['224','320']*3,fontsize=9)
        axis.set_xlim(-.4,7.4);axis.set_xlabel('Input resolution (pixels)')
        axis.grid(axis='y',alpha=.2)
        for boundary in [2,5]: axis.axvline(boundary,color='#cccccc',linewidth=.7)
    for axis in axes[:2]:
        axis.set_ylim(max(0,min(.93,min(aucs)-.01)),1.005)
        axis.set_ylabel('AUROC')
    axes[2].set_ylim(0,max(.05,max(briers)*1.15));axes[2].set_ylabel('Brier score (lower is better)')
    for axis,name in zip(axes,['A  Model selection','B  Source test (exploratory)','C  Calibrated source test']):
        axis.set_title(name,loc='left',fontsize=11)
    handles=[plt.Line2D([],[],color=color,label=label) for color,label in zip(colors,labels)]
    handles.extend(plt.Line2D([],[],marker=marker,color='black',linestyle='',label=f'Seed {seed}') for marker,seed in zip(markers,seeds))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=6,frameon=False,fontsize=9)
    fig.suptitle(title,fontsize=12,y=.99)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'matched_resolution_trials.png',dpi=240)
    fig.savefig(output/'matched_resolution_trials.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('low',type=Path);parser.add_argument('high',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.low,args.high,args.output)
