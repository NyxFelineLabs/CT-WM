"""Rebuild aggregate-result plots from public tables; no private data or weights."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/public_figures'
OUT.mkdir(parents=True,exist_ok=True)
COLORS=['#777777','#B64342','#42949E','#9A4D8E','#0F4D92']

def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'legend.frameon':False,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=read('rhc_snr.csv')
    methods=['olla','prb','sac','ppo','ctwm'];labels=['OLLA','PRB only','Joint SAC','Joint PPO','CT-WM']
    for field,ylabel,name in [('SE_change_percent','Cumulative SE change (%)','snr_se'),
                              ('PRB_saving_percent','Cumulative PRB saving (%)','snr_prb')]:
        fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
        for key,label,color in zip(methods,labels,COLORS):
            selected=sorted([r for r in rows if r['method']==key],key=lambda r:float(r['snr_db']))
            ax.plot([float(r['snr_db']) for r in selected],[float(r[field]) for r in selected],
                    color=color,marker='o',label=label)
        ax.set(xlabel='Global SINR proxy (dB)',ylabel=ylabel,xticks=[0,5,10,15,20])
        ax.grid(axis='y',color='#E3E6E9',lw=.6);ax.legend(fontsize=9)
        save(fig,name)
    rows=read('rhc_main.csv');names={'olla':'OLLA','prb':'PRB only','greedy_olla':'Greedy + OLLA',
        'sac':'SAC','ppo':'PPO','ctwm':'CT-WM','no_safety':'No gate','no_calib':'Mean-only compound'}
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for i,r in enumerate(rows):
        x,y=float(r['worst_trajectory_percent']),float(r['SE_change_percent'])
        ax.scatter(x,y,color=COLORS[i%5],s=50)
        ax.annotate(names[r['method']],(x,y),xytext=(5,7 if i%2 else -14),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Worst-trajectory SE change (%)',ylabel='Cumulative SE change (%)')
    ax.margins(x=.15,y=.18);ax.grid(color='#E3E6E9',lw=.6)
    save(fig,'rhc_benefit_risk')
    print(OUT)

if __name__=='__main__':main()
