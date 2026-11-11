"""Presentation plots from the final reproduction CSV/NPZ data.

Run with the reproduction's Python 3.7 CPU environment (matplotlib 3.2.2).
The statistics, smoothing and seed aggregation match report/figures/*.py.
Only chart dimensions, typography and panel arrangement differ.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parent.parent)
ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'images')
args = ap.parse_args()
repo, out = args.repo_root.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']
COL = {'TRPO-CBF': '#1baf7a', 'DDPG-CBF': '#eda100', 'TRPO': '#2a78d6', 'DDPG': '#eb6834'}
LS = {a: '-' if a.endswith('CBF') else (0, (5, 2)) for a in ALGS}
BOUND = (0, (4, 3))
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    'axes.linewidth': .65, 'axes.edgecolor': '#333333',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': .45,
    'lines.linewidth': 1.25, 'legend.frameon': False, 'pdf.fonttype': 42,
    'savefig.dpi': 220,
})

def read(path):
    with open(str(path), newline='') as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] else np.nan for r in rows]) for k in rows[0]}

def smooth(y):
    c = np.cumsum(np.insert(np.asarray(y, float), 0, 0.))
    i = np.arange(len(y)); lo = np.maximum(0, i - 9)
    return (c[i+1] - c[lo]) / (i+1-lo)

def band(ax, x, values, alg):
    m = np.asarray(values)
    ax.fill_between(x, m.min(0), m.max(0), color=COL[alg], alpha=.18, lw=0)
    ax.plot(x, m.mean(0), color=COL[alg], ls=LS[alg])

def training_legend(fig):
    h = [Line2D([], [], color=COL[a], ls=LS[a]) for a in ALGS]
    h.append(Line2D([], [], color='black', ls=BOUND, lw=1))
    fig.legend(h, ALGS + ['Safety boundary'], ncol=3, loc='upper center',
               bbox_to_anchor=(.53, .995), handlelength=1.8,
               columnspacing=1.2, labelspacing=.35)

def save(fig, name):
    fig.savefig(str(out / (name+'.pdf')))
    fig.savefig(str(out / (name+'.png')))
    plt.close(fig)

# Training curves: identical data and aggregation to report figures 1 and 3.
pend = {a: [read(repo/'pendulum/results/final_600ep'/a/('seed%d'%s)/'episodes.csv')
            for s in (1,2,3)] for a in ALGS}
fig, ax = plt.subplots(2,1,figsize=(4.2,3.25),sharex=True)
fig.subplots_adjust(left=.17,right=.965,top=.84,bottom=.13,hspace=.32)
x = np.arange(1,601)
for a in ALGS:
    band(ax[0],x,[smooth(d['max_angle']) for d in pend[a]],a)
    band(ax[1],x,[smooth(d['reward']) for d in pend[a]],a)
ax[0].axhline(1,color='black',ls=BOUND,lw=1)
ax[0].set_ylim(0,np.pi+.05);ax[0].set_yticks([0,1,2,3]);ax[0].set_ylabel('Max. angle (rad)')
ax[1].set_yscale('symlog',linthreshy=10,linscaley=.6)
ax[1].set_ylim(-4000,3);ax[1].set_yticks([0,-10,-100,-1000])
ax[1].set_yticklabels(['0',r'$-10$',r'$-10^2$',r'$-10^3$'])
ax[1].set_ylabel('Episode return');ax[1].set_xlabel('Episode')
ax[1].set_xlim(0,600);ax[1].set_xticks([0,200,400,600])
training_legend(fig);save(fig,'repro_pendulum_training')

car = {a: [read(repo/'car/results/car_final'/a/('seed%d'%s)/'episodes.csv')
           for s in (1,2,3)] for a in ALGS}
grid=np.arange(1000,300001,1000)
def on_steps(d,y):
    idx=np.minimum(np.searchsorted(np.cumsum(d['length']),grid),len(d['length'])-1)
    return y[idx]
fig,ax=plt.subplots(2,1,figsize=(4.2,3.25),sharex=True)
fig.subplots_adjust(left=.17,right=.965,top=.84,bottom=.13,hspace=.32)
for a in ALGS:
    band(ax[0],grid/1000.,[on_steps(d,smooth(d['min_headway'])) for d in car[a]],a)
    band(ax[1],grid/1000.,[on_steps(d,smooth(d['return'])) for d in car[a]],a)
ax[0].axhline(2,color='black',ls=BOUND,lw=1);ax[0].axhline(0,color='#888888',lw=.6)
ax[0].set_ylim(-1,4.5);ax[0].set_yticks([0,2,4]);ax[0].set_ylabel('Min. gap (m)')
ax[1].set_yscale('symlog',linthreshy=100,linscaley=.5)
ax[1].set_ylim(-1e5,0);ax[1].set_yticks([0,-100,-1000,-10000,-100000])
ax[1].set_yticklabels(['0',r'$-10^2$',r'$-10^3$',r'$-10^4$',r'$-10^5$'])
ax[1].set_ylabel('Episode return');ax[1].set_xlabel(r'Environment steps ($\times10^3$)')
ax[1].set_xlim(0,300);ax[1].set_xticks([0,100,200,300])
training_legend(fig);save(fig,'repro_car_training')

# Pendulum: all ten initial states, identical first/final policy trajectories to report figure 2.
fig,ax=plt.subplots(1,2,figsize=(6.4,2.45),sharey=True)
fig.subplots_adjust(left=.095,right=.985,top=.75,bottom=.21,wspace=.19)
for a,axis in zip(ALGS[:2],ax):
    z=np.load(str(repo/'pendulum/results/checkpoint_rerun_seed1'/a/'seed1/eval_first_last.npz'))
    t=np.arange(z['first_theta'].shape[1])*float(z['dt'])
    for tag,c,ls in [('first','#2a78d6','-'),('last','#d03b3b',(0,(5,1.5,1,1.5)))]:
        for th in (z[tag+'_theta']+np.pi)%(2*np.pi)-np.pi:
            row=th.copy();row[1:][np.abs(np.diff(th))>np.pi]=np.nan
            axis.plot(t,row,color=c,ls=ls,lw=.95,alpha=.9)
    for boundary in [-1,1]:axis.axhline(boundary,color='black',ls=BOUND,lw=1)
    axis.set_title(a,pad=5);axis.set_ylim(-1.2,1.2);axis.set_yticks([-1,0,1])
    axis.set_xlim(0,10);axis.set_xticks([0,5,10]);axis.set_xlabel('Time (s)')
ax[0].set_ylabel(r'Angle $\theta$ (rad)')
h=[Line2D([],[],color='#2a78d6'),Line2D([],[],color='#d03b3b',ls=(0,(5,1.5,1,1.5))),
   Line2D([],[],color='black',ls=BOUND)]
fig.legend(h,['Before training','After training','Safety boundary'],ncol=3,
           loc='upper center',bbox_to_anchor=(.52,.995),fontsize=9,handlelength=2.1)
save(fig,'repro_pendulum_policies')

# Car following: preserve all three response variables, in a horizontal layout for projection.
z=np.load(str(repo/'car/results/car_eval/sim_seed1.npz'))
E={a:{k.split('__')[1]:z[k] for k in z.files if k.startswith(a+'__')} for a in ALGS}
fig,ax=plt.subplots(1,3,figsize=(8.2,2.55))
fig.subplots_adjust(left=.065,right=.985,top=.69,bottom=.22,wspace=.43)
for a in ALGS:
    e=E[a];dmin=np.minimum(e['d_f'],e['d_r'])
    ax[0].plot(e['t'],dmin,color=COL[a],ls=LS[a])
    if bool(e['collided']):ax[0].plot(e['t'][-1],dmin[-1],'x',color=COL[a],ms=6,mew=1.5)
    ax[1].plot(e['t'],e['v'][:,3],color=COL[a],ls=LS[a])
    ax[2].plot(e['t'][:-1],e['u'][:-1],color=COL[a],ls=LS[a],lw=1)
ax[0].axhline(2,color='black',ls=BOUND,lw=1)
ax[0].axhline(3,color='#888888',ls=':',lw=1)
ax[0].set_ylim(-.6,7.2);ax[0].set_yticks([0,2,4,6]);ax[0].set_ylabel('Min. gap (m)')
ax[1].plot(E['TRPO-CBF']['t'],E['TRPO-CBF']['v'][:,2],color='#888888',alpha=.6,lw=1.6,zorder=1)
ax[1].set_ylabel('Speed (m/s)')
ax[2].axhline(0,color='#888888',lw=.6);ax[2].set_ylabel(r'Input (m/s$^2$)')
for axis,title in zip(ax,['(a) Front/rear safety','(b) Car 4 speed','(c) Executed input']):
    axis.set_title(title,pad=4);axis.set_xlabel('Time (s)');axis.set_xlim(0,10);axis.set_xticks([0,5,10])
h=[Line2D([],[],color=COL[a],ls=LS[a]) for a in ALGS]
h.extend([Line2D([],[],color='black',ls=BOUND),Line2D([],[],color='#888888',ls=':'),
          Line2D([],[],color='#888888',alpha=.6),Line2D([],[],color='black',ls='',marker='x')])
fig.legend(h,ALGS+['2 m safety gap','3 m soft threshold','Car 3 (ahead)','Collision'],ncol=4,
           loc='upper center',bbox_to_anchor=(.53,1.005),fontsize=9.5,
           columnspacing=1.7,handlelength=2.1,labelspacing=.5)
save(fig,'repro_car_policies')
print('Created four vector plots from the final reproduction data.')
