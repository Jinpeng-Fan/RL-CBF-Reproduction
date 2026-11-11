"""Car-following: closed-loop simulation of the final trained policies (figures for the report).

Input (relative to this file): ../../car/results/car_eval/sim_seed<k>.npz and sim_summary.json,
       written by car/scripts/simulate_final.py (10 s from one fixed initial state; the same
       initial state and acceleration-noise sequence for every algorithm and seed).
Outputs (next to this file):
  car_final_sim.pdf        main text: seed 1, min headway / speed of car 4 / input of car 4
  car_final_sim_seeds.pdf  supplement: seeds 2 and 3, the same rows plus the CBF correction
  car_final_sim.json       copy of sim_summary.json (all numbers quoted in the report)
usage: python make_car_sim_figs.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.environ.get('EVAL_DIR', os.path.join(HERE, '..', '..', 'car', 'results', 'car_eval'))
ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']
COLOR = {'TRPO': '#2a78d6', 'DDPG': '#eb6834', 'TRPO-CBF': '#1baf7a', 'DDPG-CBF': '#eda100'}
STYLE = {'TRPO': (0, (5, 2)), 'DDPG': (0, (5, 2)), 'TRPO-CBF': '-', 'DDPG-CBF': '-'}
INK, GRID, REF = '#1a1a1a', '#d9d9d9', '#8c8c8c'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9.5,
    'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': 0.7, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.45, 'legend.frameon': False, 'lines.linewidth': 1.3, 'pdf.fonttype': 42,
    'savefig.dpi': 300})


def load(seed):
    Z = np.load(os.path.join(EV, 'sim_seed%d.npz' % seed))
    out = {}
    for a in ALGS:
        e = {k.split('__')[1]: Z[k] for k in Z.files if k.startswith(a + '__')}
        e['dmin'] = np.minimum(e['d_f'], e['d_r'])
        out[a] = e
    return out


def tag(ax, s):
    ax.set_title(s, loc='left', fontsize=10, pad=3)


def rows(axs, E, with_cbf=False):
    """axs: headway, speed, input[, CBF correction]."""
    ax = axs[0]
    for a in ALGS:
        e = E[a]
        ax.plot(e['t'], e['dmin'], color=COLOR[a], ls=STYLE[a], zorder=3)
        if bool(e['collided']):
            ax.plot(e['t'][-1], e['dmin'][-1], 'x', color=COLOR[a], ms=6, mew=1.6, zorder=4)
    ax.axhline(2.0, color='k', lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.axhline(3.0, color=REF, lw=0.8, ls=':', zorder=2)
    ax.axhline(0.0, color=REF, lw=0.6, zorder=1)
    ax.set_ylim(-0.6, 7.2); ax.set_ylabel(r'$\min(d_f, d_r)$ (m)')
    ax = axs[1]
    ref = E['TRPO-CBF']
    ax.plot(ref['t'], ref['v'][:, 2], color=REF, lw=1.6, alpha=0.55, zorder=2)
    for a in ALGS:
        ax.plot(E[a]['t'], E[a]['v'][:, 3], color=COLOR[a], ls=STYLE[a], zorder=3)
    ax.set_ylabel(r'$v_4$ (m/s)')
    ax = axs[2]
    for a in ALGS:
        ax.plot(E[a]['t'][:-1], E[a]['u'][:-1], color=COLOR[a], ls=STYLE[a], lw=1.0, zorder=3)
    ax.axhline(0.0, color=REF, lw=0.6, zorder=1)
    ax.set_ylabel(r'$u$ (m/s$^2$)')
    if with_cbf:
        ax = axs[3]
        for a in ('TRPO-CBF', 'DDPG-CBF'):
            ax.plot(E[a]['t'][:-1], E[a]['u_cbf'][:-1], color=COLOR[a], lw=1.0, zorder=3)
        ax.axhline(0.0, color=REF, lw=0.6, zorder=1)
        ax.set_ylabel(r'$u^{\mathrm{CBF}}$ (m/s$^2$)')
    for ax in axs:
        ax.set_xlim(0, 10); ax.minorticks_on(); ax.grid(True, which='major'); ax.grid(False, which='minor')


def legend_handles():
    h = [Line2D([], [], color=COLOR[a], ls=STYLE[a], lw=1.4) for a in ALGS]
    h += [Line2D([], [], color='k', lw=1.0, ls=(0, (4, 3))), Line2D([], [], color=REF, lw=0.8, ls=':'),
          Line2D([], [], color=REF, lw=1.6, alpha=0.55), Line2D([], [], color='k', marker='x', ls='', ms=6, mew=1.5)]
    return h, ALGS + ['Safety distance 2 m', 'Soft limit 3 m', 'Car 3 (ahead)', 'Collision']


def top_legend(fig, axs, top):
    """Two-row legend in the band above the axes (the axes end at `top`)."""
    fig.subplots_adjust(top=top)
    h, l = legend_handles()
    return fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.995), fontsize=8.5,
                      handlelength=2.4, columnspacing=1.4)


# ---------- main text: seed 1 ----------
E = load(1)
fig, axs = plt.subplots(3, 1, figsize=(6.3, 4.6), sharex=True, gridspec_kw={'hspace': 0.3})
rows(axs, E)
tag(axs[0], '(a) Minimum headway of car 4')
tag(axs[1], '(b) Speed of car 4')
tag(axs[2], '(c) Executed input of car 4')
axs[2].set_xlabel('Time (s)')
leg = top_legend(fig, axs, 0.84)
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08); fig.savefig(os.path.join(HERE, 'car_final_sim.pdf')); plt.close(fig)

# ---------- supplement: seeds 2 and 3 ----------
fig, axs = plt.subplots(4, 2, figsize=(7.0, 7.0), sharex=True, gridspec_kw={'hspace': 0.24, 'wspace': 0.26})
for c, s in enumerate((2, 3)):
    rows(axs[:, c], load(s), with_cbf=True)
    tag(axs[0, c], '(%s) Seed %d' % ('ab'[c], s))
    axs[3, c].set_xlabel('Time (s)')
leg = top_legend(fig, axs, 0.89)
fig.subplots_adjust(left=0.09, right=0.985, bottom=0.07); fig.savefig(os.path.join(HERE, 'car_final_sim_seeds.pdf')); plt.close(fig)

json.dump(json.load(open(os.path.join(EV, 'sim_summary.json'))), open(os.path.join(HERE, 'car_final_sim.json'), 'w'), indent=1)
print('done')
