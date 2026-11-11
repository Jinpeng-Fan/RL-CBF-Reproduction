"""Car-following figures and summary metrics for the reproduction report.

Inputs (relative to this file):
  ../../car/results/car_final/<ALG>/seed<k>/episodes.csv            revised code, seeds 1-3, 300k steps
  ../../car/results/car_author_default/<ALG>/seed<k>/episodes.csv   author code, default settings
      (written by extract_car_original.py), and car_author_default/qp_errors.json
Outputs (next to this file):
  car_training.pdf          main text: min headway and return of the revised code vs environment steps
  car_original_training.pdf supplement: the same for the author code, first 1000 episodes
  car_safety_cbf.pdf        supplement: cumulative collisions / 2 m exits, RMS of u_CBF and compensator
  car_per_seed.pdf          supplement: per-seed return curves (revised and author code)
  car_metrics.json
usage: python make_car_figs.py

Curves: each seed is smoothed with a trailing 10-episode moving average; the line is the mean over
the 3 seeds and the band the min-max over seeds.  Revised runs end episodes at a collision, so the
number of episodes differs between algorithms; their curves are therefore drawn against environment
steps (value of the episode that contains the step).  Author episodes all have 80 steps and are drawn
against the episode index.
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CAR = os.path.join(HERE, '..', '..', 'car', 'results')
REV = os.path.join(CAR, 'car_final')
ORIG = os.path.join(CAR, 'car_author_default')
ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']
COLOR = {'TRPO': '#2a78d6', 'DDPG': '#eb6834', 'TRPO-CBF': '#1baf7a', 'DDPG-CBF': '#eda100'}
STYLE = {'TRPO': (0, (5, 2)), 'DDPG': (0, (5, 2)), 'TRPO-CBF': '-', 'DDPG-CBF': '-'}
SEEDS = (1, 2, 3)
WIN = 10
D_MIN = 2.0
INK, GRID = '#1a1a1a', '#d9d9d9'
BOUND = dict(color='k', lw=1.0, ls=(0, (4, 3)), zorder=4)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9.5,
    'legend.fontsize': 8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': 0.7, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.45, 'grid.linestyle': '-', 'legend.frameon': True, 'legend.framealpha': 0.92,
    'legend.edgecolor': '#999999', 'legend.fancybox': False, 'lines.linewidth': 1.3,
    'pdf.fonttype': 42, 'savefig.dpi': 300})


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    num = lambda v: float(v) if v not in ('', None) else np.nan
    return {k: np.array([num(r[k]) for r in rows]) for k in rows[0]}


def smooth(x, w=WIN):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def on_steps(d, y, grid):
    """Value of y for the episode that contains each environment step in grid."""
    end = np.cumsum(d['length'])
    idx = np.minimum(np.searchsorted(end, grid), len(end) - 1)
    return y[idx]


def band(ax, x, M, alg, label=True):
    ax.fill_between(x, M.min(0), M.max(0), color=COLOR[alg], alpha=0.18, lw=0, zorder=2)
    ax.plot(x, M.mean(0), color=COLOR[alg], ls=STYLE[alg], label=alg if label else None, zorder=3)


def tag(ax, s):
    ax.set_title(s, loc='left', fontsize=10, pad=4)


def symlog_return(ax, lo):
    ax.set_yscale('symlog', linthreshy=100, linscaley=0.5, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_ylim(lo, 0)
    ks = [k for k in range(2, 9) if -10 ** k >= lo]
    ax.set_yticks([0] + [-10 ** k for k in ks])
    ax.set_yticklabels(['$0$'] + [r'$-10^{%d}$' % k for k in ks])


def finish(axs):
    for ax in axs:
        ax.minorticks_on(); ax.grid(True, which='major'); ax.grid(False, which='minor')


def top_legend(fig, ax, y=0.965):
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], **{k: v for k, v in BOUND.items() if k != 'zorder'})); l.append('Safety boundary')
    fig.legend(h, l, loc='upper center', ncol=5, bbox_to_anchor=(0.5, y), frameon=False, fontsize=8.5,
               handlelength=2.4, columnspacing=1.1)


rev = {a: [read(os.path.join(REV, a, 'seed%d' % s, 'episodes.csv')) for s in SEEDS] for a in ALGS}
orig = {a: [read(os.path.join(ORIG, a, 'seed%d' % s, 'episodes.csv')) for s in SEEDS] for a in ALGS}
qp_err = json.load(open(os.path.join(ORIG, 'qp_errors.json')))

# ---------- main figure: revised code vs environment steps (paper Fig. 5 style) ----------
grid = np.arange(1000, 300001, 1000)
xk = grid / 1000.
fig, axs = plt.subplots(2, 1, figsize=(6.3, 3.9), sharex=True, gridspec_kw={'hspace': 0.3})
for a in ALGS:
    band(axs[0], xk, np.array([on_steps(d, smooth(d['min_headway']), grid) for d in rev[a]]), a)
    band(axs[1], xk, np.array([on_steps(d, smooth(d['return']), grid) for d in rev[a]]), a, label=False)
axs[0].axhline(D_MIN, **BOUND)
axs[0].axhline(0.0, color='#777777', lw=0.7, zorder=1)
axs[0].set_ylim(-1, 4.5)
axs[0].set_ylabel('Min. headway (m)')
tag(axs[0], '(a) Minimum headway of car 4 per episode')
top_legend(fig, axs[0], y=1.0)
symlog_return(axs[1], -1e5)
axs[1].set_ylabel('Episode return')
axs[1].set_xlabel(r'Environment steps ($\times 10^3$)')
tag(axs[1], '(b) Episode return (symmetric-log scale)')
axs[1].set_xlim(0, 300); axs[1].set_xticks(np.arange(0, 301, 50))
finish(axs)
fig.savefig(os.path.join(HERE, 'car_training.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- supplement: author code, first 1000 episodes ----------
N_O = 1000
ep = np.arange(1, N_O + 1)
fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.4), sharex=True, gridspec_kw={'hspace': 0.28})
for a in ALGS:
    band(axs[0], ep, np.array([smooth(d['min_headway'])[:N_O] for d in orig[a]]), a)
    band(axs[1], ep, np.array([smooth(d['return'])[:N_O] for d in orig[a]]), a, label=False)
axs[0].axhline(D_MIN, **BOUND)
axs[0].axhline(0.0, color='#777777', lw=0.7, zorder=1)
axs[0].set_ylim(-10, 6)          # as paper Fig. 5; author DDPG reaches -150 m (clipped)
axs[0].set_ylabel('Min. headway (m)')
tag(axs[0], '(a) Minimum headway of car 4 per episode')
top_legend(fig, axs[0])
symlog_return(axs[1], -1e6)
axs[1].set_ylabel('Episode return')
axs[1].set_xlabel('Episode (80 steps each)')
tag(axs[1], '(b) Episode return (symmetric-log scale)')
axs[1].set_xlim(0, N_O)
finish(axs)
fig.savefig(os.path.join(HERE, 'car_original_training.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- supplement: safety counts and CBF intervention (revised code) ----------
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw={'wspace': 0.36})
for a in ALGS:
    band(axs[0], xk, np.array([on_steps(d, np.cumsum(d['collision']), grid) for d in rev[a]]), a)
    band(axs[1], xk, np.array([on_steps(d, np.cumsum(d['exit_2m']), grid) for d in rev[a]]), a, label=False)
for a in ('TRPO-CBF', 'DDPG-CBF'):
    band(axs[2], xk, np.array([on_steps(d, smooth(d['cbf_rms']), grid) for d in rev[a]]), a, label=False)
    M = np.array([on_steps(d, smooth(d['comp_rms']), grid) for d in rev[a]])
    axs[2].plot(xk, M.mean(0), color=COLOR[a], ls=(0, (1, 1.2)), lw=1.1, zorder=3)
axs[0].set_ylabel('Cumulative collision episodes'); tag(axs[0], '(a) Collisions')
axs[1].set_ylabel(r'Cumulative episodes below 2 m'); tag(axs[1], '(b) Safety-distance exits')
axs[2].set_ylabel(r'RMS (m/s$^2$)'); tag(axs[2], '(c) CBF correction / compensator')
axs[2].set_yscale('symlog', linthreshy=1, linscaley=0.6)
axs[2].legend(handles=[Line2D([], [], color='k', lw=1.2), Line2D([], [], color='k', lw=1.1, ls=(0, (1, 1.2)))],
              labels=[r'$u^{\mathrm{CBF}}$', r'$\bar u_\phi$'], loc='upper right', fontsize=7.5)
axs[0].legend(loc='upper left', fontsize=7)
for ax in axs:
    ax.set_xlabel(r'Steps ($\times 10^3$)'); ax.set_xlim(0, 300); ax.set_xticks(np.arange(0, 301, 100))
fig.savefig(os.path.join(HERE, 'car_safety_cbf.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- supplement: per-seed return curves ----------
fig, axs = plt.subplots(3, 2, figsize=(7.0, 6.2), gridspec_kw={'hspace': 0.34, 'wspace': 0.22})
for k, s in enumerate(SEEDS):
    for a in ALGS:
        d = rev[a][k]
        axs[k, 0].plot(xk, on_steps(d, smooth(d['return']), grid), color=COLOR[a], ls=STYLE[a], label=a)
        d = orig[a][k]
        axs[k, 1].plot(ep, smooth(d['return'])[:N_O], color=COLOR[a], ls=STYLE[a])
    symlog_return(axs[k, 0], -1e5)
    symlog_return(axs[k, 1], -1e6)
    tag(axs[k, 0], '(%s) Revised code, seed %d' % ('ace'[k], s))
    tag(axs[k, 1], '(%s) Author code, seed %d' % ('bdf'[k], s))
    axs[k, 0].set_xlim(0, 300); axs[k, 1].set_xlim(0, N_O)
axs[2, 0].set_xlabel(r'Environment steps ($\times 10^3$)'); axs[2, 1].set_xlabel('Episode (80 steps each)')
for k in range(3):
    axs[k, 0].set_ylabel('Episode return')
finish(axs.ravel())
axs[0, 0].legend(loc='lower center', ncol=4, bbox_to_anchor=(1.08, 1.16), frameon=False, fontsize=8.5,
                 handlelength=2.4, columnspacing=1.3)
fig.savefig(os.path.join(HERE, 'car_per_seed.pdf'), bbox_inches='tight')
plt.close(fig)


# ---------- metrics ----------
def steps_to(d, thr=-5000., w=50):
    """Environment steps until the w-episode moving average of the return first exceeds thr."""
    r, end = d['return'], np.cumsum(d['length'])
    for i in range(w, len(r) + 1):
        if r[i - w:i].mean() > thr:
            return int(end[i - 1])
    return None


def summary(d, run=None):
    e = dict(episodes=int(len(d['return'])), env_steps=int(d['length'].sum()),
             first100=float(d['return'][:100].mean()), last100=float(d['return'][-100:].mean()),
             collisions=int(np.nansum(d['collision'])), exits_2m=int(np.nansum(d['exit_2m'])),
             collisions_last100=int(np.nansum(d['collision'][-100:])), exits_last100=int(np.nansum(d['exit_2m'][-100:])),
             min_headway_overall=float(np.nanmin(d['min_headway'])),
             min_headway_last100_mean=float(np.nanmean(d['min_headway'][-100:])),
             min_headway_last100_min=float(np.nanmin(d['min_headway'][-100:])),
             return_per_step_last100=float(d['return'][-100:].sum() / d['length'][-100:].sum()))
    if np.isfinite(d['cbf_rms']).any() and np.nanmax(d['cbf_rms']) > 0:
        e.update(cbf_rms_first10=float(np.nanmean(d['cbf_rms'][:10])), cbf_rms_last100=float(np.nanmean(d['cbf_rms'][-100:])),
                 comp_rms_first10=float(np.nanmean(d['comp_rms'][:10])), comp_rms_last100=float(np.nanmean(d['comp_rms'][-100:])))
    if np.isfinite(d['slack_max']).any():
        e.update(slack_episodes=int(np.nansum(d['slack_max'] > 0)), max_slack=float(np.nanmax(d['slack_max'])),
                 infeasible_steps=int(np.nansum(d['infeasible_steps'])), cbf_cond_viol=int(np.nansum(d['cbf_cond_viol'])))
    e['collisions_by_quarter'] = [int(np.nansum(q)) for q in np.array_split(d['collision'], 4)]
    if run is not None:
        e['qp_error_prints'] = qp_err.get(run)
    return e


metrics = {}
for group, src in (('revised', rev), ('author', orig)):
    for a in ALGS:
        per = []
        for s, d in zip(SEEDS, src[a]):
            e = dict(seed=s, **summary(d, None if group == 'revised' else '%s_s%d' % (a, s)))
            e['steps_to_ma50_above_m5000'] = steps_to(d)
            per.append(e)
        L = np.array([e['last100'] for e in per])
        metrics['%s %s' % (group, a)] = dict(last100_mean=float(L.mean()), last100_std=float(L.std(ddof=1)), seeds=per)
with open(os.path.join(HERE, 'car_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=1)
print('done')
