"""Pendulum figures and summary metrics for the reproduction report.

Inputs (relative to this file):
  ../../pendulum/results/final_600ep/<ALG>/seed<k>/episodes.csv            (final code, seeds 1-3, 600 episodes)
  ../../pendulum/results/checkpoint_rerun_seed1/<ALG>/seed1/eval_first_last.npz (first/last policy evaluation)
  ../../pendulum/results/author_code_default/<ALG>/seed<k>/episodes.csv     (author code, default settings)
Outputs (next to this file):
  pendulum_training.pdf   main text Fig. 1  (max |theta| and return, mean +- 1 std over seeds)
  pendulum_safety_cbf.pdf supplement        (cumulative unsafe episodes, RMS of u_CBF)
  pendulum_first_last.pdf main text Fig. 2  (angle trajectories under first vs last policy)
  pendulum_metrics.json
usage: python make_pendulum_figs.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PEND = os.path.join(HERE, '..', '..', 'pendulum')
ROOT = os.path.join(PEND, 'results', 'final_600ep')
CKPT = os.environ.get('CKPT_DIR', os.path.join(PEND, 'results', 'checkpoint_rerun_seed1'))   # override only for previews
ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']
# validated categorical palette (fixed order); baselines dashed, CBF solid as secondary encoding
COLOR = {'TRPO': '#2a78d6', 'DDPG': '#eb6834', 'TRPO-CBF': '#1baf7a', 'DDPG-CBF': '#eda100'}
STYLE = {'TRPO': (0, (5, 2)), 'DDPG': (0, (5, 2)), 'TRPO-CBF': '-', 'DDPG-CBF': '-'}
SEEDS = (1, 2, 3)
WIN = 10          # moving-average window (episodes)
INK, GRID = '#1a1a1a', '#d9d9d9'

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


def load(alg, seed):
    with open(os.path.join(ROOT, alg, 'seed%d' % seed, 'episodes.csv')) as f:
        rows = list(csv.DictReader(f))
    out = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    assert len(out['episode']) == 600
    return out


def smooth(x, w=WIN):
    """Trailing moving average (shorter window at the start, no look-ahead)."""
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0.))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def band(ax, x, M, alg, label=True):
    """Mean over seeds (line) and min-max over seeds (shaded), as in the SAC paper."""
    m = M.mean(0)
    ax.fill_between(x, M.min(0), M.max(0), color=COLOR[alg], alpha=0.18, lw=0, zorder=2)
    ax.plot(x, m, color=COLOR[alg], ls=STYLE[alg], label=alg if label else None, zorder=3)


def tag(ax, s):
    ax.set_title(s, loc='left', fontsize=10, pad=4)


data = {a: [load(a, s) for s in SEEDS] for a in ALGS}
ep = np.arange(1, 601)

# ---------- Fig. 1: training curves (paper Fig. 3 style) ----------
fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.3), sharex=True, gridspec_kw={'hspace': 0.28})
ax = axs[0]
for a in ALGS:
    band(ax, ep, np.array([smooth(d['max_angle']) for d in data[a]]), a)
ax.axhline(1.0, color='k', lw=1.0, ls=(0, (4, 3)), zorder=4)
ax.set_ylim(0, np.pi + 0.05)
ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax.set_ylabel(r'Max. angle $|\theta|$ (rad)')
tag(ax, r'(a) Maximum angle per episode')
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], color='k', lw=1.0, ls=(0, (4, 3)))); l.append('Safety boundary')
fig.legend(h, l, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.965), frameon=False, fontsize=8.5,
           handlelength=2.4, columnspacing=1.1)

ax = axs[1]
for a in ALGS:
    band(ax, ep, np.array([smooth(d['reward']) for d in data[a]]), a, label=False)
ax.set_yscale('symlog', linthreshy=10, linscaley=0.6, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
ax.set_ylim(-4000, 3)
ax.set_yticks([0, -10, -100, -1000])
ax.set_yticklabels(['$0$', r'$-10$', r'$-10^2$', r'$-10^3$'])
ax.set_ylabel('Episode return')
ax.set_xlabel('Episode')
tag(ax, r'(b) Episode return (symmetric-log scale)')
ax.set_xlim(0, 600)
ax.set_xticks(np.arange(0, 601, 100))
for ax in axs:
    ax.minorticks_on()
    ax.grid(True, which='major')
    ax.grid(False, which='minor')
fig.savefig(os.path.join(HERE, 'pendulum_training.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- supplement: safety counts and CBF intervention ----------
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.35), gridspec_kw={'wspace': 0.3})
ax = axs[0]
for a in ALGS:
    band(ax, ep, np.array([np.cumsum(d['violation']) for d in data[a]]), a)
ax.set_ylabel('Cumulative unsafe episodes')
ax.set_ylim(-5, 300)
tag(ax, '(a) Unsafe episodes')
ax.legend(loc='upper left', fontsize=7.5)
ax = axs[1]
for a in ('TRPO-CBF', 'DDPG-CBF'):
    band(ax, ep, np.array([smooth(d['correction_rms']) for d in data[a]]), a)
ax.set_ylabel(r'RMS of $u^{\mathrm{CBF}}$ (N$\cdot$m)')
ax.set_ylim(0, 4.5)
tag(ax, r'(b) CBF correction')
ax.legend(loc='upper right', fontsize=7.5)
for ax in axs:
    ax.set_xlabel('Episode'); ax.set_xlim(0, 600); ax.set_xticks(np.arange(0, 601, 200))
fig.savefig(os.path.join(HERE, 'pendulum_safety_cbf.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- supplement: per-seed return curves ----------
fig, axs = plt.subplots(3, 1, figsize=(6.3, 6.2), sharex=True, gridspec_kw={'hspace': 0.3})
for k, (ax, s) in enumerate(zip(axs, SEEDS)):
    for a in ALGS:
        ax.plot(ep, smooth(data[a][k]['reward']), color=COLOR[a], ls=STYLE[a], label=a)
    ax.set_yscale('symlog', linthreshy=10, linscaley=0.6, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_ylim(-6000, 3)
    ax.set_yticks([0, -10, -100, -1000])
    ax.set_yticklabels(['$0$', r'$-10$', r'$-10^2$', r'$-10^3$'])
    ax.set_ylabel('Episode return')
    tag(ax, '(%s) Seed %d' % ('abc'[k], s))
    ax.minorticks_on(); ax.grid(True, which='major'); ax.grid(False, which='minor')
axs[-1].set_xlabel('Episode'); axs[-1].set_xlim(0, 600); axs[-1].set_xticks(np.arange(0, 601, 100))
axs[0].legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize=8.5,
              handlelength=2.4, columnspacing=1.3)
fig.savefig(os.path.join(HERE, 'pendulum_return_per_seed.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- supplement: author code, default settings (first 600 episodes) ----------
ORIG = os.path.join(PEND, 'results', 'author_code_default')


def load_orig(alg, seed, n=600):
    with open(os.path.join(ORIG, alg, 'seed%d' % seed, 'episodes.csv')) as f:
        rows = list(csv.DictReader(f))[:n]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


orig = {a: [load_orig(a, s) for s in SEEDS] for a in ALGS}
fig, axs = plt.subplots(2, 1, figsize=(6.3, 4.3), sharex=True, gridspec_kw={'hspace': 0.28})
for a in ALGS:
    band(axs[0], ep, np.array([smooth(d['max_angle']) for d in orig[a]]), a)
    band(axs[1], ep, np.array([smooth(d['reward']) for d in orig[a]]), a, label=False)
axs[0].axhline(1.0, color='k', lw=1.0, ls=(0, (4, 3)), zorder=4)
axs[0].set_ylim(0, np.pi + 0.05); axs[0].set_yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
axs[0].set_ylabel(r'Max. angle $|\theta|$ (rad)')
tag(axs[0], '(a) Maximum angle per episode')
h, l = axs[0].get_legend_handles_labels()
h.append(Line2D([], [], color='k', lw=1.0, ls=(0, (4, 3)))); l.append('Safety boundary')
fig.legend(h, l, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.965), frameon=False, fontsize=8.5,
           handlelength=2.4, columnspacing=1.1)
ax = axs[1]
ax.set_yscale('symlog', linthreshy=10, linscaley=0.6, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
ax.set_ylim(-4000, 3); ax.set_yticks([0, -10, -100, -1000])
ax.set_yticklabels(['$0$', r'$-10$', r'$-10^2$', r'$-10^3$'])
ax.set_ylabel('Episode return'); ax.set_xlabel('Episode')
tag(ax, '(b) Episode return (symmetric-log scale)')
ax.set_xlim(0, 600); ax.set_xticks(np.arange(0, 601, 100))
for ax in axs:
    ax.minorticks_on(); ax.grid(True, which='major'); ax.grid(False, which='minor')
fig.savefig(os.path.join(HERE, 'pendulum_original_training.pdf'), bbox_inches='tight')
plt.close(fig)

# ---------- Fig. 2: first vs last policy (paper Fig. 4 style) ----------
ev = {}
for a in ALGS:
    p = os.path.join(CKPT, a, 'seed1', 'eval_first_last.npz')
    if os.path.exists(p):
        ev[a] = dict(np.load(p))
SHOWN = ['TRPO-CBF', 'DDPG-CBF']        # only the CBF variants are drawn (as in the paper's Fig. 4)
if all(a in ev for a in SHOWN):
    FIRST, LAST = '#2a78d6', '#d03b3b'
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.45), gridspec_kw={'wspace': 0.28})
    for ax, a in zip(axs.ravel(), SHOWN):
        E = ev[a]
        t = np.arange(E['first_theta'].shape[1]) * float(E['dt'])
        for key, col, ls in (('first', FIRST, '-'), ('last', LAST, (0, (5, 1.5, 1, 1.5)))):
            th = (E[key + '_theta'] + np.pi) % (2 * np.pi) - np.pi
            for row in th:                     # break the line at +-pi wrap-arounds
                r = row.copy(); r[1:][np.abs(np.diff(row)) > np.pi] = np.nan
                ax.plot(t, r, color=col, ls=ls, lw=0.9, alpha=0.8)
        for b in (1, -1):
            ax.axhline(b, color='k', lw=1.0, ls=(0, (4, 3)))
        cbf = a.endswith('CBF')
        ax.set_ylim(*((-1.25, 1.25) if cbf else (-np.pi - 0.1, np.pi + 0.1)))
        ax.set_yticks([-1, -0.5, 0, 0.5, 1] if cbf else [-3, -2, -1, 0, 1, 2, 3])
        ax.set_xlim(0, 10)
        ax.set_title(a, fontsize=10, fontweight='bold', pad=4)
        ax.minorticks_on(); ax.grid(False, which='minor')
    axs[0].set_ylabel(r'Angle $\theta$ (rad)')
    for ax in axs:
        ax.set_xlabel('Time (s)')
    handles = [Line2D([], [], color=FIRST, lw=1.2), Line2D([], [], color=LAST, lw=1.2, ls=(0, (5, 1.5, 1, 1.5))),
               Line2D([], [], color='k', lw=1.0, ls=(0, (4, 3)))]
    fig.subplots_adjust(top=0.78)
    fig.legend(handles, ['First policy', 'Last policy', 'Safety boundary'], loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=8.5)
    fig.savefig(os.path.join(HERE, 'pendulum_first_last.pdf'), bbox_inches='tight')
    plt.close(fig)
else:
    print('first/last evaluation not available yet:', sorted(ev))

# ---------- metrics ----------
def first_cross(r, thr=-10., w=20):
    s = np.convolve(r, np.ones(w) / w, mode='valid')
    idx = np.where(s > thr)[0]
    return int(idx[0] + w) if len(idx) else None

metrics = {}
for name, src in (('', data), ('original ', orig)):
  for a in ALGS:
    per = []
    for s, d in zip(SEEDS, src[a]):
        e = dict(seed=s,
                 first100=float(d['reward'][:100].mean()),
                 last100=float(d['reward'][-100:].mean()),
                 last100_std_within=float(d['reward'][-100:].std()),
                 unsafe_episodes=int(d['violation'].sum()),
                 unsafe_last100=int(d['violation'][-100:].sum()),
                 max_angle_overall=float(d['max_angle'].max()),
                 first_ep_ma20_above_m10=first_cross(d['reward']))
        if 'slack' in d:
            e.update(slack_episodes=int((d['slack'] > 1e-9).sum()),
                     max_slack=float(d['slack'].max()))
        if 'correction_rms' in d:
            e.update(corr_rms_first10=float(d['correction_rms'][:10].mean()),
                     corr_rms_last100=float(d['correction_rms'][-100:].mean()),
                     unsafe_episode_ids=[int(i) for i in d['episode'][d['violation'] > 0]][:20])
        e['unsafe_by_segment'] = [int(d['violation'][i:j].sum()) for i, j in ((0, 100), (100, 300), (300, 600))]
        per.append(e)
    L = np.array([e['last100'] for e in per])
    metrics[name + a] = dict(last100_mean=float(L.mean()), last100_std=float(L.std(ddof=1)), seeds=per)
for a, E in ev.items():
    out = {}
    for key in ('first', 'last'):
        th = (E[key + '_theta'] + np.pi) % (2 * np.pi) - np.pi
        out[key] = dict(max_abs_theta=[float(x) for x in np.abs(th).max(1)],
                        final_abs_theta_mean=float(np.abs(th[:, -20:]).mean()),
                        max_abs_ucbf=float(np.abs(E[key + '_ucbf']).max()))
    metrics[a + ' eval_seed1'] = out
with open(os.path.join(HERE, 'pendulum_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=1)
