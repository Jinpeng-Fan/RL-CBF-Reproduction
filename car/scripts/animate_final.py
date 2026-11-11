"""Animated replay of the final-policy simulation: four algorithms side by side with live curves.

usage (from car/):  python scripts/animate_final.py [--seeds 1 2 3] [--eval results/car_eval]
Reads <eval>/sim_seed<k>.npz (scripts/simulate_final.py) and writes <eval>/animation_seed<k>.gif
(real time: every second step at 10 frames/s) and <eval>/animation_seed<k>_final.png.
Top: one lane per algorithm, view centred on car 4 (blue); cars are points in the model and are drawn\nas 1.8 m markers, so the drawn spacing equals the headway; the numbers between the cars are the
front / rear headways of car 4 (orange below 3 m, red below 2 m). Bottom: minimum headway of car 4,
speed of car 4 (grey: car 3, identical for all algorithms) and executed input, drawn up to the current time.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

CAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']
COLOR = {'TRPO': '#2a78d6', 'DDPG': '#eb6834', 'TRPO-CBF': '#1baf7a', 'DDPG-CBF': '#eda100'}
STYLE = {'TRPO': (0, (5, 2)), 'DDPG': (0, (5, 2)), 'TRPO-CBF': '-', 'DDPG-CBF': '-'}
INK, GRID, REF, ROAD = '#1a1a1a', '#d9d9d9', '#8c8c8c', '#3b3f46'
DT, CAR_LEN, VIEW = 0.05, 1.8, (-17.0, 15.0)   # cars are points in the model; drawn as 1.8 m markers
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.7, 'axes.edgecolor': INK, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.45,
    'legend.frameon': False})


def load(path):
    Z = np.load(path)
    E = {}
    for a in ALGS:
        e = {k.split('__')[1]: Z[k] for k in Z.files if k.startswith(a + '__')}
        e['dmin'] = np.minimum(e['d_f'], e['d_r'])
        e['L'] = len(e['t'])
        E[a] = e
    return E


def animate(E, seed, out_gif, fps=10, stride=2):
    T = max(e['L'] for e in E.values())
    fig = plt.figure(figsize=(10.0, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.24, left=0.115, right=0.985, top=0.9, bottom=0.08)
    lanes = gs[0].subgridspec(4, 1, hspace=0.55)
    curves = gs[1].subgridspec(1, 3, wspace=0.3)
    fig.text(0.115, 0.955, 'Final trained policies, seed %d: closed-loop simulation on the car-following dynamics' % seed,
             fontsize=12, fontweight='bold')
    clock = fig.text(0.985, 0.955, '', fontsize=12, ha='right')

    L = []
    for i, a in enumerate(ALGS):
        ax = fig.add_subplot(lanes[i])
        ax.set_facecolor(ROAD); ax.grid(False); ax.set_xlim(*VIEW); ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.005, 0.5, a, transform=ax.transAxes, ha='right', va='center', fontsize=11,
                fontweight='bold', color=COLOR[a])
        dashes = [ax.plot([], [], color='#d8d8d8', lw=1.3, solid_capstyle='butt')[0] for _ in range(10)]
        cars = []
        for j in range(5):
            p = FancyBboxPatch((0, -0.38), CAR_LEN, 0.76, boxstyle='round,pad=0,rounding_size=0.25',
                               fc='#2a78d6' if j == 3 else '#c9ccd1', ec='none', zorder=3)
            ax.add_patch(p); cars.append(p)
        labels = [ax.text(0, 0, str(j + 1), ha='center', va='center', fontsize=9, fontweight='bold',
                          color='w' if j == 3 else '#2b2b2b', zorder=4) for j in range(5)]
        gaps = [ax.text(0, 0, '', ha='center', va='center', fontsize=9.5, fontweight='bold', zorder=4) for _ in range(2)]
        state = ax.text(1.0, 1.06, '', transform=ax.transAxes, ha='right', va='bottom', fontsize=9.5)
        L.append(dict(ax=ax, dashes=dashes, cars=cars, labels=labels, gaps=gaps, state=state))

    axh, axv, axu = (fig.add_subplot(curves[k]) for k in range(3))
    allv = np.concatenate([e['v'][:, 3] for e in E.values()] + [E['TRPO-CBF']['v'][:, 2]])
    allu = np.concatenate([e['u'][np.isfinite(e['u'])] for e in E.values()])
    for ax, lab, ylim in ((axh, r'Min. headway $\min(d_f,d_r)$ (m)', (-0.6, 7.2)),
                          (axv, r'Speed of car 4 $v_4$ (m/s)', (allv.min() - 1, allv.max() + 1)),
                          (axu, r'Input of car 4 $u$ (m/s$^2$)', (allu.min() - 0.5, allu.max() + 0.5))):
        ax.set_xlim(0, 10); ax.set_ylim(*ylim); ax.set_xlabel('Time (s)'); ax.set_ylabel(lab)
    axh.axhline(2, color='k', ls=(0, (4, 3)), lw=1.0); axh.axhline(3, color=REF, ls=':', lw=0.8)
    axh.axhline(0, color=REF, lw=0.6); axu.axhline(0, color=REF, lw=0.6)
    ref, = axv.plot([], [], color=REF, lw=1.6, alpha=0.55, label='car 3')
    lh = {a: axh.plot([], [], color=COLOR[a], ls=STYLE[a], lw=1.5, label=a)[0] for a in ALGS}
    lv = {a: axv.plot([], [], color=COLOR[a], ls=STYLE[a], lw=1.5)[0] for a in ALGS}
    lu = {a: axu.plot([], [], color=COLOR[a], ls=STYLE[a], lw=1.1)[0] for a in ALGS}
    mk = {a: axh.plot([], [], 'x', color=COLOR[a], ms=8, mew=2)[0] for a in ALGS}
    h = [lh[a] for a in ALGS] + [ref]
    fig.legend(h, ALGS + ['Car 3 (ahead)'], loc='lower center', ncol=5, bbox_to_anchor=(0.55, 0.415), fontsize=10)

    def draw(k):
        for a, Q in zip(ALGS, L):
            e = E[a]; i = min(k, e['L'] - 1); p = e['p'][i]; c = p[3]
            for j in range(5):
                Q['cars'][j].set_x(p[j] - c - CAR_LEN / 2)
                Q['labels'][j].set_position((p[j] - c, 0))
            for g, (fr, bk) in zip(Q['gaps'], ((2, 3), (3, 4))):
                d = p[fr] - p[bk]
                g.set_position((0.5 * (p[fr] + p[bk]) - c, 0.72 if fr == 2 else -0.72))
                g.set_text('%.2f m' % d)
                g.set_color('#ff6b6b' if d < 2 else ('#ffbf47' if d < 3 else '#ffffff'))
            off = (-c) % 4.0
            for m, ln in enumerate(Q['dashes']):
                x0 = VIEW[0] + off + 4 * m - 4
                ln.set_data([x0, x0 + 1.5], [-0.9, -0.9])
            done = i == e['L'] - 1
            if done and bool(e['collided']):
                Q['state'].set_text('collision at t = %.2f s' % e['t'][-1]); Q['state'].set_color('#c0392b')
            else:
                Q['state'].set_text(r'$v_4$ = %.1f m/s    $u$ = %+.1f m/s$^2$    min. headway so far %.2f m'
                                    % (e['v'][i, 3], e['u'][i - 1 if done else i], np.min(e['dmin'][:i + 1])))
                Q['state'].set_color(INK)
            t = e['t'][:i + 1]
            lh[a].set_data(t, e['dmin'][:i + 1]); lv[a].set_data(t, e['v'][:i + 1, 3])
            lu[a].set_data(t[:-1] if done else t, e['u'][:i] if done else e['u'][:i + 1])
            mk[a].set_data(([e['t'][-1]], [e['dmin'][-1]]) if done and bool(e['collided']) else ([], []))
        r = E['TRPO-CBF']; i = min(k, r['L'] - 1)
        ref.set_data(r['t'][:i + 1], r['v'][:i + 1, 2])
        clock.set_text('t = %.2f s' % (min(k, T - 1) * DT))

    frames = list(range(0, T, stride)) + [T - 1] * 15
    FuncAnimation(fig, draw, frames=frames).save(out_gif, writer=PillowWriter(fps=fps), dpi=90)
    draw(T - 1)
    fig.savefig(os.path.splitext(out_gif)[0] + '_final.png', dpi=130)
    plt.close(fig)
    print('wrote', out_gif, '%.1f MB' % (os.path.getsize(out_gif) / 1e6))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--eval', default=os.path.join(CAR, 'results', 'car_eval'))
    a = ap.parse_args()
    for s in a.seeds:
        animate(load(os.path.join(a.eval, 'sim_seed%d.npz' % s)), s, os.path.join(a.eval, 'animation_seed%d.gif' % s))
