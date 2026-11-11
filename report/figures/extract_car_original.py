"""Per-episode records from the AUTHORS' car-following runs (unmodified source, author defaults).

Input : <RUNS>/<ALG>/seed<k>/data*.mat   (written by the author code; RUNS = --root of scripts/launch_orig.py)
        <RUNS>/logs/<ALG>_s<k>.out        (stdout of the run)
Output: ../../car/results/car_author_default/<ALG>/seed<k>/episodes.csv with the columns of the revised
        code's episodes.csv:
        episode, return, min_headway, collision, length, collisions_cum, exit_2m,
        cbf_rms, comp_rms, slack_max, infeasible_steps, cbf_cond_viol
        and ../../car/results/car_author_default/qp_errors.json (number of "Error in QP" prints per run).
usage : python extract_car_original.py --runs <RUNS> [--out <dir>]
        (the .mat files are not in the repository; the extracted csv files are)

Notes on the author logs (they differ from the revised logs):
  * every episode has 80 steps (4 s); TRPO logs are per iteration (15 episodes) and are split into
    80-step blocks. TRPO, TRPO-CBF and DDPG-CBF log the state BEFORE each step (x_0..x_79); DDPG logs
    the state AFTER each step (x_1..x_80). Both are checked below against the fixed reset state
    (all gaps 6 m, so omitting x_0 or x_80 cannot hide a collision at the start; x_80 is not logged
    by the first three). min_headway is the minimum over the 80 logged states of
    min(p3 - p4, p4 - p5) (centre-to-centre positions, as in the paper).
  * the author environment never terminates on a collision: collision = min_headway <= 0,
    exit_2m = min_headway < 2 (CBF safety distance of the paper).
  * cbf_rms / comp_rms = RMS of 'Action_bar' (CBF correction) / 'Action_BAR' (compensator).
    The QP slack is computed by the author's cvxopt QP but never returned or stored, so slack_max,
    infeasible_steps and cbf_cond_viol are left empty.  The only solver diagnostic is the
    "Error in QP" print (solution outside the +/-100 bound, then clipped); it is counted per run.
"""
import argparse, csv, glob, json, os
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True, help='root directory written by scripts/launch_orig.py')
ap.add_argument('--out', default=os.path.join(HERE, '..', '..', 'car', 'results', 'car_author_default'))
args = ap.parse_args()
ROOT, OUT = args.runs, args.out
L = 80
RESET = np.array([34, 30, 0, 28, 30, 0, 22, 30, 0, 16, 30, 0, 10, 30, 0], float)
COLS = ['episode', 'return', 'min_headway', 'collision', 'length', 'collisions_cum', 'exit_2m',
        'cbf_rms', 'comp_rms', 'slack_max', 'infeasible_steps', 'cbf_cond_viol']


def field(e, k):
    return np.asarray(e[k][0, 0], float)


def records(alg, d):
    """Yield (obs[80,15], reward[80], bar[80] or None, BAR[80] or None) per episode."""
    P = d['data']
    for i in range(P.shape[1]):
        e = P[0, i]
        names = e.dtype.names
        ob = field(e, 'Observation').reshape(-1, 15)
        r = field(e, 'Reward').ravel()
        bar = field(e, 'Action_bar').ravel() if 'Action_bar' in names else None
        BAR = field(e, 'Action_BAR').ravel() if 'Action_BAR' in names else None
        assert len(r) % L == 0 and len(ob) == len(r), (alg, i, len(r), ob.shape)
        for k in range(0, len(r), L):
            yield (ob[k:k + L], r[k:k + L], None if bar is None else bar[k:k + L],
                   None if BAR is None else BAR[k:k + L])


qp_errors = {}
for alg in ['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF']:
    for seed in (1, 2, 3):
        run = os.path.join(ROOT, alg, 'seed%d' % seed)
        mats = glob.glob(os.path.join(run, '*.mat'))
        if len(mats) != 1:
            print('skip', alg, seed, mats)
            continue
        d = sio.loadmat(mats[0])
        rows, cum = [], 0
        for n, (ob, r, bar, BAR) in enumerate(records(alg, d), 1):
            if alg == 'DDPG':          # logs the state after each step: x_1 .. x_80
                assert not np.allclose(ob[0], RESET) and np.isclose(ob[0, 9], RESET[9] + 1.5, atol=0.1), (alg, seed, n, ob[0])
            else:                      # logs the state before each step: x_0 .. x_79
                assert np.allclose(ob[0], RESET), (alg, seed, n, ob[0])
            gap = np.minimum(ob[:, 6] - ob[:, 9], ob[:, 9] - ob[:, 12]).min()
            coll = int(gap <= 0)
            cum += coll
            rms = lambda a: '' if a is None else float(np.sqrt(np.mean(a ** 2)))
            rows.append([n, float(r.sum()), float(gap), coll, L, cum, int(gap < 2.0),
                         rms(bar), rms(BAR), '', '', ''])
        dst = os.path.join(OUT, alg, 'seed%d' % seed)
        os.makedirs(dst, exist_ok=True)
        with open(os.path.join(dst, 'episodes.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            w.writerows(rows)
        out = os.path.join(ROOT, 'logs', '%s_s%d.out' % (alg, seed))
        qp_errors['%s_s%d' % (alg, seed)] = sum(l.count('Error in QP') for l in open(out, errors='replace'))
        print(alg, seed, len(rows), 'episodes, collisions', cum)
with open(os.path.join(OUT, 'qp_errors.json'), 'w') as f:
    json.dump(qp_errors, f, indent=2)
