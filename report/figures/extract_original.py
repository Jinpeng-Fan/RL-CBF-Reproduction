"""Per-episode records from the AUTHORS' code runs (unmodified source, author defaults).

Input : <run root>/<ALG>/seed<k>/*.mat  (written by the author code; produced by pendulum/scripts/launch_orig.py)
Output: same folder, episodes.csv with columns episode, reward, max_angle, violation[, correction_rms]
usage : python extract_original.py [run root]

Notes on what the author logs contain (they differ from our logs):
  * DDPG / DDPG-CBF paths store the state AFTER each step (obs.append(s) after s = s2), so the initial
    state of an episode is not in the log; TRPO / TRPO-CBF logs store the state BEFORE each step.
    max_angle is therefore the max over 200 logged states; initial states of the CBF runs lie in the
    safe set, so the unsafe-episode count is unaffected.
  * TRPO logs are per iteration (15 episodes x 200 steps); they are split into 200-step episodes.
  * correction_rms = RMS of the CBF correction u_bar_ ('Action_bar') over the episode.
"""
import csv, glob, os
import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
# The author-code .mat files (~515 MB) are not part of this repository; re-run scripts/launch_orig.py
# (see README) and point ROOT at that run root, or pass it as the first argument.
import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'pendulum', 'results', 'author_code_default')


def cell(e, k):
    return np.asarray(e[k][0, 0], float)


for alg in ['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF']:
    for seed in (1, 2, 3):
        run = os.path.join(ROOT, alg, 'seed%d' % seed)
        mats = glob.glob(os.path.join(run, '*.mat'))
        assert len(mats) == 1, mats
        d = sio.loadmat(mats[0])
        rows = []
        P = d['data']
        for i in range(P.shape[1]):
            e = P[0, i]
            ob = cell(e, 'Observation').reshape(-1, 3)
            r = cell(e, 'Reward').ravel()
            bar = cell(e, 'Action_bar').ravel() if 'CBF' in alg else None
            step = 200 if alg.startswith('TRPO') else len(r)
            assert len(r) % step == 0, (alg, seed, i, len(r))
            for k in range(0, len(r), step):
                o = ob[k:k + step]
                ang = np.abs(np.arctan2(o[:, 1], o[:, 0])).max()
                row = [len(rows) + 1, r[k:k + step].sum(), ang, int(ang > 1 + 1e-6)]
                if bar is not None:
                    row.append(float(np.sqrt(np.mean(bar[k:k + step] ** 2))))
                rows.append(row)
        with open(os.path.join(run, 'episodes.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'reward', 'max_angle', 'violation'] + (['correction_rms'] if 'CBF' in alg else []))
            w.writerows(rows)
        print(alg, seed, len(rows), 'episodes')
