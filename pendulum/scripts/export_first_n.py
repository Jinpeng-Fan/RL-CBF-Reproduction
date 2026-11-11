"""Export the first N episodes of the final runs of the four algorithms.

usage: export_first_n.py <runs_root> <out_dir> <tb_dir (ASCII path)> [N=600]

Per algorithm / seed, <out_dir>/<ALG>/seed<k>/ gets
  episodes.csv     per-episode metrics of episodes 1..N (from the 'LIVE' log lines)
  train_log.mat    TRPO / TRPO-CBF only: the native per-iteration logs covering episodes 1..N
  run_config.json  exact command line and source hashes
  source/          source snapshot that produced the run
and <tb_dir> gets TensorBoard event files with the same N episodes.
"""
import csv, json, shutil, sys, time, glob
from pathlib import Path
import numpy as np
import scipy.io as sio

root, out, tbdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 600
RUNS = {'DDPG': 'DDPG-e5', 'DDPG-CBF': 'DDPG-CBF-e5', 'TRPO': 'TRPO-fixed-e5', 'TRPO-CBF': 'TRPO-CBF-fixed-e5'}
if out.exists() or tbdir.exists():
    sys.exit('refusing to overwrite existing %s / %s' % (out, tbdir))

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2, summary_pb2

summary = {}
for alg, label in RUNS.items():
    for seed in (1, 2, 3):
        rows = [json.loads(l[5:]) for l in open(str(root / 'logs' / ('%s_s%d.out' % (label, seed))), errors='replace')
                if l.startswith('LIVE ')]
        rows = [r for r in rows if r['episode'] <= N]
        assert [r['episode'] for r in rows] == list(range(1, N + 1)), (alg, seed, len(rows))
        dst = out / alg / ('seed%d' % seed)
        dst.mkdir(parents=True)
        keys = ['episode', 'reward', 'max_angle'] + (['slack', 'correction_rms'] if alg.endswith('CBF') else [])
        with open(str(dst / 'episodes.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(keys + ['violation'])
            for r in rows:
                w.writerow([r[k] for k in keys] + [int(r['max_angle'] > 1.0 + 1e-6)])
        src = root / label / ('seed%d' % seed)
        shutil.copy2(str(src / 'run_config.json'), str(dst / 'run_config.json'))
        shutil.copytree(str(src / 'source'), str(dst / 'source'))
        if alg.startswith('TRPO'):
            mats = glob.glob(str(src / '*.mat'))
            assert len(mats) == 1, mats
            d = sio.loadmat(mats[0])
            logs = d['data']                                    # (1, iterations) struct/cell array
            # each entry is a 1x1 struct whose fields are 1x1 object cells
            per_iter = [int(np.squeeze(logs[0, i]['Num episode'][0, 0])) for i in range(logs.shape[1])]
            cum = np.cumsum(per_iter)
            k = int(np.searchsorted(cum, N) + 1)                # iterations covering episodes 1..N
            assert cum[k - 1] == N, (alg, seed, cum[:k])
            sio.savemat(str(dst / 'train_log.mat'), dict(data=logs[:, :k], args=d['args']),
                        long_field_names=True, do_compression=True)
        # TensorBoard
        tb = tbdir / ('seed%d' % seed) / alg
        tb.mkdir(parents=True)
        wr = EventFileWriter(str(tb), max_queue_size=1000, flush_secs=1)
        cum_v = 0
        for r in rows:
            v = float(r['max_angle'] > 1.0 + 1e-6); cum_v += v
            vals = [('reward/episode_return', r['reward']), ('safety/max_abs_angle_rad', r['max_angle']),
                    ('safety/violation', v), ('safety/violations_cumulative', cum_v)]
            if 'slack' in r:
                vals += [('control/cbf_max_slack', r['slack']), ('control/cbf_correction_rms', r['correction_rms'])]
            wr.add_event(event_pb2.Event(wall_time=time.time(), step=r['episode'], summary=summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag=t, simple_value=float(x)) for t, x in vals])))
        wr.close()
        R = np.array([r['reward'] for r in rows]); A = np.array([r['max_angle'] for r in rows])
        summary['%s seed%d' % (alg, seed)] = dict(last100_mean_return=float(R[-100:].mean()),
                                                  violations=int((A > 1 + 1e-6).sum()))
(out / 'summary.json').write_text(json.dumps(dict(episodes=N, runs=RUNS, results=summary), indent=2), encoding='utf-8')
for k, v in summary.items():
    print('%-14s last100 mean %8.1f  violations %4d' % (k, v['last100_mean_return'], v['violations']))
