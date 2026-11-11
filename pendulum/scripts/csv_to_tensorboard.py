"""Write TensorBoard event files from the per-episode records in pendulum/results/.

usage: python csv_to_tensorboard.py [--results ../results] [--out ../../tensorboard]
Needs only the `tensorboard` package (no TensorFlow). Run names: <group>/seed<k>/<ALG>, e.g.
  final_600ep/seed1/TRPO-CBF, author_code_default/seed3/DDPG-CBF.
Scalars (x axis = training episode):
  return/episode_return          episode return
  return/episode_return_ma10     10-episode trailing moving average (as in the report figures)
  safety/max_abs_angle_rad       max |theta| over the episode
  safety/unsafe_episode          1 if max |theta| > 1 rad
  safety/unsafe_episodes_cum     cumulative count of unsafe episodes
  cbf/correction_rms             RMS of the CBF correction u_CBF (CBF variants)
  cbf/max_slack                  max slack epsilon in the episode (modified CBF variants)
"""
import argparse, csv, os, time
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2, summary_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(HERE, '..', 'results'))
ap.add_argument('--out', default=os.path.join(HERE, '..', '..', 'tensorboard'))
a = ap.parse_args()
# checkpoint_rerun_seed1 (re-run of seed 1 for the first/last-policy figure) is deliberately not exported.
GROUPS = ['final_600ep', 'author_code_default']
ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']


def scalar(tag, v):
    return summary_pb2.Summary.Value(tag=tag, simple_value=float(v))


n_runs = 0
for g in GROUPS:
    for alg in ALGS:
        for seed in (1, 2, 3):
            f = os.path.join(a.results, g, alg, 'seed%d' % seed, 'episodes.csv')
            if not os.path.exists(f):
                continue
            rows = list(csv.DictReader(open(f)))
            run = os.path.join(a.out, g, 'seed%d' % seed, alg)
            os.makedirs(run, exist_ok=True)
            w = EventFileWriter(run, max_queue_size=1000, flush_secs=5)
            cum, window = 0, []
            for r in rows:
                ret, ang = float(r['reward']), float(r['max_angle'])
                unsafe = int(ang > 1 + 1e-6)
                cum += unsafe
                window = (window + [ret])[-10:]
                vals = [scalar('return/episode_return', ret),
                        scalar('return/episode_return_ma10', sum(window) / len(window)),
                        scalar('safety/max_abs_angle_rad', ang),
                        scalar('safety/unsafe_episode', unsafe),
                        scalar('safety/unsafe_episodes_cum', cum)]
                if 'correction_rms' in r:
                    vals.append(scalar('cbf/correction_rms', r['correction_rms']))
                if 'slack' in r:
                    vals.append(scalar('cbf/max_slack', r['slack']))
                w.add_event(event_pb2.Event(wall_time=time.time(), step=int(r['episode']),
                                            summary=summary_pb2.Summary(value=vals)))
            w.close()
            n_runs += 1
            print('%-24s seed%d %-9s %5d episodes' % (g, seed, alg, len(rows)))
print(n_runs, 'runs written to', os.path.abspath(a.out))
