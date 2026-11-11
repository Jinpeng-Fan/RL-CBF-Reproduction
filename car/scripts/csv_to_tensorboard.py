"""Write TensorBoard event files from the per-episode records in car/results/.

usage: python csv_to_tensorboard.py [--results ../results] [--out ../../tensorboard]
Needs only the `tensorboard` package (no TensorFlow). Run names: <group>/seed<k>/<ALG>, e.g.
  car_final/seed1/TRPO-CBF, car_author_default/seed3/DDPG-CBF.
Scalars (x axis = training episode; car_author_default episodes have 80 steps, car_final at most 200):
  return/episode_return, return/episode_return_ma10   episode return and 10-episode trailing mean
  safety/min_headway_m                                minimum gap of car 4 to its neighbours
  safety/collision, safety/collisions_cum             gap <= 0 in the episode, cumulative count
  safety/below_2m, safety/below_2m_cum                gap < 2 m (safety distance), cumulative count
  progress/env_steps                                  cumulative environment steps
  cbf/correction_rms, cbf/compensator_rms             RMS of u_CBF and of the compensator (CBF variants)
  cbf/max_slack                                       max QP slack (car_final CBF variants; not logged by the author code)
"""
import argparse, csv, os, time
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto import event_pb2, summary_pb2

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--results', default=os.path.join(HERE, '..', 'results'))
ap.add_argument('--out', default=os.path.join(HERE, '..', '..', 'tensorboard'))
a = ap.parse_args()
GROUPS = ['car_final', 'car_author_default']
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
            coll = exits = steps = 0
            window = []
            for r in rows:
                ret, gap = float(r['return']), float(r['min_headway'])
                c, e = int(gap <= 0), int(gap < 2.0)
                coll += c; exits += e; steps += int(r['length'])
                window = (window + [ret])[-10:]
                vals = [scalar('return/episode_return', ret),
                        scalar('return/episode_return_ma10', sum(window) / len(window)),
                        scalar('safety/min_headway_m', gap),
                        scalar('safety/collision', c), scalar('safety/collisions_cum', coll),
                        scalar('safety/below_2m', e), scalar('safety/below_2m_cum', exits),
                        scalar('progress/env_steps', steps)]
                if alg.endswith('CBF'):
                    for k, tag in (('cbf_rms', 'cbf/correction_rms'), ('comp_rms', 'cbf/compensator_rms'),
                                   ('slack_max', 'cbf/max_slack')):
                        if r.get(k, '') != '':
                            vals.append(scalar(tag, r[k]))
                w.add_event(event_pb2.Event(wall_time=time.time(), step=int(r['episode']),
                                            summary=summary_pb2.Summary(value=vals)))
            w.close()
            n_runs += 1
            print('%-20s seed%d %-9s %6d episodes' % (g, seed, alg, len(rows)))
print(n_runs, 'runs written to', os.path.abspath(a.out))
