import csv
import os
import time
import numpy as np
import tensorflow as tf
from trpo import TRPO
from config import D_MIN

tf1 = tf.compat.v1


def ascii_alias(path):
    """TensorFlow's file IO fails on non-ASCII paths (this project sits under a Chinese folder
    name). Return an ASCII directory junction that points to `path`; the data stays physically
    inside the project."""
    os.makedirs(path, exist_ok=True)
    try:
        path.encode('ascii')
        return path
    except UnicodeEncodeError:
        pass
    import hashlib, subprocess
    alias = os.path.join(os.path.expanduser('~'), '.rlcbf_alias_' + hashlib.md5(path.encode('utf-8')).hexdigest()[:8])
    if not os.path.exists(alias):
        subprocess.check_call(['cmd', '/c', 'mklink', '/J', alias, path], stdout=subprocess.DEVNULL)
    return alias


class TBWriter():
    def __init__(self, logdir):
        self.w = tf1.summary.FileWriter(ascii_alias(logdir))

    def scalar(self, tag, value, step):
        self.w.add_summary(tf1.Summary(value=[tf1.Summary.Value(tag=tag, simple_value=float(value))]), step)

    def flush(self):
        self.w.flush()


class LEARNER():
    """Training loop + live TensorBoard logging.

    Per episode (x-axis = episode):  safety/min_headway, safety/collision, safety/collisions_cumulative,
                                     safety/safe_set_exits_cumulative, reward/episode_return, ...
    Per iteration (x-axis = iteration): optimisation / CBF / GP / compensator diagnostics.
    """
    def __init__(self, args, sess, env):
        self.args = args
        self.agent = TRPO(args, env, sess)
        self._setup_logging(args)

    def _setup_logging(self, args):
        """TensorBoard + csv bookkeeping shared with the DDPG learner."""
        self.tb = TBWriter(args.logdir)
        self.episode = 0
        self.collisions = 0
        self.exits = 0
        self.csv_path = os.path.join(args.logdir, 'episodes.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['episode', 'return', 'min_headway', 'collision', 'length', 'collisions_cum',
                           'exit_2m', 'cbf_rms', 'comp_rms', 'slack_max', 'infeasible_steps', 'cbf_cond_viol'])
        self.agent.on_episode = self.log_episode
        self.start = time.time()

    def log_episode(self, path):
        self.episode += 1
        e = self.episode
        self.agent.last_min_headway = path['MinHeadway']
        coll = int(path['Collision'])
        exit_safe = int(path['MinHeadway'] < D_MIN)
        self.collisions += coll
        self.exits += exit_safe
        self.tb.scalar('safety/min_headway', path['MinHeadway'], e)
        self.tb.scalar('safety/collision', coll, e)
        self.tb.scalar('safety/collisions_cumulative', self.collisions, e)
        self.tb.scalar('safety/safe_set_exit', exit_safe, e)
        self.tb.scalar('safety/safe_set_exits_cumulative', self.exits, e)
        self.tb.scalar('safety/threshold_safe_2m', D_MIN, e)
        self.tb.scalar('safety/threshold_collision_0m', 0.0, e)
        self.tb.scalar('reward/episode_return', path['Return'], e)
        self.tb.scalar('reward/episode_length', path['Length'], e)
        # logging only (no RNG use): per-episode CBF statistics; zeros for the runs without CBF
        rms = lambda k: float(np.sqrt(np.mean(np.square(path[k])))) if k in path and len(path[k]) else 0.0
        slack = np.asarray(path.get('CBF_slack', [0.0]), float)
        slack_max, infeas = float(np.max(slack)) if slack.size else 0.0, int(np.sum(slack > 0))
        viol = int(np.sum(path.get('CBF_violated', [])))
        cbf_rms, comp_rms = rms('Action_bar'), rms('Action_BAR')
        self.tb.scalar('cbf/correction_rms', cbf_rms, e)
        self.tb.scalar('cbf/compensator_rms', comp_rms, e)
        self.tb.scalar('cbf/slack_max', slack_max, e)
        self.csv.writerow([e, path['Return'], path['MinHeadway'], coll, path['Length'], self.collisions,
                           exit_safe, cbf_rms, comp_rms, slack_max, infeas, viol])
        self.csv_file.flush()
        self.tb.flush()

    def learn(self):
        it, total_steps = 0, 0
        while total_steps < self.args.total_train_step:
            it += 1
            logs = self.agent.train()
            total_steps += logs["Total Step"]
            for k, v in logs.items():
                self.tb.scalar('train/' + k.replace(' ', '_'), v, it)
            self.tb.scalar('train/total_steps', total_steps, it)
            self.tb.flush()
            print('[%s] iter %d  steps %d  ep %d  avg_return %.1f  collisions(batch) %d  min_headway(last) %.2f  %.0fs' % (
                self.args.name, it, total_steps, self.episode, logs['Episode Avg. Reward'], logs['Collisions'],
                self.agent.last_min_headway if hasattr(self.agent, 'last_min_headway') else float('nan'),
                time.time() - self.start), flush=True)
            if self.args.use_cbf:
                print('      CBF: |u_bar| %.2f  |u_BAR| %.2f  infeasible %d  cond.viol %d | GP: n=%d coverage %.2f rms %.3f | comp loss %.2f->%.2f | KL %.4f acc %d' % (
                    logs['CBF_correction_abs_mean'], logs['Compensator_abs_mean'], logs['CBF_infeasible_steps'],
                    logs['CBF_condition_violations'], logs['GP_samples'], logs['GP_coverage'], logs['GP_error_rms'],
                    logs['Comp_loss_before'], logs['Comp_loss_after'], logs['KL_DIV'], logs['Policy_accepted']), flush=True)
            if it % 25 == 0:
                tf1.train.Saver().save(self.agent.sess, os.path.join(ascii_alias(self.args.logdir), 'model'), global_step=it)
        self.csv_file.close()
