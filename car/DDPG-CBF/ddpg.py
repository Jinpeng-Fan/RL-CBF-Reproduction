"""DDPG-CBF (use_cbf) and plain DDPG for the car-following task, revised.

Shares the environment, GP, CBF, compensating network, observation scaling and TensorBoard
logging with ../TRPO-CBF.  Differences from the original DDPG code:
  * no tflearn / batch-norm (BN in training mode gave a state-independent action for batch size 1);
  * fixed physical input scaling, tau=1e-3, buffer 1e5, exploration noise sized to the input range;
  * the replay buffer stores the RAW actor action (before compensator and CBF), so the actor
    learns from its own output; the CBF/compensator act as part of the environment;
  * collision = true terminal (no bootstrap), time limit = truncation (bootstrapped);
  * the compensating network is ENABLED (it was commented out) and trained once per iteration;
  * the GP is refit after every episode, with a calibration pre-fit.

  DDPG-CBF:  python ddpg.py        plain DDPG:  python ddpg.py --no_cbf   (or ../DDPG/ddpg.py)
"""
import os
import sys
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..', 'TRPO-CBF'))
sys.path.insert(0, SHARED)          # shared modules take priority over the old files in this folder

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf1 = tf.compat.v1
import sim as sim_module
from car_simulator import allCars
from config import *
from utils import LINEAR
from obs_scaler import FixedObservationScaler
from barrier_comp import BARRIER
from learn import LEARNER, ascii_alias
from trpo import TRPO
import cbf
import dynamics_gp

A_RL = 100.0  # range of the actor output (tanh), same as the paper's input bound


class OUNoise():
    def __init__(self, sigma, theta=0.15):
        self.sigma, self.theta, self.x = sigma, theta, 0.0

    def reset(self):
        self.x = 0.0

    def __call__(self):
        self.x += self.theta * (0.0 - self.x) + self.sigma * np.random.randn()
        return self.x


class ReplayBuffer():
    def __init__(self, size, seed):
        self.size, self.n, self.i = size, 0, 0
        self.s = np.zeros((size, 15), np.float32); self.s2 = np.zeros((size, 15), np.float32)
        self.a = np.zeros((size, 1), np.float32); self.r = np.zeros((size, 1), np.float32)
        self.term = np.zeros((size, 1), np.float32)
        self.rng = np.random.RandomState(seed)

    def add(self, s, a, r, term, s2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.term[i], self.s2[i] = s, a, r, float(term), s2
        self.i = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, k):
        idx = self.rng.randint(0, self.n, size=k)
        return self.s[idx], self.a[idx], self.r[idx], self.term[idx], self.s2[idx]


class DDPG():
    calibrate_GP = TRPO.calibrate_GP        # same GP pre-fit as TRPO-CBF

    def __init__(self, args, env, sess):
        self.args, self.sess, self.env = args, sess, env
        self.use_cbf = bool(args.use_cbf)
        self.firstIter = True
        self.on_episode = None
        self.observation_size, self.action_size = 15, 1
        self.buffer = ReplayBuffer(args.buffer_size, args.random_seed)
        self.total_updates = 0
        self.stats = dict(critic_loss=[], q=[])
        self.build_model()
        if self.use_cbf:
            cbf.build_barrier(self)
            dynamics_gp.build_GP_model(self)
            self.calibrate_GP()

    # ------------------------------------------------------------------ networks
    def actor(self, s, scope):
        with tf1.variable_scope(scope, reuse=tf1.AUTO_REUSE):
            h1 = tf.nn.relu(LINEAR(self.obs_norm(s), 128, name='h1', weight_std=0.2, bias_std=0.))
            h2 = tf.nn.relu(LINEAR(h1, 128, name='h2', weight_std=0.1, bias_std=0.))
            return A_RL * tf.tanh(LINEAR(h2, 1, name='out', weight_std=3e-3, bias_std=0.))

    def critic(self, s, a, scope):
        with tf1.variable_scope(scope, reuse=tf1.AUTO_REUSE):
            x = tf.concat([self.obs_norm(s), a / A_RL], axis=1)
            h1 = tf.nn.relu(LINEAR(x, 128, name='h1', weight_std=0.2, bias_std=0.))
            h2 = tf.nn.relu(LINEAR(h1, 128, name='h2', weight_std=0.1, bias_std=0.))
            return LINEAR(h2, 1, name='out', weight_std=3e-3, bias_std=0.)

    def build_model(self):
        a = self.args
        self.obs_norm = FixedObservationScaler(15)
        self.s = tf1.placeholder(tf.float32, [None, 15])
        self.a = tf1.placeholder(tf.float32, [None, 1])
        self.y = tf1.placeholder(tf.float32, [None, 1])     # critic target
        self.s2 = tf1.placeholder(tf.float32, [None, 15])
        self.mu = self.actor(self.s, 'Actor')
        q_sa = self.critic(self.s, self.a, 'Critic')
        q_pi = self.critic(self.s, self.mu, 'Critic')
        mu_t = self.actor(self.s2, 'ActorT')
        self.q_target_next = self.critic(self.s2, mu_t, 'CriticT')
        self.critic_loss = tf.reduce_mean(tf.square(self.y - q_sa))
        self.actor_loss = -tf.reduce_mean(q_pi)
        var = lambda scope: tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/')
        with tf1.variable_scope('DDPG_opt'):
            self.critic_op = tf1.train.AdamOptimizer(a.critic_lr).minimize(self.critic_loss, var_list=var('Critic'))
            self.actor_op = tf1.train.AdamOptimizer(a.actor_lr).minimize(self.actor_loss, var_list=var('Actor'))
        pairs = list(zip(var('Actor'), var('ActorT'))) + list(zip(var('Critic'), var('CriticT')))
        self.soft_update = tf.group(*[t.assign(a.tau * o + (1 - a.tau) * t) for o, t in pairs])
        self.hard_update = tf.group(*[t.assign(o) for o, t in pairs])
        self.q_pi = q_pi
        self.bar_comp = BARRIER(a, self.sess, 15, 1, obs_normalizer=self.obs_norm)
        self.sess.run(tf1.global_variables_initializer())
        self.sess.run(self.hard_update)

    def update(self):
        a = self.args
        s, act, r, term, s2 = self.buffer.sample(a.batch_size)
        q2 = self.sess.run(self.q_target_next, {self.s2: s2})
        y = r * a.reward_scale + a.gamma * (1.0 - term) * q2      # collision: no bootstrap
        cl, _ = self.sess.run([self.critic_loss, self.critic_op], {self.s: s, self.a: act, self.y: y})
        qm, _, _ = self.sess.run([self.q_pi, self.actor_op, self.soft_update], {self.s: s})
        self.stats['critic_loss'].append(float(cl)); self.stats['q'].append(float(np.mean(qm)))
        self.total_updates += 1

    # ------------------------------------------------------------------ episode
    def run_episode(self, noise, env=None):
        env = env or self.env
        L = self.args.max_path_length
        keys = ['obs', 'next', 'exe', 'raw', 'bar', 'comp', 'rew', 'slack', 'violated', 'covered', 'err_sq', 'minhead']
        rec = {k: [] for k in keys}
        state = env.reset()
        noise.reset()
        terminated = truncated = False
        for i in range(L):
            before = np.asarray(state).copy()
            mu = float(self.sess.run(self.mu, {self.s: before.reshape(1, -1)}).item())
            raw = float(np.clip(mu + noise(), -U_MAX, U_MAX))          # sampled ONCE, before the filter
            comp, bar, slack, violated, covered, err_sq = 0.0, 0.0, 0.0, False, 1.0, 0.0
            if self.use_cbf:
                if not self.firstIter:
                    comp = float(self.bar_comp.get_action(before).item())
                candidate = raw + comp
                f, g, x, sigma = dynamics_gp.get_GP_dynamics(self, before, env.t)
                bar = float(cbf.control_barrier(self, before, candidate, f, g, x, sigma)[0])
                executed = candidate + bar
                slack = self.last_cbf['slack']
            else:
                executed = raw
            executed = float(np.clip(executed, -U_MAX, U_MAX))
            after, reward, terminated, truncated, info = env.step(executed)
            if self.use_cbf:
                pred = f + g * executed
                err = (after - pred)[GP_IDX]
                covered = float(np.mean(np.abs(err) <= self.barrier_kd * sigma[GP_IDX] + 1e-9))
                err_sq = float(np.mean(err ** 2))
                h_next = self.P.dot(after) + self.q
                violated = bool(np.any(h_next < (1 - self.barrier_eta) * self.last_cbf['h_now'] - 1e-9))
            # replay buffer: RAW actor action; only a collision is a true terminal
            self.buffer.add(before, raw, reward, terminated, after)
            if self.buffer.n >= self.args.start_steps:
                self.update()
            for k, v in zip(keys, [before, after.copy(), executed, raw, bar, comp, reward, slack, violated,
                                   covered, err_sq, info['min_headway']]):
                rec[k].append(v)
            state = after
            if terminated or truncated:
                break
        path = dict(Observation=np.asarray(rec['obs']), NextObservation=np.asarray(rec['next']),
                    Action=np.asarray(rec['exe']).reshape(-1, 1), Action_RL=np.asarray(rec['raw']).reshape(-1, 1),
                    Action_bar=np.asarray(rec['bar']).reshape(-1, 1), Action_BAR=np.asarray(rec['comp']).reshape(-1, 1),
                    Reward=np.asarray(rec['rew']), Terminated=bool(terminated), Collision=bool(terminated),
                    CBF_slack=np.asarray(rec['slack']), CBF_violated=np.asarray(rec['violated']),
                    GP_covered=np.asarray(rec['covered']), GP_error_sq=np.asarray(rec['err_sq']),
                    MinHeadway=float(np.min(rec['minhead'])), Return=float(np.sum(rec['rew'])),
                    Length=len(rec['rew']))
        if self.use_cbf:
            dynamics_gp.update_GP_dynamics(self, path)
        if self.on_episode is not None:
            self.on_episode(path)
        return path

    def end_iteration(self, paths):
        """Train the compensator on this iteration's episodes (after the actor/critic updates)."""
        logs = {"Episode Avg. Reward": float(np.mean([p['Return'] for p in paths])),
                "Collisions": sum(int(p['Collision']) for p in paths),
                "Total Step": sum(p['Length'] for p in paths),
                "Num episode": len(paths),
                "Critic_loss": float(np.mean(self.stats['critic_loss'])) if self.stats['critic_loss'] else 0.0,
                "Q_mean": float(np.mean(self.stats['q'])) if self.stats['q'] else 0.0,
                "Updates_total": self.total_updates}
        self.stats = dict(critic_loss=[], q=[])
        if self.use_cbf:
            self.bar_comp.get_training_rollouts(paths)
            self.bar_comp.train()
            self.firstIter = False
            fit = self.bar_comp.last_fit
            logs["Comp_loss_before"], logs["Comp_loss_after"] = fit['loss_before'], fit['loss_after']
            logs["Comp_accepted_steps"] = fit['accepted_steps']
            bar = np.concatenate([p["Action_bar"] for p in paths]).reshape(-1)
            BAR = np.concatenate([p["Action_BAR"] for p in paths]).reshape(-1)
            logs["CBF_correction_abs_mean"] = float(np.mean(np.abs(bar)))
            logs["Compensator_abs_mean"] = float(np.mean(np.abs(BAR)))
            logs["CBF_slack_max"] = max(float(np.max(p['CBF_slack'])) for p in paths)
            logs["CBF_infeasible_steps"] = int(sum(np.sum(p['CBF_slack'] > 0) for p in paths))
            logs["CBF_condition_violations"] = int(sum(np.sum(p['CBF_violated']) for p in paths))
            logs["GP_samples"] = len(self.gp_Z)
            logs["GP_coverage"] = float(np.mean(np.concatenate([p['GP_covered'] for p in paths])))
            logs["GP_error_rms"] = float(np.sqrt(np.mean(np.concatenate([p['GP_error_sq'] for p in paths]))))
        return logs


class DDPGLearner(LEARNER):
    def __init__(self, args, sess, env):
        self.args = args
        self.agent = DDPG(args, env, sess)
        self._setup_logging(args)
        self.agent.on_episode = self.log_episode
        self.noise = OUNoise(args.noise_sigma)

    def learn(self):
        it, total_steps = 0, 0
        while total_steps < self.args.total_train_step:
            it += 1
            paths = [self.agent.run_episode(self.noise) for _ in range(self.args.iter_episodes)]
            logs = self.agent.end_iteration(paths)
            total_steps += logs["Total Step"]
            for k, v in logs.items():
                self.tb.scalar('train/' + k.replace(' ', '_'), v, it)
            self.tb.scalar('train/total_steps', total_steps, it)
            self.tb.flush()
            print('[%s] iter %d  steps %d  ep %d  avg_return %.1f  collisions(batch) %d  min_headway(last) %.2f  critic %.4f  Q %.2f  %.0fs' % (
                self.args.name, it, total_steps, self.episode, logs['Episode Avg. Reward'], logs['Collisions'],
                self.agent.last_min_headway, logs['Critic_loss'], logs['Q_mean'], time.time() - self.start), flush=True)
            if self.args.use_cbf:
                print('      CBF: |u_bar| %.2f  |u_BAR| %.2f  infeasible %d  cond.viol %d | GP: n=%d coverage %.2f | comp loss %.2f->%.2f' % (
                    logs['CBF_correction_abs_mean'], logs['Compensator_abs_mean'], logs['CBF_infeasible_steps'],
                    logs['CBF_condition_violations'], logs['GP_samples'], logs['GP_coverage'],
                    logs['Comp_loss_before'], logs['Comp_loss_after']), flush=True)
            if it % 25 == 0:
                tf1.train.Saver().save(self.agent.sess, os.path.join(ascii_alias(self.args.logdir), 'model'), global_step=it)
        self.csv_file.close()


def main(argv=None, default_cbf=True):
    args = sim_module.parse(argv, default_cbf, algo='DDPG')
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf1.set_random_seed(args.random_seed)
    env = allCars(init_seed=1233 + args.random_seed, noise_seed=4320 + args.random_seed,
                  randomize=not args.fixed_init,
                  sigma_w=args.sigma_w, episode_steps=args.max_path_length)
    cfg = tf1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf1.Session(config=cfg) as sess:
        DDPGLearner(args, sess, env).learn()
        print("Learning Process Finished")


if __name__ == "__main__":
    main(sys.argv[1:])
