import numpy as np
import tensorflow as tf
from utils import *
from config import *
from gae import GAE
from barrier_comp import BARRIER
from obs_scaler import FixedObservationScaler
from car_simulator import allCars
import cbf
import dynamics_gp

tf1 = tf.compat.v1


class TRPO():
    """TRPO (use_cbf=False) or TRPO-CBF (use_cbf=True): RL + CBF + GP + compensating network.

    Per decision (revised):  raw ~ N(mu, sigma)   (sampled ONCE, before the filter)
                             candidate = raw + u_BAR(s)           (compensating network)
                             executed  = candidate + u_bar        (CBF correction on the sample)
    The policy gradient uses log pi(raw | s), i.e. pi_prop(a|s) = pi_RL(a - u_barrier | s).
    """
    def __init__(self, args, env, sess):
        self.args = args
        self.sess = sess
        self.env = env
        self.use_cbf = bool(args.use_cbf)
        self.firstIter = True          # iteration k=0 deploys no compensator
        self.on_episode = None         # callback(path) used for live logging
        self.torque_bound = U_MAX
        if args.timesteps_per_batch < args.max_path_length:
            raise ValueError('timesteps_per_batch must cover at least one full episode.')
        self.observation_size = 15
        self.action_size = 1
        self.build_model()
        if self.use_cbf:
            cbf.build_barrier(self)
            dynamics_gp.build_GP_model(self)
            self.calibrate_GP()

    # ---------------------------------------------------------------- networks
    def build_model(self):
        self.obs = tf1.placeholder(tf.float32, [None, self.observation_size])
        self.obs_norm = FixedObservationScaler(self.observation_size)
        self.action = tf1.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf1.placeholder(tf.float32, [None])
        self.old_action_dist_mu = tf1.placeholder(tf.float32, [None, self.action_size])
        self.old_action_dist_logstd = tf1.placeholder(tf.float32, [None, self.action_size])

        self.action_dist_mu, action_dist_logstd = self.build_policy(self.obs)
        tr_vrbs = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
        self.action_dist_logstd = tf.tile(action_dist_logstd, (tf.shape(self.obs)[0], 1))

        self.log_policy = LOG_POLICY(self.action_dist_mu, self.action_dist_logstd, self.action)
        self.log_old_policy = LOG_POLICY(self.old_action_dist_mu, self.old_action_dist_logstd, self.action)
        policy_ratio = tf.exp(self.log_policy - self.log_old_policy)
        batch_size = tf.cast(tf.shape(self.obs)[0], tf.float32)
        surr_single_state = -tf.reduce_mean(policy_ratio * self.advantage)
        kl = GAUSS_KL(self.old_action_dist_mu, self.old_action_dist_logstd,
                      self.action_dist_mu, self.action_dist_logstd) / batch_size
        ent = GAUSS_ENTROPY(self.action_dist_mu, self.action_dist_logstd) / batch_size
        self.losses = [surr_single_state, kl, ent]
        self.pg = FLAT_GRAD(surr_single_state, tr_vrbs)

        kl_first_fixed = GAUSS_KL_FIRST_FIX(self.action_dist_mu, self.action_dist_logstd) / batch_size
        first_kl_grads = tf1.gradients(kl_first_fixed, tr_vrbs)
        self.flat_tangent = tf1.placeholder(tf.float32, [None])
        tangent, start = list(), 0
        for vrbs in tr_vrbs:
            variable_size = int(np.prod(vrbs.get_shape().as_list()))
            tangent.append(tf.reshape(self.flat_tangent[start:(start + variable_size)], vrbs.get_shape()))
            start += variable_size
        gradient_w_tangent = [tf.reduce_sum(kl_g * t) for (kl_g, t) in zip(first_kl_grads, tangent)]
        self.FVP = FLAT_GRAD(gradient_w_tangent, tr_vrbs)
        self.get_value = GetValue(self.sess, tr_vrbs, name='Policy')
        self.set_value = SetValue(self.sess, tr_vrbs, name='Policy')

        self.gae = GAE(self.sess, self.observation_size, self.args.gamma, self.args.lamda,
                       obs_normalizer=self.obs_norm, reward_scale=self.args.reward_scale)
        self.bar_comp = BARRIER(self.args, self.sess, self.observation_size, self.action_size,
                                obs_normalizer=self.obs_norm)
        self.sess.run(tf1.global_variables_initializer())

    def build_policy(self, states, name='Policy'):
        with tf1.variable_scope(name, reuse=tf1.AUTO_REUSE):
            h1 = tf.nn.relu(LINEAR(self.obs_norm(states), self.args.hidden_size, name='h1'))
            h2 = tf.nn.relu(LINEAR(h1, self.args.hidden_size, name='h2'))
            h3 = LINEAR(h2, self.action_size, name='h3')
            # logstd=0 -> initial exploration std of 1 (trainable)
            logstd = tf1.get_variable('logstd', initializer=tf1.constant_initializer(0.), shape=[1, self.action_size])
        return h3, logstd

    def act(self, obs):
        obs_expanded = np.reshape(np.asarray(obs, dtype=np.float32), (1, -1))
        mu, logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs: obs_expanded})
        return np.random.normal(loc=mu, scale=np.exp(logstd)), mu, logstd

    # ---------------------------------------------------------------- training
    def train(self):
        batch_path = self.rollout()
        theta_prev = self.get_value()
        advantage_estimated = self.gae.get_advantage(batch_path)

        mu = np.concatenate([p["Action_mu"] for p in batch_path]).reshape(-1, 1)
        logstd = np.concatenate([p["Action_logstd"] for p in batch_path]).reshape(-1, 1)
        observation = np.concatenate([p["Observation"] for p in batch_path]).reshape(-1, 15)
        action = np.concatenate([p["Action_RL"] for p in batch_path]).reshape(-1, 1)
        feed_dict = {self.obs: observation, self.action: action, self.advantage: advantage_estimated,
                     self.old_action_dist_mu: mu, self.old_action_dist_logstd: logstd}
        policy_g = self.sess.run(self.pg, feed_dict=feed_dict)

        def fisher_vector_product(gradient):
            feed_dict[self.flat_tangent] = gradient
            return self.sess.run(self.FVP, feed_dict=feed_dict)

        def evaluate_policy(theta):
            self.set_value(theta)
            loss, kl = self.sess.run(self.losses[:2], feed_dict=feed_dict)
            return float(loss), float(kl)

        new_theta, info = TRUST_REGION_STEP(evaluate_policy, theta_prev, policy_g, fisher_vector_product,
                                            self.args.kl_constraint, self.args.num_backtracking)
        self.set_value(new_theta)
        value_info = self.gae.train()

        surrogate_after, kl_after, ent = self.sess.run(self.losses, feed_dict=feed_dict)
        logs = {"Surrogate loss": surrogate_after, "KL_DIV": kl_after, "Entropy": ent,
                "Policy_accepted": float(info['accepted']), "Policy_backtracks": info['backtracks'],
                "Value_loss_after": value_info['loss_after'],
                "Value_explained_variance": value_info['explained_variance_after']}
        logs["Total Step"] = sum(len(p["Reward"]) for p in batch_path)
        logs["Num episode"] = len(batch_path)
        logs["Total Sum"] = sum(float(np.sum(p["Reward"])) for p in batch_path)
        logs["Episode Avg. Reward"] = logs["Total Sum"] / logs["Num episode"]
        logs["Collisions"] = sum(int(p["Collision"]) for p in batch_path)
        logs["Policy_logstd"] = float(np.mean(logstd))

        if self.use_cbf:
            # compensator is trained AFTER the policy step on the batch just collected (Alg. 1)
            self.bar_comp.get_training_rollouts(batch_path)
            self.bar_comp.train()
            self.firstIter = False
            fit = self.bar_comp.last_fit
            logs["Comp_loss_before"], logs["Comp_loss_after"] = fit['loss_before'], fit['loss_after']
            logs["Comp_accepted_steps"] = fit['accepted_steps']
            bar = np.concatenate([p["Action_bar"] for p in batch_path]).reshape(-1)
            BAR = np.concatenate([p["Action_BAR"] for p in batch_path]).reshape(-1)
            logs["CBF_correction_abs_mean"] = float(np.mean(np.abs(bar)))
            logs["Compensator_abs_mean"] = float(np.mean(np.abs(BAR)))
            logs["CBF_slack_max"] = max(float(np.max(p['CBF_slack'])) for p in batch_path)
            logs["CBF_infeasible_steps"] = int(sum(np.sum(p['CBF_slack'] > 0) for p in batch_path))
            logs["CBF_condition_violations"] = int(sum(np.sum(p['CBF_violated']) for p in batch_path))
            logs["GP_samples"] = len(self.gp_Z)
            logs["GP_coverage"] = float(np.mean(np.concatenate([p['GP_covered'] for p in batch_path])))
            logs["GP_error_rms"] = float(np.sqrt(np.mean(np.concatenate([p['GP_error_sq'] for p in batch_path]))))
        return logs

    # ---------------------------------------------------------------- rollouts
    def rollout(self):
        paths, timesteps = [], 0
        while timesteps < self.args.timesteps_per_batch:
            paths.append(self.rollout_episode())
            timesteps += len(paths[-1]['Reward'])
        return paths

    def rollout_episode(self, env=None, learn_gp=True):
        env = env or self.env
        L = self.args.max_path_length
        keys = ['obs', 'next', 'exe', 'raw', 'mu', 'logstd', 'bar', 'comp', 'rew', 'slack',
                'violated', 'covered', 'err_sq', 'minhead']
        rec = {k: [] for k in keys}
        state = env.reset()
        terminated = truncated = False
        for i in range(L):
            before = np.asarray(state).copy()
            raw, mean, logstd = self.act(before)
            raw, mean, logstd = float(raw.item()), float(mean.item()), float(logstd.item())
            raw = float(np.clip(raw, -U_MAX, U_MAX))     # the stored/optimised action is the clipped one
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
                # does the realised next state satisfy the CBF condition (no robust margin)?
                h_next = self.P.dot(after) + self.q
                violated = bool(np.any(h_next < (1 - self.barrier_eta) * self.last_cbf['h_now'] - 1e-9))
            for k, v in zip(keys, [before, after.copy(), executed, raw, mean, logstd, bar, comp,
                                   reward, slack, violated, covered, err_sq, info['min_headway']]):
                rec[k].append(v)
            state = after
            if terminated or truncated:
                break
        path = dict(Observation=np.asarray(rec['obs']), NextObservation=np.asarray(rec['next']),
                    Action=np.asarray(rec['exe']).reshape(-1, 1), Action_RL=np.asarray(rec['raw']).reshape(-1, 1),
                    Action_mu=np.asarray(rec['mu']).reshape(-1, 1), Action_logstd=np.asarray(rec['logstd']).reshape(-1, 1),
                    Action_bar=np.asarray(rec['bar']).reshape(-1, 1), Action_BAR=np.asarray(rec['comp']).reshape(-1, 1),
                    Reward=np.asarray(rec['rew']), Terminated=bool(terminated), Collision=bool(terminated),
                    CBF_slack=np.asarray(rec['slack']), CBF_violated=np.asarray(rec['violated']),
                    GP_covered=np.asarray(rec['covered']), GP_error_sq=np.asarray(rec['err_sq']),
                    MinHeadway=float(np.min(rec['minhead'])), Return=float(np.sum(rec['rew'])),
                    Length=len(rec['rew']))
        if self.use_cbf and learn_gp:
            dynamics_gp.update_GP_dynamics(self, path)    # refit after every episode (Alg. 1)
        if self.on_episode is not None and learn_gp:
            self.on_episode(path)
        return path

    def calibrate_GP(self):
        """Pre-fit the GP on a nominal-CBF rollout with random torques so that iteration 1 does not
        start from the (large) prior std. Uses a separate environment instance."""
        steps = getattr(self.args, 'gp_calibration_steps', 0)
        if steps <= 0:
            return
        env = allCars(init_seed=self.args.random_seed + 1000, noise_seed=self.args.random_seed + 2000,
                      sigma_w=self.args.sigma_w, episode_steps=self.args.max_path_length)
        rng = np.random.RandomState(self.args.random_seed + 3000)
        done_steps, n_ep = 0, 0
        while done_steps < steps:
            state, rec = env.reset(), dict(o=[], n=[], u=[])
            for _ in range(self.args.max_path_length):
                before = state.copy()
                cand = rng.normal(scale=self.args.calibration_torque_std)
                f, g, x, sigma = dynamics_gp.get_GP_dynamics(self, before, env.t)
                bar = float(cbf.control_barrier(self, before, cand, f, g, x, sigma)[0])
                u = float(np.clip(cand + bar, -U_MAX, U_MAX))
                after, _, term, trunc, _ = env.step(u)
                rec['o'].append(before); rec['n'].append(after.copy()); rec['u'].append(u)
                state = after
                done_steps += 1
                if term or trunc:
                    break
            dynamics_gp.update_GP_dynamics(self, dict(Observation=np.asarray(rec['o']),
                                                      NextObservation=np.asarray(rec['n']),
                                                      Action=np.asarray(rec['u'])))
            n_ep += 1
        print('GP calibration: %d steps in %d episodes, %d samples' % (done_steps, n_ep, len(self.gp_Z)))
