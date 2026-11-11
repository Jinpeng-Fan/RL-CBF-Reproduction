import tensorflow as tf
import numpy as np
from utils import *


#Generalized Advantage Estimator
class GAE():
    """Value baseline; two fitting methods (vf_method).

    'trust_region' (GAE paper, Schulman et al. 2016, Sec. 5): one CG step on
        min (1/N) sum ||V(s)-Vhat||^2  s.t. (1/N) sum ||V(s)-V_old(s)||^2 / (2 sigma^2) <= vf_constraint,
        sigma^2 = (1/N) sum ||V_old(s)-Vhat||^2, using the Gauss-Newton matrix. Raw targets.
        The author code omitted the 2 sigma^2 normalization, which capped the
        change of V at ~0.1 return units per iteration.
    'adam': minibatch Adam on normalized targets (as in SB3 / Baselines).
    """
    def __init__(self, sess, input_size, gamma, lamda, vf_constraint,
                 vf_epochs=10, vf_batch=64, vf_lr=1e-3, seed=0, obs_normalizer=None,
                 vf_method='adam'):
        self.sess = sess
        self.input_size = input_size
        self.gamma = gamma
        self.lamda = lamda
        self.vf_constraint = vf_constraint
        self.vf_method = vf_method
        self.obs_normalizer = obs_normalizer  # shared input scaling; None = raw input
        self.vf_epochs, self.vf_batch, self.vf_lr = vf_epochs, vf_batch, vf_lr
        self.rng = np.random.RandomState(seed)  # Minibatch order; leaves np.random untouched.
        # V(s) = ret_mean + ret_std * network(s); stays (0, 1) for trust_region
        self.ret_mean, self.ret_std = 0.0, 1.0
        self.build_model()

    # Input will be state : [batch_size, observation size]
    # Outputs normalized state value function
    def build_model(self):
        print('Initializing Value function network')
        with tf.variable_scope('VF'):
            self.x = tf.placeholder(tf.float32, [None, self.input_size], name='State')
            scaled = self.obs_normalizer(self.x) if self.obs_normalizer else self.x

            #Target will be the normalized discounted return
            self.target = tf.placeholder(tf.float32, [None,1], name='Target')

            #Model is MLP composed of 3 hidden layers with 100, 50, 25 tanh units
            h1 = LINEAR(scaled, 100, name='h1')
            h1_n1 = tf.tanh(h1)
            h2 = LINEAR(h1_n1, 50, name='h2')
            h2_n1 = tf.tanh(h2)
            h3 = LINEAR(h2_n1, 25, name='h3')
            h3_n1 = tf.tanh(h3)
            self.value = LINEAR(h3_n1, 1, name='FC')

        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VF')
        for i in tr_vrbs:
            print(i.op.name)

        self.loss = tf.reduce_mean(tf.pow(self.target - self.value, 2))
        with tf.variable_scope('VF_optimizer'):
            self.fit_op = tf.train.AdamOptimizer(self.vf_lr).minimize(self.loss, var_list=tr_vrbs)

        # Trust-region fit: gradient of the loss and Gauss-Newton vector product.
        # At V_old = current predictions, the Hessian of 0.5*mean((V-V_old)^2) is J^T J / N.
        self.grad_objective = FLAT_GRAD(self.loss, tr_vrbs)
        self.value_old = tf.placeholder(tf.float32, [None, 1], name='Value_old')
        distance = 0.5 * tf.reduce_mean(tf.square(self.value - self.value_old))
        self.y = tf.placeholder(tf.float32, [None])
        self.GNVP = HESSIAN_VECTOR_PRODUCT(distance, tr_vrbs, self.y)

        #To adjust weights and biases
        self.get_value = GetValue(self.sess, tr_vrbs, name='VF')
        self.set_value = SetValue(self.sess, tr_vrbs, name='VF')

        self.sess.run(tf.global_variables_initializer())

    def predict(self, observation):
        normalized = self.sess.run(self.value, feed_dict={self.x: observation})
        return np.reshape(normalized, (-1,)) * self.ret_std + self.ret_mean

    def get_advantage(self, paths):
        """GAE with time-limit bootstrapping.

        Pendulum has no terminal state: every episode ends because of the
        200-step limit. Treating that cut as terminal (V=0) makes the return
        target depend on the remaining steps, which V cannot observe, so the
        baseline cannot be fitted. The tail is therefore bootstrapped with
        V(s_T) of the last next-observation, both in the advantage and in the
        value target.
        """
        #Get observation, make it [batch_size, observation_size]
        self.observation = np.squeeze(np.concatenate([path["Observation"] for path in paths]))
        self.rewards = np.concatenate([path["Reward"] for path in paths])
        self.done = np.concatenate([path["Done"] for path in paths])
        batch_s = self.observation.shape[0]

        #Compute value function for all timesteps using current parameter
        self.value_s = self.predict(self.observation)
        last_obs = np.vstack([np.reshape(path["NextObservation"][-1], (1, -1)) for path in paths])
        value_last = self.predict(last_obs)

        boundaries = np.cumsum([len(path["Reward"]) for path in paths])
        starts = np.r_[0, boundaries[:-1]]
        advantages, returns = [], []
        for (a, b), v_last, path in zip(zip(starts, boundaries), value_last, paths):
            rewards, values = np.squeeze(self.rewards[a:b]).reshape(-1), self.value_s[a:b]
            #V(s_{t+1}); the last one is the bootstrapped tail value
            value_next = np.r_[values[1:], v_last]
            #delt_t_V : reward_t + gamma*Value(state_{t+1}) - Value(state_t)
            delta = rewards + self.gamma * value_next - values
            advantages.append(DISCOUNT_SUM(delta, self.gamma*self.lamda))
            #Value target: discounted reward sum plus gamma^(T-t) * V(s_T)
            tail = self.gamma ** np.arange(len(rewards), 0, -1) * v_last
            path["return_sum"] = DISCOUNT_SUM(rewards, self.gamma) + tail
            returns.append(path["return_sum"])
        self.delta_v = None
        self.return_sum = np.reshape(np.concatenate(returns), [batch_s, 1])
        GAE = np.concatenate(advantages)
        self.advantage_raw = GAE.copy()

        #Normalize to make mean 0
        GAE = (GAE - np.mean(GAE)) / (np.std(GAE) + 1e-6)
        return GAE

    def train(self):
        """Refit V on this batch's returns (warm start), after the policy step."""
        returns = np.reshape(self.return_sum, (-1,))
        variance = np.var(returns) + 1e-8
        # V used for this batch's advantages, before refitting
        error_before = returns - self.value_s
        if self.vf_method == 'trust_region':
            backtracks = self.fit_trust_region(returns)
        else:
            backtracks = self.fit_adam(returns)
        error_after = returns - self.predict(self.observation)
        self.last_value_update = dict(
            accepted=bool(np.mean(error_after ** 2) <= np.mean(error_before ** 2)),
            method=self.vf_method, backtracks=backtracks,
            loss_before=float(np.mean(error_before ** 2)), loss_after=float(np.mean(error_after ** 2)),
            status='fitted',
            explained_variance_before=float(1 - np.var(error_before) / variance),
            explained_variance_after=float(1 - np.var(error_after) / variance))
        return self.last_value_update

    def fit_trust_region(self, returns):
        """One CG step, GAE paper Eq. (30)-(31); raw (unnormalized) targets."""
        target = returns.reshape(-1, 1)
        value_old = self.predict(self.observation).reshape(-1, 1)
        sigma2 = float(np.mean((value_old - target) ** 2)) + 1e-8
        feed_dict = {self.x: self.observation, self.target: target, self.value_old: value_old}
        parameter_prev = self.get_value()
        gradient = self.sess.run(self.grad_objective, feed_dict=feed_dict)

        def gauss_newton_vector_product(y):
            feed_dict[self.y] = y
            return self.sess.run(self.GNVP, feed_dict=feed_dict) + 1e-6 * y

        def loss(parameter):
            self.set_value(parameter)
            return self.sess.run(self.loss, feed_dict=feed_dict)

        direction = CONJUGATE_GRADIENT(gauss_newton_vector_product, -gradient)
        curvature = 0.5 * float(direction.dot(gauss_newton_vector_product(direction)))
        if not (np.isfinite(curvature) and curvature > 0):
            loss(parameter_prev)
            return 0
        # 0.5 d^T G d <= 2 sigma^2 * vf_constraint  (paper's constraint, normalized by 2 sigma^2)
        full_step = np.sqrt(2 * sigma2 * self.vf_constraint / curvature) * direction
        new_parameter = LINE_SEARCH(loss, parameter_prev, full_step, name='Value loss')
        self.set_value(new_parameter, update_info=0)
        return 1

    def fit_adam(self, returns):
        self.ret_mean, self.ret_std = float(np.mean(returns)), float(np.std(returns) + 1e-8)
        target = ((returns - self.ret_mean) / self.ret_std).reshape(-1, 1)
        n = len(target)
        for _ in range(self.vf_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, self.vf_batch):
                idx = order[start:start+self.vf_batch]
                self.sess.run(self.fit_op, feed_dict={self.x: self.observation[idx],
                                                      self.target: target[idx]})
        return self.vf_epochs * int(np.ceil(n / self.vf_batch))
