import numpy as np
import tensorflow as tf
from utils import *

tf1 = tf.compat.v1


#Generalized Advantage Estimator + value baseline
class GAE():
    """Value baseline, fitted with minibatch Adam on normalised returns.

    Differences from the original car code:
      * advantages / returns are computed PER EPISODE (no leakage across episodes);
      * a collision is a true terminal state (V(s')=0); hitting the time limit is only a
        truncation, bootstrapped with V(s_T);
      * rewards are multiplied by `reward_scale` for learning only (env rewards stay raw).
    """
    def __init__(self, sess, input_size, gamma, lamda, obs_normalizer, reward_scale=0.01,
                 vf_epochs=10, vf_batch=64, vf_lr=1e-3, seed=0):
        self.sess = sess
        self.input_size = input_size
        self.gamma = gamma
        self.lamda = lamda
        self.reward_scale = reward_scale
        self.obs_normalizer = obs_normalizer
        self.vf_epochs, self.vf_batch, self.vf_lr = vf_epochs, vf_batch, vf_lr
        self.rng = np.random.RandomState(seed)
        self.ret_mean, self.ret_std = 0.0, 1.0   # V(s) = ret_mean + ret_std * network(s)
        self.last_value_update = dict(loss_before=0., loss_after=0., explained_variance_after=0.)
        self.build_model()

    def build_model(self):
        print('Initializing Value function network')
        with tf1.variable_scope('VF'):
            self.x = tf1.placeholder(tf.float32, [None, self.input_size], name='State')
            scaled = self.obs_normalizer(self.x)
            self.target = tf1.placeholder(tf.float32, [None, 1], name='Target')
            h1 = tf.tanh(LINEAR(scaled, 100, name='h1'))
            h2 = tf.tanh(LINEAR(h1, 50, name='h2'))
            h3 = tf.tanh(LINEAR(h2, 25, name='h3'))
            self.value = LINEAR(h3, 1, name='FC')
        tr_vrbs = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='VF')
        self.loss = tf.reduce_mean(tf.pow(self.target - self.value, 2))
        with tf1.variable_scope('VF_optimizer'):
            self.fit_op = tf1.train.AdamOptimizer(self.vf_lr).minimize(self.loss, var_list=tr_vrbs)
        # variables are initialised once by TRPO.build_model (global initializer)

    def predict(self, observation):
        normalized = self.sess.run(self.value, feed_dict={self.x: observation})
        return np.reshape(normalized, (-1,)) * self.ret_std + self.ret_mean

    def get_advantage(self, paths):
        self.observation = np.concatenate([p["Observation"] for p in paths]).reshape(-1, self.input_size)
        self.value_s = self.predict(self.observation)
        last_obs = np.vstack([np.reshape(p["NextObservation"][-1], (1, -1)) for p in paths])
        value_last = self.predict(last_obs)

        advantages, returns, start = [], [], 0
        for p, v_last in zip(paths, value_last):
            n = len(p["Reward"])
            rewards = np.asarray(p["Reward"], float).reshape(-1) * self.reward_scale
            values = self.value_s[start:start + n]
            start += n
            if p["Terminated"]:
                v_last = 0.0                       # collision: true terminal state
            value_next = np.r_[values[1:], v_last]
            delta = rewards + self.gamma * value_next - values
            advantages.append(DISCOUNT_SUM(delta, self.gamma * self.lamda))
            tail = self.gamma ** np.arange(n, 0, -1) * v_last
            returns.append(DISCOUNT_SUM(rewards, self.gamma) + tail)
        self.return_sum = np.concatenate(returns).reshape(-1, 1)
        adv = np.concatenate(advantages)
        return (adv - np.mean(adv)) / (np.std(adv) + 1e-6)

    def train(self):
        returns = self.return_sum.reshape(-1)
        variance = np.var(returns) + 1e-8
        error_before = returns - self.value_s
        self.ret_mean, self.ret_std = float(np.mean(returns)), float(np.std(returns) + 1e-8)
        target = ((returns - self.ret_mean) / self.ret_std).reshape(-1, 1)
        n = len(target)
        for _ in range(self.vf_epochs):
            order = self.rng.permutation(n)
            for s in range(0, n, self.vf_batch):
                idx = order[s:s + self.vf_batch]
                self.sess.run(self.fit_op, feed_dict={self.x: self.observation[idx], self.target: target[idx]})
        error_after = returns - self.predict(self.observation)
        self.last_value_update = dict(
            loss_before=float(np.mean(error_before ** 2)), loss_after=float(np.mean(error_after ** 2)),
            explained_variance_after=float(1 - np.var(error_after) / variance))
        return self.last_value_update
