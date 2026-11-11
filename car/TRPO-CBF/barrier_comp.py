import numpy as np
import tensorflow as tf
from utils import *
from config import U_MAX

tf1 = tf.compat.v1
OUT_SCALE = 10.0   # network output is multiplied by this (targets are up to O(10-100))


#Barrier Function Compensator: learns the accumulated CBF correction u_bar + u_BAR
class BARRIER():
    def __init__(self, args, sess, input_size, action_size, obs_normalizer):
        self.sess = sess
        self.args = args
        self.input_size = input_size
        self.action_size = action_size
        self.obs_normalizer = obs_normalizer  # shared fixed input scaling
        self.last_fit = dict(accepted_steps=0, loss_before=0., loss_after=0., stop_reason='untrained')
        self.build_model()

    def build_model(self):
        print('Initializing Barrier Compensation network')
        with tf1.variable_scope('Compensator'):
            self.x = tf1.placeholder(tf.float32, [None, self.input_size], name='Obs')
            self.target = tf1.placeholder(tf.float32, [None, self.action_size], name='Target_comp')
            h1 = tf.nn.relu(LINEAR(self.obs_normalizer(self.x), 50, name='h1'))
            h2 = tf.nn.relu(LINEAR(h1, 40, name='h2'))
            self.value = OUT_SCALE * LINEAR(h2, self.action_size, name='h3', bias_std=0.)

        tr_vrbs = tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES, scope='Compensator')
        self.loss = tf.reduce_mean(tf.pow(self.target - self.value, 2))
        self.grad_objective = FLAT_GRAD(self.loss, tr_vrbs)
        self.get_value = GetValue(self.sess, tr_vrbs, name='Compensator')
        self.set_value = SetValue(self.sess, tr_vrbs, name='Compensator')
        # variables are initialised once by TRPO.build_model (global initializer)

    def get_training_rollouts(self, paths):
        self.observation = np.concatenate([p["Observation"] for p in paths]).reshape(-1, self.input_size)
        bar = np.concatenate([p["Action_bar"] for p in paths]).reshape(-1, self.action_size)
        BAR = np.concatenate([p["Action_BAR"] for p in paths]).reshape(-1, self.action_size)
        # target = total barrier action u_barrier,k = (previous compensator) + (this CBF correction)
        self.target_value = bar + BAR

    #Given current observation, get the neural network output (barrier compensator)
    def get_action(self, obs):
        observation = np.reshape(np.asarray(obs, dtype=np.float32), (1, -1))
        u = self.sess.run(self.value, {self.x: observation})
        return np.clip(u, -U_MAX, U_MAX)

    def train(self):
        """Pendulum-style fit: up to 100 steepest-descent steps with backtracking line search."""
        self.last_fit = dict(iterations=0, accepted_steps=0, stop_reason='iteration_limit')
        feed_dict = {self.x: self.observation, self.target: self.target_value}

        def loss(parameter):
            self.set_value(parameter)
            return self.sess.run(self.loss, feed_dict=feed_dict)

        new_parameter = self.get_value()
        self.last_fit['loss_before'] = float(loss(new_parameter))
        for i in range(100):
            self.last_fit['iterations'] = i + 1
            parameter_prev = new_parameter
            self.set_value(parameter_prev)
            gradient = self.sess.run(self.grad_objective, feed_dict=feed_dict)
            full_step = -self.args.bar_constraint_max * gradient
            new_parameter = LINE_SEARCH(loss, parameter_prev, full_step,
                                        num_backtracking=getattr(self.args, 'comp_num_backtracking', 30),
                                        name='Barrier loss')
            self.set_value(new_parameter)
            if np.array_equal(new_parameter, parameter_prev):
                self.last_fit['stop_reason'] = 'no_accepted_step'
                break
            self.last_fit['accepted_steps'] += 1
        self.last_fit['loss_after'] = float(loss(new_parameter))
        return self.last_fit['loss_after']
