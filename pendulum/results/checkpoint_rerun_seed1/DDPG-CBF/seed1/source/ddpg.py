from fair_environment import configure_comparison_env
""" 
Implementation of DDPG-CBF on the Pendulum-v0 OpenAI gym task

"""
import tensorflow as tf
import numpy as np
import gym
import json
import tflearn
import argparse
import pprint as pp
from scipy.io import savemat
import random

from replay_buffer import ReplayBuffer
from obs_norm import ObsNormalizer

from learner import LEARNER
from barrier_comp import BARRIER
import cbf
import dynamics_gp
import datetime
from gym import spaces

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, obs_norm=None):
        self.sess = sess
        self.obs_norm = obs_norm  # shared running observation normalizer; None = raw input
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(self.obs_norm(inputs) if self.obs_norm else inputs, 400)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, obs_norm=None):
        self.sess = sess
        self.obs_norm = obs_norm  # shared running observation normalizer; None = raw input
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(self.obs_norm(inputs) if self.obs_norm else inputs, 400)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class GaussianActionNoise:
    """Independent N(mu, sigma^2) noise at every step (as in TD3). Unlike the OU process
    above (mean-reversion time ~667 steps > 200-step episode), it adds no slowly drifting
    bias torque within an episode."""
    def __init__(self, mu, sigma=1.0):
        self.mu, self.sigma = mu, sigma

    def __call__(self):
        return self.mu + self.sigma * np.random.normal(size=self.mu.shape)

    def reset(self):
        pass

    def __repr__(self):
        return 'GaussianActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise, reward_result, agent):
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    actor.update_target_network()
    critic.update_target_network()
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    paths = []
    batch_paths = []
    # One outer iteration is now one episode, as in Algorithm 1.
    for i in range(int(args['max_episodes'])):
        obs, next_obs, actions, rewards = [], [], [], []
        action_bar, action_BAR, slacks = [], [], []
        s = env.reset()
        while abs(env.unwrapped.state[0]) > 1.0:
            s = env.reset()
        actor_noise.reset()
        for j in range(int(args['max_episode_len'])):
            before = np.asarray(s).copy()
            raw = actor.predict(before.reshape(1, actor.s_dim))[0] + actor_noise()
            raw = np.clip(raw, -actor.action_bound, actor.action_bound)  # stored action = proposed action
            compensation = (np.zeros(actor.a_dim) if agent.firstIter else
                            agent.bar_comp.get_action(before)[0])
            candidate = raw + compensation
            f, g, x, Sigma = dynamics_gp.get_GP_dynamics(agent, before)
            correction = cbf.control_barrier(agent, before, candidate, f, g, x, Sigma)
            executed = candidate + correction
            s2, r, terminal, info = env.step(executed)
            end_episode = terminal or j + 1 == int(args['max_episode_len'])
            # Only a true environment termination stops bootstrapping; the 200-step
            # TimeLimit is a truncation (Pendulum has no terminal state).
            env_done = bool(terminal) and not info.get('TimeLimit.truncated', False)
            # The critic's action is the raw policy command into the filtered system.
            replay_buffer.add(before.reshape(actor.s_dim), raw.reshape(actor.a_dim),
                              r, env_done, np.asarray(s2).reshape(actor.s_dim))
            obs.append(before)
            next_obs.append(np.asarray(s2).copy())
            actions.append(executed.copy())
            rewards.append(r)
            action_bar.append(correction.copy())
            action_BAR.append(compensation.copy())
            slacks.append(agent.last_cbf['slack'])
            if replay_buffer.size() > int(args['minibatch_size']):
                sb, ab, rb, tb, s2b = replay_buffer.sample_batch(int(args['minibatch_size']))
                target_q = critic.predict_target(s2b, actor.predict_target(s2b)).reshape(-1)
                targets = rb + critic.gamma * (1 - tb.astype(float)) * target_q
                critic.train(sb, ab, targets.reshape(-1, 1))
                grads = critic.action_gradients(sb, actor.predict(sb))
                actor.train(sb, grads[0])
                actor.update_target_network()
                critic.update_target_network()
            s = s2
            if end_episode:
                break
        path = dict(Observation=np.asarray(obs), NextObservation=np.asarray(next_obs),
                    Action=np.asarray(actions), Reward=np.asarray(rewards),
                    Action_bar=np.asarray(action_bar), Action_BAR=np.asarray(action_BAR),
                    CBF_slack=np.asarray(slacks))
        paths.append(path)  # Archive only; never feed stale compensation targets.
        dynamics_gp.update_GP_dynamics(agent, path)
        batch_paths.append(path)
        # One policy iteration = episodes_per_iteration episodes: the compensator
        # is refit on that batch only, matching TRPO-CBF's iteration.
        if len(batch_paths) == int(args['episodes_per_iteration']):
            # Refresh the shared normalizer BEFORE fitting the compensator, so the
            # compensator is fitted under the same input scaling it will be used with.
            actor.obs_norm.update(np.concatenate([p['Observation'] for p in batch_paths]))
            agent.bar_comp.get_training_rollouts(batch_paths)
            agent.bar_comp.train()
            agent.firstIter = 0
            batch_paths = []
        reward_result[i] = np.sum(rewards)
        print('Episode {} | Reward {:.3f} | GP samples {} | Max slack {:.3g}'.format(
            i, reward_result[i], len(agent.gp_X), max(slacks)))
        _o = np.concatenate([np.asarray(obs), np.asarray(next_obs)])
        print('LIVE ' + json.dumps(dict(episode=i + 1, reward=float(reward_result[i]),
              max_angle=float(np.max(np.abs(np.arctan2(_o[:, 1], _o[:, 0])))),
              slack=float(max(slacks)), correction_rms=float(np.sqrt(np.mean(np.square(action_bar)))))), flush=True)
    return summary_ops, summary_vars, paths


def main(args, reward_result):

    with tf.Session() as sess:

        env = configure_comparison_env(gym.make(args['env']), int(args['random_seed']))
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        # Set environment parameters for pendulum
        env.unwrapped.max_torque = 15.
        env.unwrapped.max_speed = 60.   
        env.unwrapped.action_space = spaces.Box(low=-env.unwrapped.max_torque, high=env.unwrapped.max_torque, shape=(1,))
        high = np.array([1., 1., env.unwrapped.max_speed])
        env.unwrapped.observation_space = spaces.Box(low=-high, high=high)
        
        # Gym 0.17 wrappers retain their own spaces; expose the revised bounds.
        env.action_space = env.unwrapped.action_space
        env.observation_space = env.unwrapped.observation_space
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        # Shared running observation normalizer for actor, critic and compensator
        obs_norm = ObsNormalizer(sess, state_dim)
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']), obs_norm=obs_norm)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars(), obs_norm=obs_norm)
        
        # sigma is in torque units (action range is +-15), so it is larger than the 0.3 default.
        actor_noise = GaussianActionNoise(mu=np.zeros(action_dim), sigma=float(args['noise_sigma']))

        agent = LEARNER(env)
        # LEARNER() already built the barrier and the GP model.
        agent.enforce_speed_limit = bool(args.get('enforce_speed_limit', False))
        # Same compensator settings as TRPO-CBF (50/40 net, step 10, 100 iters, 30 backtracks).
        agent.bar_comp = BARRIER(argparse.Namespace(bar_constraint_max=float(args['bar_constraint_max']),
                                                    comp_num_backtracking=int(args['comp_num_backtracking'])),
                                 sess, 3, 1, obs_normalizer=obs_norm)

        
        [summary_ops, summary_vars, paths] = train(sess, env, args, actor, critic, actor_noise, reward_result, agent)

        
        return [summary_ops, summary_vars, paths]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    
    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.0005)  #DDPG paper 1e-3; lowered for stability
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=100000)  #500 episodes; DDPG paper 1e6
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=256)  #DDPG paper 64

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1)  #Same for all four algorithms
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=500)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200) 
    parser.add_argument('--render-env', help='render the gym env', action='store_false')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_false')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results2/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results2/tf_ddpg')

    parser.add_argument('--bar_constraint_max', type=float, default=1e1)  #Same as TRPO-CBF
    parser.add_argument('--comp_num_backtracking', type=int, default=30)  #Same as TRPO-CBF
    parser.add_argument('--noise-sigma', type=float, default=1.0, help='OU noise sigma in torque units')
    parser.add_argument('--episodes-per-iteration', type=int, default=5)  #Same as TRPO (1000 steps / 200)
    parser.add_argument('--enforce-speed-limit', action='store_true')
    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    reward_result = np.zeros(int(args['max_episodes']))
    [summary_ops, summary_vars, paths] = main(args, reward_result)

    savemat('data4_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=paths, reward=reward_result))

