from fair_environment import configure_comparison_env
import tensorflow as tf
import numpy as np
from trpo import TRPO
from scipy.io import savemat
import datetime
import gym

class LEARNER():
    def __init__(self, args, sess, simulator):
        self.args = args
        self.sess = sess
        self.simulator = simulator

        #Construct simulation environment
        self.simulator = configure_comparison_env(gym.make('Pendulum-v0'), getattr(args, 'random_seed', 1))
        self.simulator.unwrapped.max_torque = 15.0
        self.simulator.unwrapped.max_speed = 60.0
        self.simulator.unwrapped.action_space = gym.spaces.Box(low=-15., high=15., shape=(1,))
        high = np.array([1., 1., 60.])
        self.simulator.unwrapped.observation_space = gym.spaces.Box(low=-high, high=high)
        self.simulator.action_space = self.simulator.unwrapped.action_space
        self.simulator.observation_space = self.simulator.unwrapped.observation_space

        seed = getattr(args, 'random_seed', 1)
        self.simulator.seed(seed)
        self.simulator.action_space.seed(seed)
        self.simulator.observation_space.seed(seed)

        
        #Define learning agent (TRPO)
        self.agent = TRPO(self.args, self.simulator, self.sess)

    def learn(self):
        train_index = 0
        total_episode = 0
        total_steps = 0
        all_logs = list()
        while True:
            #Train the TRPO agent
            train_index += 1
            train_log = self.agent.train()
            total_steps += train_log["Total Step"]
            total_episode += train_log["Num episode"]

            all_logs.append(train_log)
            
            # Evaluation uses a separate run; do not consume training initial states.

            if total_steps >= self.args.total_train_step:
                #nn_weights = {'policy_network': self.agent.get_value(), 'advantage_network': self.agent.gae.get_value()}
                savemat(self.args.name + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=all_logs, args=self.args))
                #savemat('weights_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(policy_weights=nn_weights, args=self.args))
                break
