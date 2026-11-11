import tensorflow as tf
import numpy as np
import argparse
import random
from learn import LEARNER

#Suppress CPU warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_backtracking', type=int, default=20)
    parser.add_argument('--kl_constraint', type=float, default=0.005)  #Originally 0.01
    parser.add_argument('--timesteps_per_batch', type=int, default=1000)  #5 episodes per policy iteration (aligned with DDPG)
    parser.add_argument('--init_logstd', type=float, default=1.0986)  #log(3): initial action std 3 N*m (range +-15)
    parser.add_argument('--gamma', type=float, default=0.99)  #Same as DDPG
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--bar_constraint_max', type=float, default=1e1)
    parser.add_argument('--vf_constraint', type=float, default=1e-2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--max_path_length', type=int, default=200)
    parser.add_argument('--total_train_step', type=int, default=100000)  #500 episodes = 50 iterations
    parser.add_argument('--enforce_speed_limit', action='store_true')
    parser.add_argument('--comp_num_backtracking', type=int, default=30)
    parser.add_argument('--gp_calibration_steps', type=int, default=0)  #>0 enables GP pre-fit (off by default)
    parser.add_argument('--calibration_torque_std', type=float, default=1.0)
    parser.add_argument('--vf_method', type=str, default='adam', choices=['trust_region', 'adam'])
    parser.add_argument('--random-seed', '--random_seed', dest='random_seed', type=int, default=1)  #Same for all four algorithms
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    
    #Set up tensorflow to use GPU
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True

    #Run tensorflow session
    with tf.Session(config=config) as sess:
        trpo_gae_learner = LEARNER(args,sess,args)
        trpo_gae_learner.learn()
        print("Learning Process Finished")

if __name__ == "__main__":
    main()
