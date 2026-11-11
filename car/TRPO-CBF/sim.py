"""Entry point.  TRPO-CBF:  python sim.py            TRPO baseline:  python sim.py --no_cbf
TensorBoard:  tensorboard --logdir <car>/Results/tb
"""
import os
import sys
import random
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from car_simulator import allCars
from learn import LEARNER
from config import *


def parse(argv=None, default_cbf=True, algo='TRPO'):
    p = argparse.ArgumentParser()
    p.add_argument('--no_cbf', action='store_true', help='plain TRPO baseline')
    p.add_argument('--name', type=str, default=None)
    # all outputs stay inside the project (car/Results/tb); learn.py handles TF's non-ASCII path issue
    p.add_argument('--logroot', type=str, default=os.path.abspath(os.path.join(HERE, '..', 'Results', 'tb')))
    # TRPO
    p.add_argument('--num_backtracking', type=int, default=20)
    p.add_argument('--kl_constraint', type=float, default=0.005)
    p.add_argument('--timesteps_per_batch', type=int, default=1000)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--lamda', type=float, default=0.95)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--max_path_length', type=int, default=EPISODE_STEPS)
    p.add_argument('--total_train_step', type=int, default=300000)
    p.add_argument('--reward_scale', type=float, default=0.01)
    # DDPG (used by DDPG-CBF/ddpg.py only)
    p.add_argument('--actor_lr', type=float, default=1e-4)
    p.add_argument('--critic_lr', type=float, default=1e-3)
    p.add_argument('--tau', type=float, default=1e-3)
    p.add_argument('--buffer_size', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--noise_sigma', type=float, default=1.0, help='OU exploration noise sigma [input units]')
    p.add_argument('--start_steps', type=int, default=1000, help='steps before the first gradient update')
    p.add_argument('--iter_episodes', type=int, default=5, help='episodes per compensator update / iteration')
    # compensator
    p.add_argument('--bar_constraint_max', type=float, default=1e1)
    p.add_argument('--comp_num_backtracking', type=int, default=30)
    # CBF / GP
    p.add_argument('--cbf_alpha', type=float, default=2.0)
    p.add_argument('--cbf_eta', type=float, default=0.2)
    p.add_argument('--cbf_kappa', type=float, default=2.0)
    p.add_argument('--gp_window', type=int, default=200)
    p.add_argument('--gp_calibration_steps', type=int, default=400)
    p.add_argument('--calibration_torque_std', type=float, default=1.0)
    # environment
    p.add_argument('--sigma_w', type=float, default=SIGMA_W)
    p.add_argument('--fixed_init', action='store_true', help='no random initial state (debug)')
    p.add_argument('--random_seed', '--random-seed', dest='random_seed', type=int, default=1)
    if algo == 'DDPG':
        p.set_defaults(gamma=0.99)
    args = p.parse_args(argv)
    args.use_cbf = not args.no_cbf and default_cbf
    args.name = args.name or (algo + ('-CBF' if args.use_cbf else ''))
    args.logdir = os.path.abspath(os.path.join(args.logroot, args.name))
    return args


def main(argv=None, default_cbf=True):
    args = parse(argv, default_cbf)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.compat.v1.set_random_seed(args.random_seed)
    # identical initial states / noise for TRPO and TRPO-CBF (separate RNGs from the policy sampler)
    # env RNG follows the run seed (seed 1 -> 1234/4321, as in the first experiment set);
    # all four algorithms of one seed see the same initial-state sequence
    env = allCars(init_seed=1233 + args.random_seed, noise_seed=4320 + args.random_seed,
                  randomize=not args.fixed_init,
                  sigma_w=args.sigma_w, episode_steps=args.max_path_length)
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=cfg) as sess:
        learner = LEARNER(args, sess, env)
        learner.learn()
        print("Learning Process Finished")


if __name__ == "__main__":
    main()
