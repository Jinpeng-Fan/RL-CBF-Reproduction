"""Simulate the final trained policies of all four algorithms on the car-following dynamics.

usage (from car/, python37 runtime: Python 3.7 + TensorFlow 1.15):
    python scripts/simulate_final.py [--seeds 1 2 3] [--out results/car_eval]

Policies : the last saved checkpoint of each training run, results/car_final/<ALG>/seed<k>/checkpoint/
           (TRPO baselines end before iteration 300 because collisions shorten their episodes).
Controller: full deployed controller of each algorithm, deterministic action
           TRPO / TRPO-CBF : policy mean;  DDPG / DDPG-CBF : actor output (no exploration noise);
           CBF variants    : RL action + compensating network + CBF filter.
GP (CBF variants): not part of the checkpoint; rebuilt as in training (400-step calibration with
           the training seed) and refitted on 5 deterministic warm-up episodes of the loaded policy
           on a separate environment (init_seed = seed+500+w), then frozen during the simulation.
Scenario : one fixed initial state from the training distribution, allCars(init_seed=2026,
           noise_seed=3026): the same initial state and acceleration-noise sequence for every
           algorithm and seed; 200 steps = 10 s (the training episode length).
Output   : <out>/sim_seed<k>.npz   per algorithm: t, p, v (5 cars), d_f, d_r, u_rl, u_comp, u_cbf, u, slack
           <out>/sim_summary.json  per seed and algorithm: checkpoint, collision, min headway, return, ...
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse, importlib.util, json, random, sys

CAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CAR, 'TRPO-CBF'))
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf1 = tf.compat.v1
import sim as sim_module
from car_simulator import allCars
from trpo import TRPO
from learn import ascii_alias
from config import *
import cbf
import dynamics_gp

_spec = importlib.util.spec_from_file_location('ddpg_shared', os.path.join(CAR, 'DDPG-CBF', 'ddpg.py'))
ddpg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ddpg_mod)

ALGS = ['TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG']
INIT_SEED, NOISE_SEED, WARMUP = 2026, 3026, 5


def build_and_restore(alg, seed, sess, ckpt_dir):
    use_cbf = alg.endswith('CBF')
    argv = ['--random_seed', str(seed)] + ([] if use_cbf else ['--no_cbf'])
    random.seed(seed); np.random.seed(seed); tf1.set_random_seed(seed)     # as in training main()
    env = allCars(init_seed=1233 + seed, noise_seed=4320 + seed)
    if alg.startswith('TRPO'):
        agent = TRPO(sim_module.parse(argv, use_cbf), env, sess)        # CBF: barrier + GP calibration
        policy = lambda o: float(sess.run(agent.action_dist_mu, {agent.obs: o.reshape(1, -1)}).item())
    else:
        agent = ddpg_mod.DDPG(sim_module.parse(argv, use_cbf, algo='DDPG'), env, sess)
        policy = lambda o: float(sess.run(agent.mu, {agent.s: o.reshape(1, -1)}).item())
    d = ascii_alias(ckpt_dir)
    its = [int(f[6:-6]) for f in os.listdir(d) if f.startswith('model-') and f.endswith('.index')]
    if len(its) != 1:
        sys.exit('expected exactly one checkpoint in %s, found %s' % (ckpt_dir, its))
    tf1.train.Saver().restore(sess, os.path.join(d, 'model-%d' % its[0]))
    agent.firstIter = False
    return agent, policy, use_cbf, its[0]


def episode(agent, policy, use_cbf, env, learn_gp=False):
    keys = ['t', 'p', 'v', 'd_f', 'd_r', 'u_rl', 'u_comp', 'u_cbf', 'u', 'slack', 'r']
    rec = {k: [] for k in keys}
    O, N = [], []
    obs, term = env.getObs(), False
    for _ in range(env.L):
        u_rl = float(np.clip(policy(obs), -U_MAX, U_MAX))
        u_comp = u_cbf = slack = 0.0
        if use_cbf:
            u_comp = float(agent.bar_comp.get_action(obs).item())
            f, g, x, sigma = dynamics_gp.get_GP_dynamics(agent, obs, env.t)
            u_cbf = float(cbf.control_barrier(agent, obs, u_rl + u_comp, f, g, x, sigma)[0])
            slack = agent.last_cbf['slack']
        u = float(np.clip(u_rl + u_comp + u_cbf, -U_MAX, U_MAX))
        d_f, d_r = env.headways()
        for k, val in zip(keys[:-1], [env.t, env.p.copy(), env.v.copy(), d_f, d_r, u_rl, u_comp, u_cbf, u, slack]):
            rec[k].append(val)
        O.append(obs.copy())
        obs, r, term, trunc, _ = env.step(u)
        rec['r'].append(r); N.append(obs.copy())
        if term or trunc:
            break
    d_f, d_r = env.headways()                         # final state (no action)
    for k, val in zip(keys, [env.t, env.p.copy(), env.v.copy(), d_f, d_r] + [np.nan] * 6):
        rec[k].append(val)
    if learn_gp and use_cbf:
        dynamics_gp.update_GP_dynamics(agent, dict(Observation=np.asarray(O), NextObservation=np.asarray(N),
                                                   Action=np.asarray(rec['u'][:-1])))
    out = {k: np.asarray(v, float) for k, v in rec.items()}
    out['collided'] = bool(term)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--policies', default=os.path.join(CAR, 'results', 'car_final'))
    ap.add_argument('--out', default=os.path.join(CAR, 'results', 'car_eval'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {'scenario': dict(init_seed=INIT_SEED, noise_seed=NOISE_SEED, steps=EPISODE_STEPS, dt=DT,
                                gp_warmup_episodes=WARMUP), 'runs': {}}
    for seed in a.seeds:
        arrays = {}
        for alg in ALGS:
            with tf1.Graph().as_default(), tf1.Session(config=tf1.ConfigProto(
                    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as sess:
                agent, policy, use_cbf, it = build_and_restore(
                    alg, seed, sess, os.path.join(a.policies, alg, 'seed%d' % seed, 'checkpoint'))
                if use_cbf:
                    for w in range(WARMUP):
                        episode(agent, policy, use_cbf, allCars(init_seed=seed + 500 + w, noise_seed=seed + 600 + w),
                                learn_gp=True)
                env = allCars(init_seed=INIT_SEED, noise_seed=NOISE_SEED)
                e = episode(agent, policy, use_cbf, env)
            for k, v in e.items():
                arrays['%s__%s' % (alg, k)] = np.asarray(v)
            m = np.minimum(e['d_f'], e['d_r'])
            act = e['u_cbf'][np.isfinite(e['u_cbf'])]
            summary['runs']['seed%d/%s' % (seed, alg)] = dict(
                checkpoint=it, collided=e['collided'], steps=int(np.sum(np.isfinite(e['u']))),
                min_headway=float(np.min(m)), t_min_headway=float(e['t'][int(np.argmin(m))]),
                episode_return=float(np.nansum(e['r'])), final_speed_car4=float(e['v'][-1, 3]),
                u_rms=float(np.sqrt(np.nanmean(e['u'] ** 2))), u_max_abs=float(np.nanmax(np.abs(e['u']))),
                u_cbf_rms=float(np.sqrt(np.mean(act ** 2))), cbf_active_fraction=float(np.mean(np.abs(act) > 1e-3)),
                u_comp_rms=float(np.sqrt(np.nanmean(e['u_comp'] ** 2))), slack_steps=int(np.nansum(e['slack'] > 0)))
            s = summary['runs']['seed%d/%s' % (seed, alg)]
            print('seed %d %-9s model-%-3d collided=%-5s min gap %5.2f m  return %8.1f  u_CBF rms %5.2f (%3.0f%% of steps)'
                  % (seed, alg, it, s['collided'], s['min_headway'], s['episode_return'], s['u_cbf_rms'],
                     100 * s['cbf_active_fraction']), flush=True)
        init = allCars(init_seed=INIT_SEED, noise_seed=NOISE_SEED)
        arrays['init_gaps'] = init.p[:-1] - init.p[1:]
        arrays['init_speeds'] = init.v.copy()
        np.savez_compressed(os.path.join(a.out, 'sim_seed%d.npz' % seed), **arrays)
    json.dump(summary, open(os.path.join(a.out, 'sim_summary.json'), 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
