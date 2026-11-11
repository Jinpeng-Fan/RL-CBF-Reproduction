"""Deploy the first and the last policy of a run_ckpt.py run (paper Fig. 4 style evaluation).

usage: eval_ckpt.py --run-dir <runs_ckpt/ALG/seed1> [--n-init 10] [--init-seed 2026]
Writes <run-dir>/eval_first_last.npz with angle / torque trajectories for both policies.

Protocol (identical for the four algorithms):
  * graph rebuilt from the run's own source copy, parameters restored from ckpt/first or ckpt/last;
  * deterministic action: TRPO -> policy mean, DDPG -> actor output (no exploration noise);
  * CBF versions: first policy uses the GP prior and zero compensator (as in the first training
    iteration); last policy uses the GP fitted after the final training episode and the trained
    compensator. The GP is frozen during evaluation;
  * n_init initial states (theta, theta_dot) ~ U[-1,1]^2 from RandomState(init_seed), 200 steps (10 s),
    same environment as training (m=l=1, g=10, dt=0.05, |u|<=15).
"""
import argparse, os, sys, json, pickle, runpy, faulthandler
# BLAS threads must be pinned BEFORE numpy is imported (multithreaded OpenBLAS next to TF deadlocks).
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np
if os.environ.get('EVAL_STACKDUMP'):
    faulthandler.dump_traceback_later(int(os.environ['EVAL_STACKDUMP']), repeat=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', required=True)
ap.add_argument('--n-init', type=int, default=10)
ap.add_argument('--init-seed', type=int, default=2026)
a = ap.parse_args()
run = os.path.abspath(a.run_dir)
cfg = json.load(open(os.path.join(run, 'run_config.json')))
alg, src = cfg['algorithm'], os.path.join(run, 'source')
sys.path.insert(0, src)
import tensorflow as tf
import gym


def argv_value(name, default=None):
    v = cfg['argv']
    return v[v.index(name) + 1] if name in v else default


def build(sess):
    """Rebuild the training graph exactly as the entry script does; return (policy_fn, cbf_agent, env)."""
    if alg.startswith('DDPG'):
        g = runpy.run_path(os.path.join(src, 'ddpg.py'), run_name='ddpg_eval')
        env = g['configure_comparison_env'](gym.make('Pendulum-v0'), cfg['seed'])
        env.unwrapped.max_torque, env.unwrapped.max_speed = 15., 60.
        env.unwrapped.action_space = gym.spaces.Box(low=-15., high=15., shape=(1,))
        env.action_space = env.unwrapped.action_space
        obs_norm = g['ObsNormalizer'](sess, 3)
        actor = g['ActorNetwork'](sess, 3, 1, env.action_space.high, float(argv_value('--actor-lr')), 0.001,
                                  int(argv_value('--minibatch-size')), obs_norm=obs_norm)
        g['CriticNetwork'](sess, 3, 1, float(argv_value('--critic-lr')), 0.001, 0.99,
                           actor.get_num_trainable_vars(), obs_norm=obs_norm)
        agent = None
        if alg == 'DDPG-CBF':
            agent = g['LEARNER'](env)
            agent.enforce_speed_limit = False
            agent.bar_comp = g['BARRIER'](argparse.Namespace(bar_constraint_max=10., comp_num_backtracking=30),
                                          sess, 3, 1, obs_normalizer=obs_norm)
        g['build_summaries']()
        policy = lambda o: actor.predict(np.asarray(o).reshape(1, 3))[0]
        comp = (lambda o: agent.bar_comp.get_action(o)[0]) if agent else None
        return policy, comp, agent, env
    import learn
    with open(os.path.join(run, 'ckpt', 'args.pkl'), 'rb') as f:
        args = pickle.load(f)
    L = learn.LEARNER(args, sess, args)
    t = L.agent
    policy = lambda o: t.act(o)[1].reshape(-1)          # mean action
    comp = (lambda o: np.asarray(t.bar_comp.get_action(o)).reshape(-1)) if alg == 'TRPO-CBF' else None
    return policy, comp, (t if alg == 'TRPO-CBF' else None), L.simulator


def restore(sess, tag):
    path = os.path.join(run, 'ckpt', tag, 'model')
    reader = tf.train.NewCheckpointReader(path)
    var_list = {v.op.name: v for v in tf.global_variables() if reader.has_tensor(v.op.name)}
    missing = [v.op.name for v in tf.trainable_variables() if v.op.name not in var_list]
    assert not missing, ('variables not in checkpoint', missing[:5])
    tf.train.Saver(var_list).restore(sess, path)


inits = np.random.RandomState(a.init_seed).uniform(-1, 1, size=(a.n_init, 2))
out = dict(algorithm=alg, inits=inits, dt=0.05)
for tag in ('first', 'last'):
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)) as sess:
        policy, comp, agent, env = build(sess)
        sess.run(tf.global_variables_initializer())
        restore(sess, tag)
        if agent is not None:
            import dynamics_gp, cbf
            dynamics_gp.build_GP_model(agent)                    # prior
            if tag == 'last':
                with open(os.path.join(run, 'ckpt', 'gp_last.pkl'), 'rb') as f:
                    agent.GP_model = pickle.load(f)
        TH, U, UC = [], [], []
        for x0 in inits:
            env.reset()
            env.unwrapped.state = np.array(x0, dtype=float)
            o = env.unwrapped._get_obs()
            th, us, ucs = [x0[0]], [], []
            for _ in range(200):
                v = np.asarray(policy(o), dtype=float).reshape(1)
                v = np.clip(v, -15., 15.)
                uc = np.zeros(1)
                if not (np.all(np.isfinite(o)) and np.all(np.isfinite(v))):
                    raise FloatingPointError('non-finite obs/action: obs=%s v=%s step=%d' % (o, v, len(us)))
                if agent is not None:
                    v = v + (np.zeros(1) if tag == 'first' else np.asarray(comp(o)).reshape(1))
                    if not np.all(np.isfinite(v)):
                        raise FloatingPointError('non-finite compensated action at obs=%s' % (o,))
                    f_, g_, x_, S_ = dynamics_gp.get_GP_dynamics(agent, o)
                    uc = np.asarray(cbf.control_barrier(agent, o, v, f_, g_, x_, S_)).reshape(1)
                u = v + uc
                o, _, _, _ = env.step(u)
                th.append(float(env.unwrapped.state[0])); us.append(float(u[0])); ucs.append(float(uc[0]))
            TH.append(th); U.append(us); UC.append(ucs)
            print(alg, tag, 'init', len(TH), 'done', flush=True)
        out[tag + '_theta'] = np.array(TH)       # unwrapped simulator angle
        out[tag + '_u'] = np.array(U)
        out[tag + '_ucbf'] = np.array(UC)
    wrapped = (out[tag + '_theta'] + np.pi) % (2 * np.pi) - np.pi
    print(alg, tag, 'max|theta| per init:', np.round(np.abs(wrapped).max(1), 3))
np.savez(os.path.join(run, 'eval_first_last.npz'), **out)
print('saved', os.path.join(run, 'eval_first_last.npz'))
