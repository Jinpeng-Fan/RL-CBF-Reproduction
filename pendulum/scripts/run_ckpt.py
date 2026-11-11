"""run_one.py + parameter snapshots, for the first-vs-last-policy evaluation.

Identical training to run_one.py (same frozen source copy, same explicit hyper-parameters, CPU only).
The only additions are hooks OUTSIDE the algorithm files; they never change what the scripts compute:
  * ckpt/first : all TF variables right after the last global_variables_initializer() call that
                 precedes training (= initial policy / critic / compensator / normalizer).
  * ckpt/last  : all TF variables when the training session is closed (= final parameters),
                 plus gp_last.pkl (the CBF agent's GP models after the last episode) and
                 args.pkl (TRPO argparse namespace, needed to rebuild the graph for evaluation).
usage: run_ckpt.py --alg TRPO-CBF --seed 1 --episodes 600 --run-dir <new dir>
"""
import argparse, os, sys, json, shutil, hashlib, runpy, pickle
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ap = argparse.ArgumentParser()
ap.add_argument('--alg', required=True, choices=['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--episodes', type=int, required=True)
ap.add_argument('--run-dir', required=True)
cfg = ap.parse_args()

HERE = Path(__file__).resolve().parent.parent
run = Path(cfg.run_dir).resolve()
if run.exists():
    sys.exit('Refusing to reuse existing run directory: %s' % run)
src = run / 'source'
src.mkdir(parents=True)
for f in (HERE / cfg.alg).glob('*.py'):
    shutil.copy2(str(f), str(src / f.name))
os.chdir(str(run))
sys.path.insert(0, str(src))
(run / 'ckpt').mkdir()

import scipy.io
_savemat = scipy.io.savemat
scipy.io.savemat = lambda *a, **kw: _savemat(*a, **dict(kw, long_field_names=True))
import tensorflow as tf

STASH = {}


def _save(sess, tag):
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    saver.save(sess, str(run / 'ckpt' / tag / 'model'), write_meta_graph=False)


class _CkptSession(tf.Session):
    def __init__(self, *a, **kw):
        c = kw.get('config') or tf.ConfigProto()
        c.intra_op_parallelism_threads, c.inter_op_parallelism_threads = 2, 1
        kw['config'] = c
        super(_CkptSession, self).__init__(*a, **kw)

    def run(self, fetches, *a, **kw):
        out = super(_CkptSession, self).run(fetches, *a, **kw)
        if isinstance(fetches, tf.Operation) and fetches.type == 'NoOp' and fetches.name.startswith('init'):
            _save(self, 'first')          # overwritten until the last initializer before training
        return out

    def __exit__(self, *exc):
        if exc[0] is None:
            _save(self, 'last')
            agent = STASH.get('agent')
            if agent is not None and hasattr(agent, 'GP_model'):
                with open(str(run / 'ckpt' / 'gp_last.pkl'), 'wb') as f:
                    pickle.dump(agent.GP_model, f)
            if agent is not None and hasattr(agent, 'args'):
                with open(str(run / 'ckpt' / 'args.pkl'), 'wb') as f:
                    pickle.dump(agent.args, f)
        return super(_CkptSession, self).__exit__(*exc)


tf.Session = _CkptSession

# Remember the object that owns the GP / args (the scripts import these modules by name).
_cls = None
if cfg.alg == 'DDPG-CBF':
    import learner as _m; _cls = _m.LEARNER
elif cfg.alg.startswith('TRPO'):
    import trpo as _m; _cls = _m.TRPO
if _cls is not None:
    _init = _cls.__init__

    def _wrapped(self, *a, **kw):
        STASH['agent'] = self
        _init(self, *a, **kw)
    _cls.__init__ = _wrapped

EPI_PER_ITER = 5
if cfg.alg.startswith('DDPG'):
    entry = src / 'ddpg.py'
    argv = [str(entry), '--max-episodes', str(cfg.episodes), '--random-seed', str(cfg.seed),
            '--max-episode-len', '200', '--episodes-per-iteration', str(EPI_PER_ITER),
            '--gamma', '0.99', '--noise-sigma', '1.0', '--summary-dir', str(run / 'tf_summary'),
            '--minibatch-size', '256', '--critic-lr', '0.0005', '--actor-lr', '0.0001',
            '--buffer-size', '100000']
    if cfg.alg == 'DDPG-CBF':
        argv += ['--bar_constraint_max', '10', '--comp_num_backtracking', '30']
else:
    entry = src / 'main.py'
    argv = [str(entry), '--total_train_step', str(200 * cfg.episodes), '--random-seed', str(cfg.seed),
            '--timesteps_per_batch', str(200 * EPI_PER_ITER), '--max_path_length', '200',
            '--gamma', '0.99', '--lamda', '0.95', '--kl_constraint', '0.005', '--num_backtracking', '20',
            '--init_logstd', '1.0986']
    if cfg.alg == 'TRPO-CBF':
        argv += ['--bar_constraint_max', '10', '--comp_num_backtracking', '30']

(run / 'run_config.json').write_text(json.dumps(dict(
    algorithm=cfg.alg, seed=cfg.seed, episodes=cfg.episodes, argv=argv, checkpoints=True,
    source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*.py')}),
    indent=2), encoding='utf-8')
sys.argv = argv
runpy.run_path(str(entry), run_name='__main__')
(run / 'DONE').write_text('ok')
