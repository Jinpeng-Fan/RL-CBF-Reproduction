"""Run ONE algorithm / seed from a frozen source snapshot into a fresh directory.

Nothing is inherited from earlier runs: the source folder is copied into the run
directory first (later edits cannot leak in), every hyper-parameter is passed
explicitly on the command line, and the run directory must not exist yet.
Metrics are the 'LIVE {...}' lines the training scripts print per episode.
"""
import argparse, os, sys, json, shutil, hashlib, runpy
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'   # the 200x200 GP solves are slower when multithreaded
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CPU only: the networks are tiny, and many parallel runs exhaust GPU memory on startup.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ap = argparse.ArgumentParser()
ap.add_argument('--alg', required=True, choices=['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--episodes', type=int, required=True)
ap.add_argument('--run-dir', required=True)
ap.add_argument('--src-dir', default=None, help='override source folder (ablations)')
ap.add_argument('--epi-per-iter', type=int, default=5, help='episodes per policy iteration (all algorithms)')
ap.add_argument('--init-logstd', type=float, default=1.0986, help='TRPO initial log action std (log 3)')
cfg = ap.parse_args()

HERE = Path(__file__).resolve().parent.parent
run = Path(cfg.run_dir).resolve()
if run.exists():
    sys.exit('Refusing to reuse existing run directory: %s' % run)
src = run / 'source'
src.mkdir(parents=True)
for f in (Path(cfg.src_dir) if cfg.src_dir else HERE / cfg.alg).glob('*.py'):
    shutil.copy2(f, src / f.name)
os.chdir(str(run))
sys.path.insert(0, str(src))

import scipy.io
_savemat = scipy.io.savemat
scipy.io.savemat = lambda *a, **kw: _savemat(*a, **dict(kw, long_field_names=True))  # keys > 31 chars
import tensorflow as tf
_Session = tf.Session
class _LimitedSession(_Session):
    def __init__(self, *a, **kw):
        c = kw.get('config') or tf.ConfigProto()
        c.intra_op_parallelism_threads, c.inter_op_parallelism_threads = 2, 1
        kw['config'] = c
        super(_LimitedSession, self).__init__(*a, **kw)
tf.Session = _LimitedSession

EPI_PER_ITER = cfg.epi_per_iter
if cfg.alg.startswith('DDPG'):
    entry = src / 'ddpg.py'
    argv = [str(entry), '--max-episodes', str(cfg.episodes), '--random-seed', str(cfg.seed),
            '--max-episode-len', '200', '--episodes-per-iteration', str(EPI_PER_ITER),
            '--gamma', '0.99', '--noise-sigma', '1.0', '--summary-dir', str(run / 'tf_summary'),
            '--minibatch-size', '256', '--critic-lr', '0.0005', '--actor-lr', '0.0001',
            '--buffer-size', '100000']  # noise: independent Gaussian, sigma = --noise-sigma
    if cfg.alg == 'DDPG-CBF':
        argv += ['--bar_constraint_max', '10', '--comp_num_backtracking', '30']
else:
    entry = src / 'main.py'
    argv = [str(entry), '--total_train_step', str(200 * cfg.episodes), '--random-seed', str(cfg.seed),
            '--timesteps_per_batch', str(200 * EPI_PER_ITER), '--max_path_length', '200',
            '--gamma', '0.99', '--lamda', '0.95', '--kl_constraint', '0.005', '--num_backtracking', '20',
            '--init_logstd', str(cfg.init_logstd)]
    if cfg.alg == 'TRPO-CBF':
        argv += ['--bar_constraint_max', '10', '--comp_num_backtracking', '30']

(run / 'run_config.json').write_text(json.dumps(dict(
    algorithm=cfg.alg, seed=cfg.seed, episodes=cfg.episodes, argv=argv,
    source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*.py')}),
    indent=2), encoding='utf-8')
sys.argv = argv
runpy.run_path(str(entry), run_name='__main__')
(run / 'DONE').write_text('ok')
