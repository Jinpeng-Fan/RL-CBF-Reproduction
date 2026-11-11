"""Run the AUTHORS' UNMODIFIED pendulum code (RL-CBF-master\\pendulum\\<ALG>) for one seed.

The author's source files are copied byte-for-byte into <run-dir>\\source and executed with the
author's default hyper-parameters (nothing is passed on the command line except the seed for the
two DDPG scripts, which have a --random-seed option).  The only things done *outside* the author's
files (in this wrapper) are environment-level adaptations that do not touch the algorithm:
  * seeding   : the TRPO / TRPO-CBF scripts have no seed option, so numpy / random / TensorFlow and
                the gym environment are seeded here (the DDPG scripts seed themselves).
  * render    : PendulumEnv.render() is a no-op (TRPO's sim() calls env.render(); no display here).
  * savemat   : long_field_names=True (the authors' .mat field names exceed 31 characters).
  * CPU only  : CUDA_VISIBLE_DEVICES=-1 and limited threads (many parallel runs).
Unknown command-line arguments are passed through to the author's script (used for smoke tests only).
"""
import argparse, os, sys, json, shutil, hashlib, runpy, random
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ap = argparse.ArgumentParser()
ap.add_argument('--alg', required=True, choices=['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--run-dir', required=True)
ap.add_argument('--orig-root', required=True, help='pendulum/ folder of the authors\' repository (github.com/rcheng805/RL-CBF)')
cfg, passthrough = ap.parse_known_args()

run = Path(cfg.run_dir).resolve()
if run.exists():
    sys.exit('Refusing to reuse existing run directory: %s' % run)
src = run / 'source'
src.mkdir(parents=True)
for f in (Path(cfg.orig_root) / cfg.alg).glob('*.py'):
    shutil.copy2(str(f), str(src / f.name))
os.chdir(str(run))
sys.path.insert(0, str(src))

import numpy as np
import scipy.io
_savemat = scipy.io.savemat
scipy.io.savemat = lambda *a, **kw: _savemat(*a, **dict(kw, long_field_names=True))
import tensorflow as tf
_Session = tf.Session
class _LimitedSession(_Session):
    def __init__(self, *a, **kw):
        c = kw.get('config') or tf.ConfigProto()
        c.intra_op_parallelism_threads, c.inter_op_parallelism_threads = 2, 1
        kw['config'] = c
        super(_LimitedSession, self).__init__(*a, **kw)
tf.Session = _LimitedSession

import gym
from gym.envs.classic_control import pendulum
pendulum.PendulumEnv.render = lambda self, *a, **kw: None

if cfg.alg.startswith('TRPO'):          # the DDPG scripts seed numpy / tf / env themselves
    random.seed(cfg.seed); np.random.seed(cfg.seed); tf.set_random_seed(cfg.seed)
    _make = gym.make
    def _seeded_make(*a, **kw):
        env = _make(*a, **kw)
        env.seed(cfg.seed)
        return env
    gym.make = _seeded_make

if cfg.alg.startswith('DDPG'):
    entry = src / 'ddpg.py'
    argv = [str(entry), '--random-seed', str(cfg.seed)] + passthrough
else:
    entry = src / 'main.py'
    argv = [str(entry)] + passthrough

(run / 'run_config.json').write_text(json.dumps(dict(
    algorithm=cfg.alg, seed=cfg.seed, argv=argv, original_source=str(Path(cfg.orig_root) / cfg.alg),
    source_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*.py')}),
    indent=2, ensure_ascii=False), encoding='utf-8')
sys.argv = argv
runpy.run_path(str(entry), run_name='__main__')
(run / 'DONE').write_text('ok')
