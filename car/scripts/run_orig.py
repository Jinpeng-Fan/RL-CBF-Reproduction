"""Run the AUTHORS' UNMODIFIED car-following code (RL-CBF-master\\car\\<ALG>) for one seed.

The author's source files are copied byte-for-byte into <run-dir>\\source and executed with the
author's default hyper-parameters through the entry point named in the author's README
(TRPO / TRPO-CBF: sim.py, DDPG / DDPG-CBF: ddpg.py).  Nothing is passed on the command line except
the seed for the two DDPG scripts (they have a --random-seed option).  Everything else is done
*outside* the author's files, in this wrapper, and does not touch the algorithm:
  * seeding   : sim.py has no seed option, so random / numpy / TensorFlow are seeded here
                (the DDPG scripts seed numpy / TensorFlow / the replay buffer themselves).
  * plotting  : MPLBACKEND=Agg (the TRPO car_simulator imports pyplot; nothing is rendered).
  * savemat   : long_field_names=True (the authors' .mat field names exceed 31 characters).
  * CPU only  : CUDA_VISIBLE_DEVICES=-1, single-threaded BLAS, 2 TF intra-op threads.
The copied files are hashed (SHA256) and checked against RL-CBF-master\\car\\<ALG> before running;
the result is written to run_config.json.  Unknown arguments are passed through to the author's
script (used for the smoke test only).
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'                       # must precede the numpy import (OpenBLAS + TF deadlock)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse, sys, json, shutil, hashlib, runpy, random, platform, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--alg', required=True, choices=['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--run-dir', required=True)
ap.add_argument('--orig-root', required=True, help='<author repository>/car (rcheng805/RL-CBF)')
cfg, passthrough = ap.parse_known_args()

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
orig = Path(cfg.orig_root) / cfg.alg
run = Path(cfg.run_dir).resolve()
if run.exists():
    sys.exit('Refusing to reuse existing run directory: %s' % run)
src = run / 'source'
src.mkdir(parents=True)
for f in orig.glob('*.py'):
    shutil.copy2(str(f), str(src / f.name))
src_hash = {f.name: sha(f) for f in sorted(src.glob('*.py'))}
orig_hash = {f.name: sha(f) for f in sorted(orig.glob('*.py'))}
if src_hash != orig_hash:
    sys.exit('Copied source differs from %s' % orig)
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

if cfg.alg.startswith('TRPO'):
    random.seed(cfg.seed); np.random.seed(cfg.seed); tf.set_random_seed(cfg.seed)
    entry = src / 'sim.py'
    argv = [str(entry)] + passthrough
else:
    entry = src / 'ddpg.py'
    argv = [str(entry), '--random-seed', str(cfg.seed)] + passthrough

(run / 'run_config.json').write_text(json.dumps(dict(
    algorithm=cfg.alg, seed=cfg.seed, argv=argv, passthrough=passthrough,
    python=sys.executable, python_version=platform.python_version(), tensorflow=tf.__version__,
    numpy=np.__version__, started=datetime.datetime.now().isoformat(timespec='seconds'),
    original_source=str(orig), source_sha256=src_hash, sha256_matches_original=True),
    indent=2, ensure_ascii=False), encoding='utf-8')
sys.argv = argv
runpy.run_path(str(entry), run_name='__main__')
(run / 'DONE').write_text(datetime.datetime.now().isoformat(timespec='seconds'))
