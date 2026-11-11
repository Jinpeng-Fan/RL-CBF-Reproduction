"""Start run_orig.py for every (algorithm, seed) in a NEW root directory (detached processes).

usage: python scripts/launch_orig.py --root <RUN_ROOT> --orig-root <author repository>/car
(--orig-root and any other unknown arguments are forwarded to run_orig.py; the author scripts run
with their default settings.)  Start-ups are not staggered here: in our run 4 of 12 processes
crashed natively at start-up and were restarted one by one with run_orig.py.
"""
import argparse, subprocess, sys, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='new directory; must not exist')
ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--algs', nargs='+', default=['TRPO-CBF', 'TRPO', 'DDPG-CBF', 'DDPG'])
a, passthrough = ap.parse_known_args()      # passthrough: smoke-test overrides for the author's script

root = Path(a.root).resolve()
if root.exists():
    sys.exit('Refusing to reuse existing directory: %s' % root)
(root / 'logs').mkdir(parents=True)
here = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2')
flags = 0x00000008 | 0x00000200 if os.name == 'nt' else 0
for alg in a.algs:
    for seed in a.seeds:
        name = '%s_s%d' % (alg, seed)
        out = open(str(root / 'logs' / (name + '.out')), 'wb')
        err = open(str(root / 'logs' / (name + '.err')), 'wb')
        p = subprocess.Popen(
            [sys.executable, str(here / 'run_orig.py'), '--alg', alg, '--seed', str(seed),
             '--run-dir', str(root / alg / ('seed%d' % seed))] + passthrough,
            stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        (root / 'logs' / (name + '.pid')).write_text(str(p.pid))
        print('started', name, 'pid', p.pid)
