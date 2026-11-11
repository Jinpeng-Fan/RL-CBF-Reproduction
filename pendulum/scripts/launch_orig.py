"""Start run_orig.py for every (algorithm, seed) in a NEW root directory (detached)."""
import argparse, subprocess, sys, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='new directory; must not exist')
ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--algs', nargs='+', default=['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF'])
a, passthrough = ap.parse_known_args()      # passthrough: smoke-test overrides for the author's script

root = Path(a.root).resolve()
if root.exists():
    sys.exit('Refusing to reuse existing directory: %s' % root)
(root / 'logs').mkdir(parents=True)
here = Path(__file__).resolve().parent
flags = 0x00000008 | 0x00000200 if os.name == 'nt' else 0
for seed in a.seeds:
    for alg in a.algs:
        name = '%s_s%d' % (alg, seed)
        out = open(str(root / 'logs' / (name + '.out')), 'wb')
        err = open(str(root / 'logs' / (name + '.err')), 'wb')
        p = subprocess.Popen(
            [sys.executable, str(here / 'run_orig.py'), '--alg', alg, '--seed', str(seed),
             '--run-dir', str(root / alg / ('seed%d' % seed))] + passthrough,
            stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags)
        (root / 'logs' / (name + '.pid')).write_text(str(p.pid))
        print('started', name, 'pid', p.pid)
