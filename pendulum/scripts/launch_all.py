"""Start run_one.py for every (algorithm, seed) in a NEW root directory (detached)."""
import argparse, subprocess, sys, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='new directory; must not exist')
ap.add_argument('--episodes', type=int, required=True)
ap.add_argument('--seeds', type=int, nargs='+', default=[1])
ap.add_argument('--algs', nargs='+', default=['DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF'])
ap.add_argument('--src-dir', default=None, help='ablation source folder (single --algs entry)')
ap.add_argument('--label', default=None, help='run name used instead of the algorithm name')
ap.add_argument('--allow-existing-root', action='store_true', help='add runs to an existing root (new run dirs only)')
a = ap.parse_args()

root = Path(a.root).resolve()
if root.exists() and not a.allow_existing_root:
    sys.exit('Refusing to reuse existing directory: %s' % root)
(root / 'logs').mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
flags = 0x00000008 | 0x00000200 if os.name == 'nt' else 0  # DETACHED_PROCESS | NEW_PROCESS_GROUP
for seed in a.seeds:
    for alg in a.algs:
        label = a.label or alg
        name = '%s_s%d' % (label, seed)
        out = open(str(root / 'logs' / (name + '.out')), 'wb')
        err = open(str(root / 'logs' / (name + '.err')), 'wb')
        p = subprocess.Popen(
            [sys.executable, str(here / 'run_one.py'), '--alg', alg, '--seed', str(seed),
             '--episodes', str(a.episodes), '--run-dir', str(root / label / ('seed%d' % seed))]
            + (['--src-dir', a.src_dir] if a.src_dir else []),
            stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=flags)
        (root / 'logs' / (name + '.pid')).write_text(str(p.pid))
        print('started', name, 'pid', p.pid)
