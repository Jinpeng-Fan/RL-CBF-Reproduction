"""Train the revised car-following code (DDPG / DDPG-CBF / TRPO / TRPO-CBF), one process per run.

usage (from car/):  python scripts/run_final.py --root runs/final [--seeds 1 2 3] [--algs ...] [--workers 5]
                    [-- --total_train_step 3000]          (arguments after -- go to every run; smoke tests)
Only --random_seed and --logroot are passed, so all other settings are the defaults in
TRPO-CBF/sim.py and TRPO-CBF/config.py.  Output (root must not exist):
  <root>/seed<k>/<ALG>/  episodes.csv, TensorBoard events, checkpoints, run_config.json
  <root>/logs/seed<k>_<ALG>.log / .err
Runs with the interpreter that starts this script (the report used Python 3.7.9 + TensorFlow 1.15, CPU).
"""
import argparse, datetime, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CAR = Path(__file__).resolve().parents[1]
ALGOS = {'DDPG-CBF': ('DDPG-CBF', 'ddpg.py'), 'DDPG': ('DDPG', 'ddpg.py'),
         'TRPO-CBF': ('TRPO-CBF', 'sim.py'), 'TRPO': ('TRPO', 'sim.py')}

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='new directory (relative to car/ or absolute)')
ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--algs', nargs='+', default=['DDPG-CBF', 'DDPG', 'TRPO-CBF', 'TRPO'])
ap.add_argument('--workers', type=int, default=5)
ap.add_argument('--python', default=sys.executable)
a, extra = ap.parse_known_args()
extra = [x for x in extra if x != '--']
root = (CAR / a.root).resolve()
if root.exists():
    sys.exit('Refusing to reuse existing directory: %s' % root)
(root / 'logs').mkdir(parents=True)

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_sha = {'%s/%s' % (d, f.name): sha(f) for d in ('DDPG', 'DDPG-CBF', 'TRPO', 'TRPO-CBF')
              for f in sorted((CAR / d).glob('*.py'))}
env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg', TF_CPP_MIN_LOG_LEVEL='2')


def run(job):
    idx, seed, alg = job
    if idx < a.workers:
        time.sleep(20 * idx)    # stagger start-ups (simultaneous TF initialisation crashed natively once)
    folder, script = ALGOS[alg]
    logroot = root / ('seed%d' % seed)
    rundir = logroot / alg
    rundir.mkdir(parents=True)
    cmd = [a.python, script, '--random_seed', str(seed), '--logroot', str(logroot)] + extra
    (rundir / 'run_config.json').write_text(json.dumps(dict(
        algorithm=alg, seed=seed, cwd=folder, command=[Path(cmd[0]).name] + cmd[1:],
        started=datetime.datetime.now().isoformat(timespec='seconds'), source_sha256=source_sha),
        indent=2, ensure_ascii=False), encoding='utf-8')
    tag = 'seed%d_%s' % (seed, alg)
    t0 = time.time()
    print('start  ', tag, flush=True)
    with open(str(root / 'logs' / (tag + '.log')), 'w') as lo, open(str(root / 'logs' / (tag + '.err')), 'w') as le:
        rc = subprocess.call(cmd, cwd=str(CAR / folder), env=env, stdout=lo, stderr=le)
    print('finish ', tag, 'rc=%d' % rc, '%.0fs' % (time.time() - t0), flush=True)
    return tag, rc


if __name__ == '__main__':
    jobs = [(i, s, al) for i, (s, al) in enumerate((s, al) for al in a.algs for s in a.seeds)]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, jobs))
    print('ALL DONE; failed:', [t for t, rc in res if rc != 0] or 'none', flush=True)

    sys.exit(1 if any(rc != 0 for _, rc in res) else 0)
