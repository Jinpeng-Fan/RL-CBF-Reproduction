"""Windows defense launcher. Training remains in the original TF 1.15 interpreter.

Only the standard library is used here; TensorBoard gets a separate managed venv.
All generated data, managed environments and temporary caches live on E:.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime
from urllib.request import urlopen
import webbrowser


class ChildProcesses:
    """Kill the entire child process tree, including on closing the console window."""

    def __init__(self):
        self.processes = []
        self.files = []
        self.handle = None
        if os.name == 'nt':
            from ctypes import wintypes as w

            class BasicLimits(ctypes.Structure):
                _fields_ = [('ProcessTime', ctypes.c_int64), ('JobTime', ctypes.c_int64),
                            ('Flags', w.DWORD), ('MinWorkingSet', ctypes.c_size_t),
                            ('MaxWorkingSet', ctypes.c_size_t), ('ActiveProcesses', w.DWORD),
                            ('Affinity', ctypes.c_size_t), ('Priority', w.DWORD),
                            ('SchedulingClass', w.DWORD)]

            class IOCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_uint64) for n in
                            ('ReadOps', 'WriteOps', 'OtherOps', 'ReadBytes', 'WriteBytes', 'OtherBytes')]

            class ExtendedLimits(ctypes.Structure):
                _fields_ = [('Basic', BasicLimits), ('IO', IOCounters),
                            ('ProcessMemory', ctypes.c_size_t), ('JobMemory', ctypes.c_size_t),
                            ('PeakProcessMemory', ctypes.c_size_t), ('PeakJobMemory', ctypes.c_size_t)]

            self.k32 = ctypes.WinDLL('kernel32', use_last_error=True)
            self.k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
            self.k32.CreateJobObjectW.restype = w.HANDLE
            self.k32.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
            self.k32.SetInformationJobObject.restype = w.BOOL
            self.k32.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
            self.k32.AssignProcessToJobObject.restype = w.BOOL
            self.k32.CloseHandle.argtypes = [w.HANDLE]
            self.k32.CloseHandle.restype = w.BOOL
            self.handle = self.k32.CreateJobObjectW(None, None)
            limits = ExtendedLimits()
            limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.k32.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                error = ctypes.WinError(ctypes.get_last_error())
                self.k32.CloseHandle(self.handle)
                raise error

    def start(self, command, log, env, cwd):
        stream = open(log, 'w', encoding='utf-8')
        self.files.append(stream)
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=stream, stderr=stream,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.processes.append(process)
        if self.handle and not self.k32.AssignProcessToJobObject(self.handle, int(process._handle)):
            process.terminate()
            raise ctypes.WinError(ctypes.get_last_error())
        return process

    def close(self):
        if self.handle:
            self.k32.CloseHandle(self.handle)
            self.handle = None
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=15)
        for stream in self.files:
            stream.close()


def say(message):
    print(message, flush=True)


def e_drive_path(value, label):
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute() or path.drive.upper() != 'E:':
        raise ValueError(label + ' must be an absolute path on E:; C: output is disabled.')
    path = path.resolve()
    if path.drive.upper() != 'E:' or not str(path).isascii():
        raise ValueError(label + ' must resolve to an ASCII path on E: (no link to C:).')
    return path


def configure_storage(repo, config, override=None):
    base = e_drive_path(os.environ.get('RLCBF_DATA_ROOT') or
                        config.get('storage_root', 'E:/RL-CBF-Data'), 'storage_root')
    repo_id = hashlib.sha256(str(repo).encode('utf-8')).hexdigest()[:10]
    work = e_drive_path(override or base / repo_id, 'work_dir')
    paths = dict(TEMP=base / 'temp', TMP=base / 'temp', TMPDIR=base / 'temp',
                 PIP_CACHE_DIR=base / 'cache/pip', XDG_CACHE_HOME=base / 'cache',
                 KERAS_HOME=base / 'cache/keras', MPLCONFIGDIR=work / 'matplotlib')
    # Validate all output paths before creating any directories or importing training packages.
    paths = {key: e_drive_path(value, key) for key, value in paths.items()}
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(value) for key, value in paths.items()})
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    import tempfile
    tempfile.tempdir = str(paths['TEMP'])
    return base, work


def probe(python, code):
    try:
        result = subprocess.run([str(python), '-B', '-c', code], capture_output=True,
                                timeout=60, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def viewer_python(base, override=None):
    code = "import tensorboard; from tensorboard import default; assert tensorboard.__version__ == '2.21.0'"
    if override:
        python = e_drive_path(override, 'viewer_python')
        if not probe(python, code):
            raise RuntimeError('看板解释器不可用，需要 TensorBoard 2.21.0: ' + str(python))
        return python
    venv = e_drive_path(base / 'viewer', 'viewer environment')
    python = venv / 'Scripts/python.exe'
    base.mkdir(parents=True, exist_ok=True)
    # Two double-clicks during first-time installation must not modify the venv together.
    with (base / 'viewer-install.lock').open('a+b') as lock:
        import msvcrt
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        start = time.monotonic()
        while True:
            try:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() - start > 600:
                    raise RuntimeError('另一个窗口正在安装看板环境，请等待其完成。')
                time.sleep(1)
        try:
            if not probe(python, code):
                say('首次准备独立看板环境（需要联网，后续启动无需重复安装）...')
                subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
                subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                                'tensorboard==2.21.0'], check=True)
                if not probe(python, code):
                    raise RuntimeError('TensorBoard 安装后自检失败。')
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    return python


def training_python(repo, configured):
    explicit = os.environ.get('RLCBF_TRAIN_PY') or configured
    if explicit:
        candidates = [repo / os.path.expandvars(explicit)]
    else:
        candidates = [repo / '.venv-train/Scripts/python.exe',
                      repo.parent / '.runtime/python37_cpu/python.exe',
                      Path(sys.executable)]
    code = ('import sys, tensorflow as tf, numpy, scipy, sklearn, cvxopt, tflearn, gym; '
            "assert sys.version_info[:2] == (3, 7); assert tf.__version__ == '1.15.0'")
    for candidate in candidates:
        if probe(candidate, code):
            return candidate.resolve()
    raise RuntimeError('未找到可用的 Python 3.7 / TensorFlow 1.15 训练环境。'
                       '\n请在 demo_config.json 的 training.python 填写已安装环境的 python.exe 路径。'
                       '\n本机自动查找 ../.runtime/python37_cpu/python.exe；环境不会随代码仓库上传。')


def dashboard_entries(repo, session, run, config, mode, task):
    entries = []
    if mode == 'train':
        sources = [(run / ('seed%d' % seed) / alg / 'episodes.csv', 'car',
                    'car/seed%d/%s' % (seed, alg))
                   for seed in config['training']['seeds']
                   for alg in config['training']['algorithms']]
    else:
        groups = {'car_final': ('car', 'car'), 'final_600ep': ('pendulum', 'pendulum'),
                  'car_author_default': ('car', 'car_author'),
                  'author_code_default': ('pendulum', 'pendulum_author')}
        selected = config['history_groups']
        if not selected or len(selected) != len(set(selected)) or any(g not in groups for g in selected):
            raise ValueError('Invalid history_groups in demo_config.json')
        sources = []
        for group in selected:
            kind, label = groups[group]
            if task != 'both' and task != kind:
                continue
            for alg in ('TRPO-CBF', 'DDPG-CBF', 'TRPO', 'DDPG'):
                for seed in (1, 2, 3):
                    csv = repo / kind / 'results' / group / alg / ('seed%d' % seed) / 'episodes.csv'
                    if not csv.is_file():
                        raise FileNotFoundError(csv)
                    sources.append((csv, kind, '%s/seed%d/%s' % (label, seed, alg)))
    for csv, kind, name in sources:
        entries.append(dict(csv=str(csv), task=kind, name=name,
                            out=str(session / 'curves' / kind / name)))
    if not entries:
        raise ValueError('No experiment group selected for this panel')
    return entries


def free_port(preferred):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', preferred))
        except OSError:
            sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_server(process, url, log):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('TensorBoard 启动失败:\n' + log.read_text(encoding='utf-8', errors='replace')[-3000:])
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError('TensorBoard 启动超时，日志: ' + str(log))


def progress(run):
    import csv
    entries = []
    for csvfile in sorted(run.glob('seed*/*/episodes.csv')):
        try:
            with csvfile.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if rows and rows[-1].get('collisions_cum') is not None:
                last = rows[-1]
                entries.append('%s/%s: episode %s, collisions %s' %
                               (csvfile.parent.parent.name, csvfile.parent.name,
                                last['episode'], last['collisions_cum']))
        except (OSError, csv.Error):
            pass
    return ' | '.join(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['train', 'history'])
    ap.add_argument('--task', choices=['both', 'pendulum', 'car'], default='both')
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--config', type=Path)
    ap.add_argument('--viewer-python')
    ap.add_argument('--work-dir', type=Path)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--auto-close', type=float)
    ap.add_argument('--steps', type=int)
    ap.add_argument('--port', type=int, help='Preferred port for first panel; 0 chooses a free port')
    args = ap.parse_args()
    if args.mode == 'train' and args.task == 'pendulum':
        ap.error('The live training entry runs car experiments. Use history for pendulum results.')
    repo = args.repo.resolve()
    config = json.loads((args.config or repo / 'demo_config.json').read_text(encoding='utf-8-sig'))
    base, work = configure_storage(repo, config, args.work_dir)
    say('数据与缓存目录: ' + str(base))
    say('正在检查演示环境，请稍候...')
    training = config['training']
    algorithms, seeds = training['algorithms'], training['seeds']
    steps = args.steps if args.steps is not None else training['steps']
    if (not algorithms or len(set(algorithms)) != len(algorithms)
            or any(a not in ['TRPO-CBF', 'TRPO', 'DDPG-CBF', 'DDPG'] for a in algorithms)
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)
            or type(steps) is not int or steps <= 0
            or type(training['workers']) is not int or training['workers'] < 1):
        raise ValueError('demo_config.json 中的训练参数无效。')
    train_py = training_python(repo, training.get('python', '')) if args.mode == 'train' else None
    viewer = viewer_python(base, args.viewer_python or os.environ.get('RLCBF_VIEW_PY'))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session = work / 'sessions' / (args.mode + '_' + stamp)
    session.mkdir(parents=True)
    run = work / 'runs' / stamp
    entries = dashboard_entries(repo, session, run, config, args.mode, args.task)
    manifest = session / 'dashboard_runs.json'
    manifest.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    children = ChildProcesses()
    record = dict(mode=args.mode, repo=str(repo), run_dir=str(run) if train_py else None,
                  session_dir=str(session), status='starting', panels={})
    record_path = session / 'session.json'
    write_record = lambda: record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    write_record()
    try:
        ready = session / 'export.ready'
        exporter = children.start([str(viewer), '-B', str(repo / 'scripts/dashboard_export.py'),
                                   '--manifest', str(manifest), '--ready', str(ready)] +
                                  (['--watch'] if train_py else []), session / 'export.log', env, session)
        record['export_pid'] = exporter.pid
        deadline = time.monotonic() + 120
        while not ready.exists():
            if exporter.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('曲线准备失败，请检查 ' + str(session / 'export.log'))
            time.sleep(0.2)
        if not train_py and exporter.wait(timeout=15) != 0:
            raise RuntimeError('曲线导出失败，请检查 ' + str(session / 'export.log'))
        servers = []
        tasks = [task for task in ('pendulum', 'car') if any(e['task'] == task for e in entries)]
        for index, task in enumerate(tasks):
            subset = [e for e in entries if e['task'] == task]
            preferred = (config['ports']['train'] if train_py else
                         config['ports'].get(task, 6007 if task == 'pendulum' else 6009))
            port = free_port(args.port if args.port is not None and index == 0 else preferred)
            url = 'http://127.0.0.1:%d/' % port
            log = session / ('tensorboard_' + task + '.log')
            # Each task has its own root; native directory names avoid TensorBoard's
            # Windows logdir_spec alias suffix ("\\.").
            server = children.start([str(viewer), '-m', 'tensorboard.main',
                                     '--logdir', str(session / 'curves' / task),
                                     '--host', '127.0.0.1', '--port', str(port), '--reload_interval', '2',
                                     '--load_fast', 'false', '--samples_per_plugin', 'scalars=100000'],
                                    log, env, session)
            servers.append(server)
            record['panels'][task] = dict(url=url + '#scalars', pid=server.pid, runs=len(subset))
            write_record()
            wait_server(server, url, log)
            say(('%s 面板 (%d 组): ' % (task, len(subset))) + url + '#scalars')
        say('保持本窗口打开。按 Ctrl+C 或关闭本窗口可停止本次训练和看板。')
        if train_py:
            command = [str(train_py), '-B', str(repo / 'car/scripts/run_final.py'), '--root', str(run),
                       '--seeds'] + list(map(str, seeds)) + ['--algs'] + algorithms + [
                       '--workers', str(training['workers']), '--', '--total_train_step', str(steps)]
            train = children.start(command, session / 'training.log', env, repo)
            record.update(training_pid=train.pid, status='training', training_command=command)
            say('车辆训练输出: ' + str(run))
            say('正在初始化训练。并行进程会间隔 20 秒启动，曲线按 episode 更新。')
        else:
            record['status'] = 'viewing'
        write_record()
        for panel in record['panels'].values():
            if not args.no_browser and not webbrowser.open(panel['url']):
                say('浏览器未自动打开，请打开上面的地址。')
        def check_services():
            if any(server.poll() is not None for server in servers):
                raise RuntimeError('TensorBoard 意外退出，请检查 ' + str(session))
            if train_py and exporter.poll() is not None:
                raise RuntimeError('实时曲线导出进程退出，请检查 ' + str(session / 'export.log'))
        if train_py:
            previous = ''
            while train.poll() is None:
                check_services()
                current = progress(run)
                if current and current != previous:
                    say(current)
                    previous = current
                time.sleep(2)
            if train.returncode:
                raise RuntimeError('训练失败。检查 ' + str(session / 'training.log') + ' 及 ' + str(run / 'logs'))
            time.sleep(3)  # Allow the exporter to flush the final completed episodes.
            check_services()
            record['status'] = 'completed'
            write_record()
            say('训练完成，面板保持打开。短训练用于演示过程，完整结果请打开历史面板。')
        if args.auto_close is None:
            input('按 Enter 关闭本次面板服务，或继续在浏览器中查看曲线...')
        else:
            deadline = time.monotonic() + max(0, args.auto_close)
            while time.monotonic() < deadline:
                check_services()
                time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        record['status'] = 'stopped'
        say('正在停止本次进程，已写入的日志会保留。')
        return 0
    except Exception:
        record['status'] = 'failed'
        raise
    finally:
        children.close()
        record['closed_at'] = datetime.now().isoformat(timespec='seconds')
        write_record()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        say('\n启动失败: ' + str(error))
        sys.exit(1)
