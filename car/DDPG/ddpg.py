"""Plain DDPG baseline (no CBF / GP / compensator).

The implementation is shared with ../DDPG-CBF/ddpg.py (same environment, networks, replay and
training loop); this entry only switches the safety layer off.   python ddpg.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..', 'DDPG-CBF'))
sys.path.insert(0, SHARED)
import importlib.util

spec = importlib.util.spec_from_file_location('ddpg_shared', os.path.join(SHARED, 'ddpg.py'))
ddpg_shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ddpg_shared)

if __name__ == "__main__":
    ddpg_shared.main(sys.argv[1:] + ['--no_cbf'], default_cbf=False)
