"""Plain TRPO baseline (no CBF / GP / compensator).

The implementation is shared with ../TRPO-CBF (same environment, networks, GAE, trust region);
this entry only switches the safety layer off.   python sim.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, '..', 'TRPO-CBF'))
sys.path.insert(0, SHARED)          # shared modules take priority over the old files in this folder
import sim as shared_sim

if __name__ == "__main__":
    shared_sim.main(argv=sys.argv[1:] + ['--no_cbf'], default_cbf=False)
