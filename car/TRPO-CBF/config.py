"""Constants of the car-following task (paper Appendix B + revisions).

Everything physical lives here so that the simulator, the nominal model,
the GP, the CBF and the learners agree on the same numbers.
"""
DT = 0.05              # integration step [s]
EPISODE_STEPS = 200    # 10 s per episode
KD = 0.1               # true damping (car 4's nominal model assumes 0)
KP, KB = 4.0, 20.0     # true driver gains (paper eq. 27)
NOM_KP, NOM_KB = 3.5, 18.0   # car 4's (inaccurate) belief about the others
V_DES = 30.0
D_REF, D_REF5 = 6.0, 12.0    # drivers start braking below these gaps
A_MAX = 100.0          # acceleration/control bound, paper eq. (27)
U_MAX = 100.0
SIGMA_W = 1.0          # std of Gaussian noise added to every car's acceleration

D_SOFT = 3.0           # soft constraint (reward) [m]
D_MIN = 2.0            # hard constraint (CBF) [m]
EPS_D = 0.5            # floor of the distance in the 500/d penalty
C_COLL0, C_COLL1 = 500.0, 100.0   # collision penalty = C0 + C1 * remaining steps

# ---- initial condition (used by allCars.reset) ----
INIT_GAP = 6.0         # nominal spacing between neighbouring cars [m]
INIT_GAP_RANGE = 0.5   # random +/- on each of the 4 gaps (0 = no randomness)
INIT_V = 30.0          # nominal initial speed of every car [m/s]
INIT_V_RANGE = 1.0     # random +/- on each car's speed
INIT_P_LAST = 10.0     # position of car 5 (the rearmost)

# state layout: obs = [p1 v1 a1, p2 v2 a2, ..., p5 v5 a5]; car 4 is index 3.
# Indices of the states that enter the CBF (cars 3, 4, 5: position, velocity)
GP_IDX = [6, 7, 9, 10, 12, 13]
