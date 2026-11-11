"""Five-car following simulator (paper Appendix B), revised.

Fixes w.r.t. the original code
  A1  collision (gap<=0 to the front or rear neighbour of car 4) terminates the episode
      with a penalty that grows with the number of remaining steps; the soft penalty
      500/max(d,eps) is only evaluated for d>0.
  A2  correct integration: p+ = p + v*dt + 0.5*a*dt^2 with the OLD velocity.
  A3  Gaussian acceleration noise on every car; randomised initial state with its own RNG.
  A4  drivers brake harder when the gap is SMALLER (-kb*max(0, d_ref-gap)), accelerations clipped.
  A6  nominal model is a pure function of (obs, t).
  A7  the reward uses the realised acceleration of car 4.
"""
import numpy as np
from config import *


def leader_vdes(t):
    return V_DES - 10.0 * np.sin(0.2 * t)


def driver_controls(p, v, t, kp, kb):
    """Control of cars 1,2,3,5 (index 0,1,2,4); index 3 (car 4) is left at 0."""
    u = np.zeros(5)
    u[0] = kp * (leader_vdes(t) - v[0])
    u[1] = kp * (V_DES - v[1]) - kb * max(0.0, D_REF - (p[0] - p[1]))
    u[2] = kp * (V_DES - v[2]) - kb * max(0.0, D_REF - (p[1] - p[2]))
    u[4] = kp * (V_DES - v[4]) - 0.5 * kb * max(0.0, D_REF5 - (p[2] - p[4]))
    return np.clip(u, -A_MAX, A_MAX)


def nominal_dynamics(obs, t):
    """Car 4's crude model: x+ = f(x) + g(x) u. No damping (kd=0), kp=3.5, kb=18.

    Pure function of the passed state (A6). Returns flat (15,) arrays f, g, x.
    """
    s = np.reshape(np.asarray(obs, dtype=float), (5, 3))
    p, v = s[:, 0], s[:, 1]
    a = driver_controls(p, v, t, NOM_KP, NOM_KB)   # car 4 entry is 0: no actuation in f
    f = np.zeros((5, 3))
    f[:, 0] = p + v * DT + 0.5 * a * DT ** 2
    f[:, 1] = v + a * DT
    f[:, 2] = a
    g = np.zeros((5, 3))
    g[3, 0] = 0.5 * DT ** 2
    g[3, 1] = DT
    return f.ravel(), g.ravel(), s.ravel().copy()


def soft_penalty(d):
    return 500.0 / max(d, EPS_D) if d <= D_SOFT else 0.0


class allCars():
    """Gym-like environment; step() returns (obs, reward, terminated, truncated, info)."""

    def __init__(self, init_seed=1234, noise_seed=4321, randomize=True,
                 sigma_w=SIGMA_W, episode_steps=EPISODE_STEPS):
        self.L = episode_steps
        self.sigma_w = sigma_w
        self.randomize = randomize
        self.init_rng = np.random.RandomState(init_seed)     # initial states only
        self.noise_rng = np.random.RandomState(noise_seed)   # acceleration noise only
        self.numCars = 5
        self.reset()

    def reset(self):
        self.t, self.count = 0.0, 0
        if self.randomize:
            gaps = INIT_GAP + self.init_rng.uniform(-INIT_GAP_RANGE, INIT_GAP_RANGE, size=4)
            vel = INIT_V + self.init_rng.uniform(-INIT_V_RANGE, INIT_V_RANGE, size=5)
        else:
            gaps, vel = np.full(4, INIT_GAP), np.full(5, INIT_V)
        p = np.zeros(5)
        p[4] = INIT_P_LAST
        for k in (3, 2, 1, 0):          # car k sits gaps[k] ahead of car k+1
            p[k] = p[k + 1] + gaps[k]
        self.p, self.v, self.a = p, vel, np.zeros(5)
        return self.getObs()

    def set_state(self, gaps=None, speeds=None, p_last=None):
        """Overwrite the state after reset() (used by the demo only).
        gaps: 4 values [gap12, gap23, gap34, gap45]; speeds: 5 values (car 1..5); p_last: position of car 5."""
        p = self.p.copy()
        if p_last is not None or gaps is not None:
            p_last = p[4] if p_last is None else p_last
            gaps = (p[:-1] - p[1:]) if gaps is None else np.asarray(gaps, float)
            p[4] = p_last
            for k in (3, 2, 1, 0):
                p[k] = p[k + 1] + gaps[k]
        self.p = p
        if speeds is not None:
            self.v = np.asarray(speeds, float).copy()
        return self.getObs()

    def getObs(self):
        return np.stack([self.p, self.v, self.a], axis=1).ravel().copy()

    def getAllStates(self):
        return np.stack([self.p, self.v, self.a], axis=1)

    def headways(self):
        return self.p[2] - self.p[3], self.p[3] - self.p[4]   # d_f, d_r of car 4

    def step(self, action):
        u4 = float(np.clip(np.squeeze(action), -U_MAX, U_MAX))
        u = driver_controls(self.p, self.v, self.t, KP, KB)
        u[3] = u4
        w = self.noise_rng.normal(0.0, self.sigma_w, size=5)
        a = u - KD * self.v + w
        self.p = self.p + self.v * DT + 0.5 * a * DT ** 2     # A2: old velocity
        self.v = self.v + a * DT
        self.a = a
        self.t += DT
        self.count += 1

        d_f, d_r = self.headways()
        collision = min(d_f, d_r) <= 0.0
        truncated = (self.count >= self.L) and not collision
        if collision:
            r = -(C_COLL0 + C_COLL1 * (self.L - self.count))
        else:
            r = -self.v[3] * max(a[3], 0.0) - soft_penalty(d_f) - soft_penalty(d_r)
        info = dict(d_f=d_f, d_r=d_r, min_headway=min(d_f, d_r), collision=collision,
                    accel4=a[3])
        return self.getObs(), float(r), bool(collision), bool(truncated), info
