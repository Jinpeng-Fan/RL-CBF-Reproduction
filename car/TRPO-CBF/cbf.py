"""Second-order discrete-time CBF (paper eq. 9/16 with an affine, velocity-aware h).

For each neighbour pair (front f, rear b) of car 4
    h(s) = (v_f - v_b) + alpha*(p_f - p_b - D_MIN)           (affine in s: h = P s + q)
and the condition on the NEXT state
    h(s+) >= (1-eta)*h(s)  with  s+ = f_gp(s) + g*u,  robustified by  kd*|P|*sigma.
Position-only barriers (the original H) give the input a gain of 0.5*dt^2 only; here u enters
through the velocity with gain dt.

One scalar input and two rows => the QP  min |u_bar|^2 + K*eps  has a closed form:
  * feasible: u = clip(candidate, lo, hi) (smallest correction);
  * infeasible: minimise the worst violation eps (convex piecewise-linear in u), reported as slack.
"""
import numpy as np
from config import *


def build_barrier(self):
    a = getattr(self.args, 'cbf_alpha', 2.0)
    P = np.zeros((2, 15))
    P[0, 6], P[0, 7], P[0, 9], P[0, 10] = a, 1., -a, -1.      # car 3 (front) - car 4
    P[1, 9], P[1, 10], P[1, 12], P[1, 13] = a, 1., -a, -1.    # car 4 - car 5 (rear)
    self.P = P
    self.q = np.array([-a * D_MIN, -a * D_MIN])
    self.barrier_eta = getattr(self.args, 'cbf_eta', 0.2)
    self.barrier_kd = getattr(self.args, 'cbf_kappa', 2.0)
    self.cbf_u_max = U_MAX
    self.last_cbf = dict(slack=0.0, infeasible=False, executed=0.0, margin=0.0)


def barrier_value(self, obs):
    return self.P.dot(np.asarray(obs, float).ravel()) + self.q


def control_barrier(self, obs, u_cand, f, g, x, sigma):
    """Return the correction u_bar = executed - candidate (shape (1,))."""
    v = float(np.asarray(u_cand).reshape(-1)[0])
    P, q, eta, kd = self.P, self.q, self.barrier_eta, self.barrier_kd
    h_now = P.dot(x) + q
    b = P.dot(f) + q - kd * np.abs(P).dot(sigma) - (1 - eta) * h_now     # margin at u = 0
    A = P.dot(g)                                                         # input gain per row
    lo_c, hi_c = -np.inf, np.inf
    for i in range(len(b)):
        if A[i] > 1e-12:
            lo_c = max(lo_c, -b[i] / A[i])
        elif A[i] < -1e-12:
            hi_c = min(hi_c, -b[i] / A[i])
        elif b[i] < 0:
            lo_c, hi_c = np.inf, -np.inf                                 # unactuated & violated
    U = self.cbf_u_max
    lo, hi = max(lo_c, -U), min(hi_c, U)
    if lo <= hi:
        executed, eps = min(max(v, lo), hi), 0.0
    else:
        def worst(u):
            return float(np.max(-(b + A * u)))
        cands = [-U, U]
        den = A[0] - A[1]
        if abs(den) > 1e-12:
            cands.append(min(max((b[1] - b[0]) / den, -U), U))          # rows cross
        executed = min(cands, key=lambda u: (round(worst(u), 9), abs(u - v)))
        eps = max(worst(executed), 0.0)
    margin = float(np.min(b + A * executed))
    self.last_cbf = dict(slack=eps, infeasible=eps > 0, executed=executed, margin=margin,
                         h_now=h_now.copy(), predicted_next=f + g * executed)
    return np.array([executed - v])
