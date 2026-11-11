"""Paper Eq. (16) with a control-affine GP residual.

Row i of the barrier condition, for executed torque u = v + a, is
    b_i - A_i*u + kd*s_i(u) <= epsilon,
with A_i = H_i.(g+mu_g), s_i(u) = |H_i|.std(u) and std_j(u)^2 = [1,u] Sigma_j [1,u]^T.
s_i is convex in u, so phi(u) = max_i(...) is convex (a second-order-cone
constraint). With the large slack penalty K, min |a| + K*epsilon is solved
lexicographically: first minimal epsilon = max(0, min phi), then minimal |a|.
"""
import numpy as np
import dynamics_gp

GOLDEN = (np.sqrt(5.) - 1) / 2


def build_barrier(self):
    self.H = np.array([[1., 0.], [-1., 0.]])
    self.F = 1.0
    self.barrier_eta = getattr(self, 'barrier_eta', 0.5)
    self.barrier_kd = getattr(self, 'barrier_kd', 1.5)
    self.slack_penalty = getattr(self, 'slack_penalty', 1e12)
    self.enforce_speed_limit = getattr(self, 'enforce_speed_limit', False)


def _minimize_convex(phi, low, high, iterations=100):
    a, b = low, high
    for _ in range(iterations):
        c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
        if phi(c) <= phi(d):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


def _boundary(phi, inside, outside, iterations=100):
    """Point of {phi <= 0} closest to `outside`, given phi(inside) <= 0."""
    for _ in range(iterations):
        middle = 0.5 * (inside + outside)
        if phi(middle) <= 0:
            inside = middle
        else:
            outside = middle
    return inside


def control_barrier(self, obs, u_rl, f, g, x, Sigma):
    """Filter the sampled candidate; return the correction only."""
    v = float(np.asarray(u_rl).reshape(-1)[0])
    f, g, x = [np.asarray(z, float).reshape(2) for z in (f, g, x)]
    H, eta, kd = self.H, self.barrier_eta, self.barrier_kd
    b = (1 - eta) * (H.dot(x) + self.F) - (H.dot(f) + self.F)
    A = H.dot(g)

    def violation(u):
        return b - A * u + kd * np.abs(H).dot(dynamics_gp.predictive_std(Sigma, u))

    def phi(u):
        return float(np.max(violation(u)))

    low, high = -self.torque_bound, self.torque_bound
    if self.enforce_speed_limit:
        # Optional mean speed bound, absent from paper Eq. (16).
        limits = sorted([(-self.max_speed - f[1]) / g[1], (self.max_speed - f[1]) / g[1]])
        low, high = max(low, limits[0]), min(high, limits[1])
        if low > high:
            raise ValueError('Speed bound infeasible within torque limits.')
    if not np.all(np.isfinite(np.r_[b, A, v, Sigma.ravel()])):
        raise ValueError('Nonfinite CBF data.')

    target = min(max(v, low), high)
    if phi(target) <= 0:
        executed, epsilon = target, 0.
    else:
        best = _minimize_convex(phi, low, high)
        if phi(best) > 0:
            executed, epsilon = best, phi(best)
        else:
            executed, epsilon = _boundary(phi, best, target), 0.
    a = executed - v
    std = dynamics_gp.predictive_std(Sigma, executed)
    margin = violation(executed)
    self.last_cbf = dict(slack=epsilon, correction=a, executed=executed,
                         min_robust_margin=float(np.min(epsilon - margin)),
                         objective=abs(a) + self.slack_penalty * epsilon)
    self.last_cbf['unrelaxed_margin'] = float(np.min(-margin))
    self.last_cbf['predicted_next_state'] = (f + g * executed).copy()
    self.last_cbf['sigma'] = std.copy()
    self.last_cbf['state'] = x.copy()
    self.last_cbf['confidence_multiplier'] = kd
    if (not np.isfinite(epsilon) or epsilon < 0 or
            abs(executed) > self.torque_bound + 1e-7 or
            self.last_cbf['min_robust_margin'] < -1e-7):
        raise FloatingPointError('CBF solver returned an invalid constrained action.')
    return np.array([a])
