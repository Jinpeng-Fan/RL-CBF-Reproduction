"""Episode-wise control-affine residual GP: d(s,u) = d_f(s) + d_g(s)*u.

Compound kernel k((s,u),(s',u')) = k_f(s,s') + u*k_g(s,s')*u', one GP per
state dimension, fixed hyperparameters. The posterior over (d_f, d_g) at a
query state is jointly Gaussian, so the predicted residual is affine in u
(mean) and its variance is the quadratic form [1,u] Sigma [1,u]^T.
"""
import numpy as np

DT, GRAVITY, MASS_NOM, LENGTH_NOM = 0.05, 10.0, 1.4, 1.4
# Prior scales relative to nominal terms; lengthscales over (theta, theta_dot).
# kd*GAIN_SCALE must stay < 1, else the prior leaves the sign of the input
# gain uncertain and the robust CBF term cancels any torque benefit.
DRIFT_SCALE, GAIN_SCALE, NOISE_SCALE = 0.3, 0.5, 0.02
LENGTH_F = np.array([1.0, 50.0])
LENGTH_G = np.array([10.0, 50.0])


def _se(A, B, sigma, lengths):
    d = (A[:, None, :] - B[None, :, :]) / lengths
    return sigma ** 2 * np.exp(-0.5 * np.sum(d ** 2, axis=-1))


class CompoundGP():
    def __init__(self, sigma_f, sigma_g, sigma_noise):
        self.sigma_f, self.sigma_g, self.sigma_noise = sigma_f, sigma_g, sigma_noise
        self.X = None

    def fit(self, X, U, y):
        self.X, self.U = np.asarray(X, float), np.asarray(U, float).reshape(-1)
        K = (_se(self.X, self.X, self.sigma_f, LENGTH_F)
             + np.outer(self.U, self.U) * _se(self.X, self.X, self.sigma_g, LENGTH_G)
             + self.sigma_noise ** 2 * np.eye(len(self.X)))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.asarray(y, float)))

    def predict(self, x):
        """Posterior mean (2,) and covariance (2,2) of (d_f(x), d_g(x))."""
        prior = np.diag([self.sigma_f ** 2, self.sigma_g ** 2])
        if self.X is None:
            return np.zeros(2), prior
        x = np.asarray(x, float).reshape(1, -1)
        A = np.column_stack([_se(self.X, x, self.sigma_f, LENGTH_F)[:, 0],
                             self.U * _se(self.X, x, self.sigma_g, LENGTH_G)[:, 0]])
        V = np.linalg.solve(self.L, A)
        cov = prior - V.T.dot(V)
        return A.T.dot(self.alpha), 0.5 * (cov + cov.T)


def build_GP_model(self):
    """Initialize once, not at each episode."""
    _, g, _ = get_dynamics(self, np.array([1., 0., 0.]), 0.0)
    drift = 3 * GRAVITY / (2 * LENGTH_NOM) * np.array([DT ** 2, DT])
    self.GP_model = [CompoundGP(DRIFT_SCALE * drift[j], GAIN_SCALE * g[j],
                                NOISE_SCALE * DRIFT_SCALE * drift[j]) for j in range(2)]
    self.gp_X = np.empty((0, 2))
    self.gp_Y = np.empty((0, 2))
    self.gp_fit_count = 0


def state_from_obs(obs):
    obs = np.asarray(obs, dtype=float).reshape(3)
    return np.array([np.arctan2(obs[1], obs[0]), obs[2]])


def get_dynamics(self, obs, u_rl):
    """Full nominal next state F_nom(x,u), input gain g, and x."""
    dt, gravity, mass, length = DT, GRAVITY, MASS_NOM, LENGTH_NOM
    x = state_from_obs(obs)
    u = float(np.asarray(u_rl).reshape(-1)[0])
    acceleration = 3 * gravity / (2 * length) * np.sin(x[0])
    g = 3 / (mass * length ** 2) * np.array([dt ** 2, dt])
    f0 = np.array([x[0] + x[1] * dt + acceleration * dt ** 2,
                   x[1] + acceleration * dt])
    return f0 + g * u, g, x


def update_GP_dynamics(self, path):
    """Refit using every aligned transition from this episode only."""
    obs = np.asarray(path['Observation'], dtype=float).reshape(-1, 3)
    nxt = np.asarray(path['NextObservation'], dtype=float).reshape(-1, 3)
    actions = np.asarray(path['Action'], dtype=float).reshape(-1)
    if not (len(obs) == len(nxt) == len(actions)) or not len(obs):
        raise ValueError('GP needs equally sized, nonempty (obs, action, next_obs).')
    states, errors = [], []
    for before, u, after in zip(obs, actions, nxt):
        prediction, _, x = get_dynamics(self, before, u)
        error = state_from_obs(after) - prediction
        error[0] = (error[0] + np.pi) % (2 * np.pi) - np.pi
        states.append(x)
        errors.append(error)
    # Replace, rather than append: no samples from earlier episodes survive.
    self.gp_X = np.asarray(states, dtype=float)
    self.gp_Y = np.asarray(errors, dtype=float)
    for j, gp in enumerate(self.GP_model):
        gp.fit(self.gp_X, actions, self.gp_Y[:, j])
    self.gp_fit_count += 1


def get_GP_dynamics(self, obs, u_rl=None):
    """Return f_nom(x)+mu_f(x), g_nom+mu_g(x), x, Sigma; Sigma[j] is the 2x2
    posterior covariance of (d_f, d_g) for state dimension j.

    u_rl is accepted for compatibility. The CBF adds g*u exactly once and
    evaluates std_j(u) = sqrt([1,u] Sigma[j] [1,u]^T) at the executed u.
    Before the first fit, the GP prior is returned.
    """
    f0, g, x = get_dynamics(self, obs, 0.0)
    means, covs = zip(*[gp.predict(x) for gp in self.GP_model])
    means = np.asarray(means)
    return f0 + means[:, 0], g + means[:, 1], x, np.asarray(covs)


def predictive_std(Sigma, u):
    """Per-dimension residual std at input u from (2,2,2) covariance."""
    z = np.array([1., float(u)])
    return np.sqrt(np.maximum(np.einsum('i,jik,k->j', z, Sigma, z), 0.))
