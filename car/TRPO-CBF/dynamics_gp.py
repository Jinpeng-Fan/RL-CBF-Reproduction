"""Residual GP of car 4's nominal model (6 GPs: position and velocity of cars 3, 4, 5).

residual_j(s) = x_{j}^{+}(true) - [f_nom(s) + g_nom(s) u]_j,   j in GP_IDX

Revised vs. the original code
  * hand-written GP with FIXED hyper-parameters set from the physical size of the model
    mismatch (no sklearn marginal-likelihood fit that collapses the prior std);
  * inputs are physical relative quantities (gaps, speed deviations), not absolute positions;
  * the dataset is the latest `gp_window` (~1000) transitions (paper: "latest batch of ~1000
    points"), refit after every episode; no GP_prev / firstIter double bookkeeping;
  * the GP prior is used before the first fit; `calibrate_GP` pre-fits it (see trpo.py);
  * observation noise (std of the injected acceleration noise propagated through dt) is
    added to the posterior std, so the CBF margin covers the realised next state.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from config import *
from car_simulator import nominal_dynamics

LENGTH = np.array([3., 3., 3., 5., 5., 5.])   # [m, m, m, m/s, m/s, m/s]


def features(obs):
    """(gap23, gap34, gap45, v3-30, v4-30, v5-30)"""
    s = np.reshape(np.asarray(obs, dtype=float), (5, 3))
    p, v = s[:, 0], s[:, 1]
    return np.array([p[1] - p[2], p[2] - p[3], p[3] - p[4],
                     v[2] - V_DES, v[3] - V_DES, v[4] - V_DES])


class ResidualGPBank():
    def __init__(self, sigma_f, sigma_n):
        self.sigma_f = np.asarray(sigma_f, float)    # (6,) prior std of each residual
        self.sigma_n = np.asarray(sigma_n, float)    # (6,) observation-noise std
        self.Zn = None

    def fit(self, Z, Y):
        """Z: (N,6) features, Y: (N,6) residuals."""
        self.Zn = np.asarray(Z, float) / LENGTH
        sq = np.sum((self.Zn[:, None, :] - self.Zn[None, :, :]) ** 2, axis=-1)
        K0 = np.exp(-0.5 * sq)
        n = len(self.Zn)
        self.cho, self.alpha = [], []
        for j in range(6):
            K = self.sigma_f[j] ** 2 * K0 + self.sigma_n[j] ** 2 * np.eye(n)
            c = cho_factor(K, lower=True)
            self.cho.append(c)
            self.alpha.append(cho_solve(c, Y[:, j]))

    def predict(self, z):
        """posterior mean (6,) and latent std (6,) at feature vector z."""
        if self.Zn is None:
            return np.zeros(6), self.sigma_f.copy()
        zn = np.asarray(z, float) / LENGTH
        k0 = np.exp(-0.5 * np.sum((self.Zn - zn) ** 2, axis=1))
        mean, std = np.zeros(6), np.zeros(6)
        for j in range(6):
            k = self.sigma_f[j] ** 2 * k0
            mean[j] = k.dot(self.alpha[j])
            vv = solve_triangular(self.cho[j][0], k, lower=True)
            std[j] = np.sqrt(max(self.sigma_f[j] ** 2 - vv.dot(vv), 0.0))
        return mean, std


def build_GP_model(self):
    """Create once. Prior scales: velocity residual 0.3 m/s per step (covers kd*v*dt=0.15 and the
    kp/kb mismatch), position residual 0.5*dt times that."""
    sw = getattr(self.args, 'sigma_w', SIGMA_W)
    sf_v, sn_v = 0.3, sw * DT
    sf_p, sn_p = 0.5 * DT * sf_v, 0.5 * sw * DT ** 2
    self.gp_noise = np.array([sn_p, sn_v, sn_p, sn_v, sn_p, sn_v])
    self.GP_model = ResidualGPBank([sf_p, sf_v] * 3, self.gp_noise)
    self.gp_Z = np.empty((0, 6))
    self.gp_Y = np.empty((0, 6))
    self.gp_window = getattr(self.args, 'gp_window', 1000)
    self.gp_fit_count = 0


def update_GP_dynamics(self, path):
    """Append this episode's transitions, keep the latest window, refit."""
    obs = np.asarray(path['Observation'], dtype=float).reshape(-1, 15)
    nxt = np.asarray(path['NextObservation'], dtype=float).reshape(-1, 15)
    act = np.asarray(path['Action'], dtype=float).reshape(-1)
    if not (len(obs) == len(nxt) == len(act)) or len(obs) < 2:
        return
    Z, Y = [], []
    for i in range(len(obs)):
        f, g, _ = nominal_dynamics(obs[i], DT * i)          # t of the step (env time restarts at 0)
        Y.append((nxt[i] - f - g * act[i])[GP_IDX])
        Z.append(features(obs[i]))
    self.gp_Z = np.vstack([self.gp_Z, np.asarray(Z)])[-self.gp_window:]
    self.gp_Y = np.vstack([self.gp_Y, np.asarray(Y)])[-self.gp_window:]
    self.GP_model.fit(self.gp_Z, self.gp_Y)
    self.gp_fit_count += 1


def get_GP_dynamics(self, obs, t):
    """f = f_nom + mu, g, x, sigma (15,) = sqrt(latent std^2 + noise^2) on the modelled states."""
    f_nom, g, x = nominal_dynamics(obs, t)
    mean, std = self.GP_model.predict(features(obs))
    f = f_nom.copy()
    f[GP_IDX] += mean
    sigma = np.zeros(15)
    sigma[GP_IDX] = np.sqrt(std ** 2 + self.gp_noise ** 2)
    return f, g, x, sigma
