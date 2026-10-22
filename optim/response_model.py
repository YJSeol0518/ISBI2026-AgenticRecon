"""Feature-resolved response model (Ours 2).

For one organ, 20 independent GPs over u = log(beta), one per Haralick feature f, each modelling

    y_f(u) = log(r_f + eps),   r_f = (chi2_f(x(beta)) + eps) / (chi2_f(x_LDCT) + eps)

with a shared, fixed RBF kernel (length scale 0.55 of the log-range, unit signal variance,
noise 1e-3) and a constant mean equal to the mean of the observed y_f.

The 20 posterior means are reduced to the feature-balanced score

    J(y) = mean_f y_f + lambda * smoothmax_tau(y),     lambda = 0.25, tau = 8

and the next candidate minimises  J(mu(u)) - kappa * mean_f sigma_f(u)   (kappa = 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class ResponseModel:
    """1-D GP over log(beta) with a constant mean (mean of the observations)."""
    bounds: Tuple[float, float]          # (lo, hi) in log(beta)
    length_scale: float = 0.55           # as a fraction of hi - lo
    noise: float = 1e-3
    X: List[float] = field(default_factory=list)
    Y: List[float] = field(default_factory=list)

    def add(self, log_beta: float, y: float) -> None:
        self.X.append(float(log_beta))
        self.Y.append(float(y))

    @property
    def n_probes(self) -> int:
        return len(self.X)

    def _k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        span = (self.bounds[1] - self.bounds[0]) + 1e-12
        return np.exp(-0.5 * ((a[:, None] - b[None, :]) / span) ** 2 / self.length_scale ** 2)

    def predict(self, log_beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean / std. With fewer than 2 probes: the prior (mean of Y, std 1)."""
        log_beta = np.asarray(log_beta, dtype=float)
        m = float(np.mean(self.Y)) if self.Y else 0.0
        if self.n_probes < 2:
            return np.full_like(log_beta, m), np.ones_like(log_beta)
        Xs, y = np.array(self.X), np.array(self.Y)
        L = np.linalg.cholesky(self._k(Xs, Xs) + self.noise * np.eye(len(Xs)))
        a = np.linalg.solve(L.T, np.linalg.solve(L, y - m))
        Ks = self._k(Xs, log_beta)
        mu = m + Ks.T @ a
        v = np.linalg.solve(L, Ks)
        return mu, np.sqrt(np.clip(1.0 - np.sum(v ** 2, 0), 1e-9, None))


def smooth_max(v: np.ndarray, tau: float = 8.0) -> float:
    """Log-sum-exp smooth maximum: max(v) + log(mean exp(tau (v - max v))) / tau."""
    v = np.asarray(v, dtype=np.float64)
    m = float(np.max(v))
    return m + float(np.log(np.mean(np.exp(tau * (v - m))))) / tau


@dataclass
class FeatureResponseModel:
    bounds: Tuple[float, float]
    feature_names: Sequence[str]
    length_scale: float = 0.55
    noise: float = 1e-3
    eps: float = 1e-6
    models: Dict[str, ResponseModel] = field(default_factory=dict)

    def __post_init__(self):
        if not self.models:
            self.models = {f: ResponseModel(self.bounds, self.length_scale, self.noise)
                           for f in self.feature_names}

    @property
    def probed(self) -> List[float]:
        return list(next(iter(self.models.values())).X)

    @property
    def n_probes(self) -> int:
        return len(self.probed)

    def add(self, log_beta: float, r_vec: Sequence[float]) -> None:
        """``r_vec``: LDCT-normalised Pearson divergence per feature (not log-transformed)."""
        for name, r in zip(self.feature_names, r_vec):
            self.models[name].add(log_beta, float(np.log(max(r, 0.0) + self.eps)))

    def predict_all(self, log_beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(mu, sd), each (len(log_beta), n_features), in log(r + eps) space."""
        log_beta = np.asarray(log_beta, dtype=float)
        mu = np.zeros((len(log_beta), len(self.feature_names)))
        sd = np.zeros_like(mu)
        for j, name in enumerate(self.feature_names):
            mu[:, j], sd[:, j] = self.models[name].predict(log_beta)
        return mu, sd

    @staticmethod
    def objective(y_row: np.ndarray, lambda_worst: float) -> float:
        return float(np.mean(y_row) + lambda_worst * smooth_max(y_row))

    def observed_objective(self, r_vec: Sequence[float], lambda_worst: float) -> float:
        row = np.log(np.maximum(np.asarray(r_vec, dtype=float), 0.0) + self.eps)
        return self.objective(row, lambda_worst)

    def propose_next(self, candidates_log_beta: Sequence[float], lambda_worst: float = 0.25,
                     kappa: float = 1.0) -> int:
        """Index of the next candidate. With fewer than 2 probes there is no GP yet: the lowest
        candidate if nothing was probed, otherwise the candidate farthest from all probes."""
        c = np.asarray(candidates_log_beta, dtype=float)
        if self.n_probes < 2:
            probed = np.array(self.probed)
            if probed.size == 0:
                return int(np.argmin(c))
            return int(np.argmax(np.min(np.abs(c[:, None] - probed[None, :]), axis=1)))
        mu, sd = self.predict_all(c)
        pred_J = np.array([self.objective(mu[i], lambda_worst) for i in range(len(c))])
        return int(np.argmin(pred_J - kappa * sd.mean(axis=1)))

    def prediction_at(self, log_beta: float) -> Tuple[np.ndarray, np.ndarray]:
        mu, sd = self.predict_all(np.array([log_beta]))
        return mu[0], sd[0]
