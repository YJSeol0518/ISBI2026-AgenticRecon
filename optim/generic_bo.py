"""Generic GP Bayesian optimiser over positive constants (used per organ by Ours 1).

Unchanged from the base framework's ``ConstantOptimiser``:
  1. random exploration in log space for the first ``n_initial`` proposals
  2. then a GP (anisotropic RBF; length-scale, signal and noise variance fitted by maximum
     marginal likelihood) on the observed (constants, reward) pairs, and the next point maximises
     expected improvement (multi-start L-BFGS-B)
Search bounds are [initial * 0.01, initial * 100] in log space. ``reward`` is maximised.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from scipy.stats import norm


class ConstantOptimiser:
    """
    Bayesian Optimisation for CT reconstruction constant tuning.

    Usage:
        opt = ConstantOptimiser({'rho': 0.1, 'beta': 0.01})
        next_vals = opt.propose_next(reward=-0.8, constants={'rho': 0.1, 'beta': 0.01})
        next_vals = opt.propose_next(reward=-0.5, constants=next_vals)
        ...
        print(opt.best())
    """

    def __init__(self, initial_constants: Dict[str, float],
                 n_initial: int = 3,
                 log_scale: bool = True,
                 xi: float = 0.1):
        """
        Args:
            initial_constants: {name: initial_value} for each optimisable constant
            n_initial:         Number of random exploration steps before GP kicks in
            log_scale:         Search in log-space (recommended for positive constants)
            xi:                Exploration parameter for EI acquisition (higher = more exploration)
        """
        self.names: List[str] = list(initial_constants.keys())
        self.initial: Dict[str, float] = dict(initial_constants)
        # Need at least dim points to fit GP, but keep it small so GP kicks in early.
        # max(n_initial, dim) ensures we have enough points without wasting attempts.
        self.n_initial = max(n_initial, len(self.names))
        self.log_scale = log_scale
        self.xi = xi
        self.dim = len(self.names)

        # Search bounds: [initial × 0.01, initial × 100] in original scale
        # In log-space: [log(init×0.01), log(init×100)]
        self._bounds: List[Tuple[float, float]] = []
        for name in self.names:
            v = initial_constants[name]
            if log_scale and v > 0:
                self._bounds.append((np.log(v * 0.01), np.log(v * 100)))
            else:
                self._bounds.append((v * 0.01, v * 100))

        # Observation history
        self._X: List[np.ndarray] = []   # list of parameter vectors (in search space)
        self._y: List[float] = []         # corresponding rewards

        self._best_reward: float = float('-inf')
        self._best_constants: Dict[str, float] = dict(initial_constants)
        self._attempt: int = 0

        # GP hyperparameters (length-scale per dimension, signal variance, noise)
        self._gp_ls = np.ones(self.dim)
        self._gp_sigma = 1.0
        self._gp_noise = 1e-4

        # Random state for reproducible exploration
        self._rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose_next(self, reward: float, constants: Dict[str, float]) -> Dict[str, float]:
        """Record the reward of ``constants`` and return the constants to try next."""
        # Record observation
        x = self._to_search_space(constants)
        self._X.append(x)
        self._y.append(reward)
        self._attempt += 1

        # Track best
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_constants = dict(constants)

        # Phase 1: random exploration
        if self._attempt < self.n_initial:
            next_x = self._random_sample()
        else:
            # Phase 2: GP + EI
            self._fit_gp()
            next_x = self._maximise_ei()

        return self._to_constants(next_x)

    def best(self) -> Dict[str, float]:
        """Return the best constants found so far."""
        return dict(self._best_constants)

    def summary(self) -> str:
        lines = [
            f"BayesianOptimiser summary ({self._attempt} attempts):",
            f"  Best reward   : {self._best_reward:+.4f}",
            f"  Best constants: {self._best_constants}",
        ]
        for i, (x, r) in enumerate(zip(self._X, self._y), 1):
            c = self._to_constants(x)
            lines.append(f"  [{i:2d}] reward={r:+.6f}  {c}")
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Space conversion helpers
    # ------------------------------------------------------------------

    def _to_search_space(self, constants: Dict[str, float]) -> np.ndarray:
        """Convert constant dict → search-space vector."""
        x = np.array([constants.get(n, self.initial[n]) for n in self.names],
                     dtype=float)
        if self.log_scale:
            x = np.where(x > 0, np.log(x), np.log(1e-8))
        return x

    def _to_constants(self, x: np.ndarray) -> Dict[str, float]:
        """Convert search-space vector → constant dict."""
        vals = np.exp(x) if self.log_scale else x.copy()
        # Clamp to bounds
        for i, (lo, hi) in enumerate(self._bounds):
            lo_v = np.exp(lo) if self.log_scale else lo
            hi_v = np.exp(hi) if self.log_scale else hi
            vals[i] = float(np.clip(vals[i], lo_v, hi_v))
        return {n: round(float(v), 8) for n, v in zip(self.names, vals)}

    # ------------------------------------------------------------------
    # Random exploration
    # ------------------------------------------------------------------

    def _random_sample(self) -> np.ndarray:
        """Sample uniformly in search space bounds."""
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        return lo + self._rng.rand(self.dim) * (hi - lo)

    # ------------------------------------------------------------------
    # Gaussian Process
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Anisotropic RBF (Squared Exponential) kernel.
        K(x, x') = sigma² · exp(-0.5 · Σ_d ((x_d - x'_d) / ls_d)²)
        """
        # X1: (n, d), X2: (m, d) → (n, m)
        diff = X1[:, None, :] - X2[None, :, :]          # (n, m, d)
        scaled = diff / self._gp_ls[None, None, :]       # (n, m, d)
        sq_dist = np.sum(scaled ** 2, axis=-1)            # (n, m)
        return self._gp_sigma ** 2 * np.exp(-0.5 * sq_dist)

    def _fit_gp(self):
        """Optimise GP hyperparameters via log-marginal-likelihood."""
        X = np.array(self._X)   # (n, d)
        y = np.array(self._y)   # (n,)

        def neg_log_ml(params):
            ls = np.exp(params[:self.dim])
            sigma = np.exp(params[self.dim])
            noise = np.exp(params[self.dim + 1])

            self._gp_ls = ls
            self._gp_sigma = sigma
            self._gp_noise = noise

            K = self._rbf_kernel(X, X) + noise * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K + 1e-6 * np.eye(len(X)))
            except np.linalg.LinAlgError:
                return 1e10

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_ml = (
                -0.5 * y @ alpha
                - np.sum(np.log(np.diag(L)))
                - 0.5 * len(X) * np.log(2 * np.pi)
            )
            return -log_ml

        x0 = np.concatenate([
            np.log(self._gp_ls),
            [np.log(self._gp_sigma)],
            [np.log(self._gp_noise)]
        ])
        try:
            res = minimize(neg_log_ml, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-6})
            # Apply best found params
            neg_log_ml(res.x)
        except Exception:
            pass  # keep previous hyperparameters on failure

    def _gp_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and std at X_test."""
        K_tt = self._rbf_kernel(X_train, X_train) + self._gp_noise * np.eye(len(X_train))
        K_ts = self._rbf_kernel(X_train, X_test)   # (n_train, n_test)
        K_ss = self._rbf_kernel(X_test, X_test)

        try:
            L = np.linalg.cholesky(K_tt + 1e-6 * np.eye(len(X_train)))
        except np.linalg.LinAlgError:
            return np.zeros(len(X_test)), np.ones(len(X_test))

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_ts.T @ alpha

        v = np.linalg.solve(L, K_ts)
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 1e-10))
        return mu, std

    # ------------------------------------------------------------------
    # Expected Improvement acquisition
    # ------------------------------------------------------------------

    def _expected_improvement(self, x: np.ndarray,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               xi: float = 0.01) -> float:
        """
        EI(x) = E[max(f(x) - f_best - xi, 0)]
               = (mu - f_best - xi) · Φ(Z) + sigma · φ(Z)
        where Z = (mu - f_best - xi) / sigma
        """
        x = x.reshape(1, -1)
        mu, sigma = self._gp_predict(X_train, y_train, x)
        mu, sigma = mu[0], sigma[0]

        f_best = np.max(y_train)
        improvement = mu - f_best - xi
        Z = improvement / (sigma + 1e-9)
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return float(ei)

    def _maximise_ei(self) -> np.ndarray:
        """
        Find x* = argmax EI(x) via multi-start L-BFGS-B.
        Uses min(10, 5 * dim) random restarts (5 for a single constant).
        """
        X_train = np.array(self._X)
        y_train = np.array(self._y)

        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        bounds = list(zip(lo, hi))

        best_ei = -np.inf
        best_x = self._random_sample()

        n_restarts = min(10, 5 * self.dim)
        starts = [lo + self._rng.rand(self.dim) * (hi - lo)
                  for _ in range(n_restarts)]

        for x0 in starts:
            try:
                res = minimize(
                    lambda x: -self._expected_improvement(x, X_train, y_train, xi=self.xi),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                ei = -res.fun
                if ei > best_ei:
                    best_ei = ei
                    best_x = res.x
            except Exception:
                continue

        return best_x
