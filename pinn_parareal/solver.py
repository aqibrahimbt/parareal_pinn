import math

import numpy as np

from .black_scholes import BlackScholesParams, boundary_value_right, payoff_call


class CrankNicolsonSolver:
    """Finite-difference Crank-Nicolson solver for the transformed Black-Scholes PDE."""

    def __init__(self, params: BlackScholesParams, spatial_steps: int = 200):
        self.params = params
        self.spatial_steps = spatial_steps
        self.delta_s = params.s_max / spatial_steps
        self.s_grid = np.linspace(0.0, params.s_max, spatial_steps + 1)
        self.initial_state = payoff_call(self.s_grid, params.strike)

    def _boundary_left(self, _: float) -> float:
        return 0.0

    def _boundary_right(self, tau: float) -> float:
        return boundary_value_right(self.params.s_max, self.params.strike, self.params.r, tau)

    def _build_coefficients(self, dt: float):
        sigma = self.params.sigma
        r = self.params.r
        s = self.s_grid[1:-1]
        alpha = 0.5 * sigma ** 2 * s ** 2 / self.delta_s ** 2
        beta = r * s / (2.0 * self.delta_s)
        center = 1.0 + 0.5 * dt * (2.0 * alpha + r)
        left = -0.5 * dt * (alpha - beta)
        right = -0.5 * dt * (alpha + beta)
        center_b = 1.0 - 0.5 * dt * (2.0 * alpha + r)
        left_b = 0.5 * dt * (alpha - beta)
        right_b = 0.5 * dt * (alpha + beta)
        return left, center, right, left_b, center_b, right_b

    def _solve_tridiagonal(self, lower, diag, upper, rhs):
        n = len(diag)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([rhs[0] / diag[0]])
        cp = np.empty(n - 1)
        dp = np.empty(n)
        cp[0] = upper[0] / diag[0]
        dp[0] = rhs[0] / diag[0]
        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * cp[i - 1]
            cp[i] = upper[i] / denom
            dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom
        denom = diag[-1] - lower[-1] * cp[-1]
        dp[-1] = (rhs[-1] - lower[-1] * dp[-2]) / denom
        solution = np.empty(n)
        solution[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            solution[i] = dp[i] - cp[i] * solution[i + 1]
        return solution

    def _step(self, u: np.ndarray, tau: float, dt: float) -> np.ndarray:
        left, center, right, left_b, center_b, right_b = self._build_coefficients(dt)
        interior = u[1:-1]
        left_auc = left[1:]
        right_auc = right[:-1]
        left_b_bc = left_b[1:]
        right_b_bc = right_b[:-1]
        diag_A = center
        diag_B = center_b
        rhs = diag_B * interior
        if interior.size > 1:
            rhs[1:] += left_b_bc * interior[:-1]
            rhs[:-1] += right_b_bc * interior[1:]
        u0_current = self._boundary_left(tau)
        uN_current = self._boundary_right(tau)
        u0_next = self._boundary_left(tau + dt)
        uN_next = self._boundary_right(tau + dt)
        rhs[0] += left_b[0] * u0_current
        rhs[-1] += right_b[-1] * uN_current
        rhs[0] -= left[0] * u0_next
        rhs[-1] -= right[-1] * uN_next
        lower_diag = left_auc
        upper_diag = right_auc
        interior_next = self._solve_tridiagonal(lower_diag, diag_A, upper_diag, rhs)
        result = u.copy()
        result[1:-1] = interior_next
        result[0] = u0_next
        result[-1] = uN_next
        return result

    def propagate(self, u_init: np.ndarray, tau_start: float, tau_end: float, steps: int) -> np.ndarray:
        tau = tau_start
        u = u_init.copy()
        dt = (tau_end - tau_start) / max(steps, 1)
        for _ in range(steps):
            u = self._step(u, tau, dt)
            tau += dt
        return u

    def solve_full(self, steps: int) -> np.ndarray:
        return self.propagate(self.initial_state, 0.0, self.params.maturity, steps)
