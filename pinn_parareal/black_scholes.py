from dataclasses import dataclass

import numpy as np


@dataclass
class BlackScholesParams:
    r: float = 0.03
    sigma: float = 0.4
    strike: float = 100.0
    maturity: float = 1.0
    s_max: float = 5_000.0


def payoff_call(s: np.ndarray, strike: float) -> np.ndarray:
    """European call payoff at the start of the pricing period."""
    return np.maximum(s - strike, 0.0)


def boundary_value_right(s_max: float, strike: float, r: float, tau: float) -> float:
    """Dirichlet value at the artificial upper boundary S = s_max for time tau."""
    return float(max(s_max - strike * np.exp(-r * tau), 0.0))
