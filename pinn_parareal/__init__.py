"""PINN-based Parareal toolbox for the Black-Scholes equation."""

from .black_scholes import BlackScholesParams, boundary_value_right, payoff_call
from .parareal import (
    MultiprocessingFineExecutor,
    PararealIntegrator,
    MpiFineExecutor,
)
from .pinn import PINNCoarsePropagator, PINNModel, PINNTrainingConfig, train_pinn_model
from .solver import CrankNicolsonSolver

__all__ = [
    "BlackScholesParams",
    "boundary_value_right",
    "payoff_call",
    "PararealIntegrator",
    "MultiprocessingFineExecutor",
    "MpiFineExecutor",
    "PINNCoarsePropagator",
    "PINNModel",
    "PINNTrainingConfig",
    "train_pinn_model",
    "CrankNicolsonSolver",
]
