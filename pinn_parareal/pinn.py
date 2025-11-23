from dataclasses import dataclass
from typing import Iterable, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from .black_scholes import BlackScholesParams


@dataclass
class PINNTrainingConfig:
    interior_points: int = 100_000
    boundary_points: int = 10_000
    initial_points: int = 10_000
    stage1_epochs: int = 5_000
    stage2_epochs: int = 800
    lr_stage1: float = 1e-2
    lr_stage2: float = 1e-3
    logging_interval: int = 500


class PINNModel(nn.Module):
    """Fully connected neural network representing the PDE solution."""

    def __init__(self, hidden_layers: int = 10, hidden_size: int = 50):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_size), nn.ReLU()]  # input layer
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
        self._initialize()

    def _initialize(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, tau: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([tau, s], dim=-1)
        return self.net(x)


class PINNCollocationData(NamedTuple):
    tau_interior: torch.Tensor
    s_interior: torch.Tensor
    tau_boundary: torch.Tensor
    s_left: torch.Tensor
    s_right: torch.Tensor
    tau_initial: torch.Tensor
    s_initial: torch.Tensor

def _permute_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.size(0) <= 1:
        return tensor
    return tensor[torch.randperm(tensor.size(0), device=tensor.device)]


def _compute_pde_loss(
    model: PINNModel,
    params: BlackScholesParams,
    data: PINNCollocationData,
):
    tau_interior = _permute_tensor(data.tau_interior).detach().requires_grad_(True)
    s_interior = _permute_tensor(data.s_interior).detach().requires_grad_(True)
    u_pred = model(tau_interior, s_interior)
    du_dtau = torch.autograd.grad(u_pred, tau_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    du_ds = torch.autograd.grad(u_pred, s_interior, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    d2u_ds2 = torch.autograd.grad(du_ds, s_interior, grad_outputs=torch.ones_like(du_ds), create_graph=True)[0]
    a = 0.5 * params.sigma ** 2 * s_interior ** 2
    b = params.r * s_interior
    c = -params.r
    residual = du_dtau - (a * d2u_ds2 + b * du_ds + c * u_pred)
    return residual.pow(2).mean()


def _compute_boundary_loss(model: PINNModel, params: BlackScholesParams, data: PINNCollocationData):
    tau_boundary = _permute_tensor(data.tau_boundary)
    s_left = data.s_left
    s_right = data.s_right
    u_left = model(tau_boundary, s_left)
    u_right = model(tau_boundary, s_right)
    target_right = params.s_max - params.strike * torch.exp(-params.r * tau_boundary)
    left_loss = u_left.pow(2).mean()
    right_loss = (u_right - target_right).pow(2).mean()
    return left_loss + right_loss


def _compute_initial_loss(model: PINNModel, params: BlackScholesParams, data: PINNCollocationData):
    tau_initial = data.tau_initial  # already zeros
    s_initial = _permute_tensor(data.s_initial)
    u_initial = model(tau_initial, s_initial)
    payoff = torch.maximum(s_initial - params.strike, torch.zeros_like(s_initial))
    return (u_initial - payoff).pow(2).mean()


def _build_collocation_data(params: BlackScholesParams, config: PINNTrainingConfig, device: torch.device) -> PINNCollocationData:
    tau_interior = torch.rand(config.interior_points, 1, device=device) * params.maturity
    s_interior = torch.rand(config.interior_points, 1, device=device) * (params.s_max - 1e-3) + 1e-3
    tau_boundary = torch.rand(config.boundary_points, 1, device=device) * params.maturity
    s_left = torch.zeros_like(tau_boundary)
    s_right = torch.full_like(tau_boundary, params.s_max)
    tau_initial = torch.zeros(config.initial_points, 1, device=device)
    s_initial = torch.rand(config.initial_points, 1, device=device) * params.s_max
    return PINNCollocationData(
        tau_interior=tau_interior,
        s_interior=s_interior,
        tau_boundary=tau_boundary,
        s_left=s_left,
        s_right=s_right,
        tau_initial=tau_initial,
        s_initial=s_initial,
    )


def train_pinn_model(
    model: PINNModel,
    params: BlackScholesParams,
    config: PINNTrainingConfig,
    device: torch.device,
) -> PINNModel:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_stage1)
    stages = [
        (config.stage1_epochs, config.lr_stage1),
        (config.stage2_epochs, config.lr_stage2),
    ]
    collocation_data = _build_collocation_data(params, config, device)
    for stage_idx, (epochs, lr) in enumerate(stages, start=1):
        if epochs == 0:
            continue
        optimizer.param_groups[0]["lr"] = lr
        for epoch in range(1, epochs + 1):
            pde_loss = _compute_pde_loss(model, params, collocation_data)
            boundary_loss = _compute_boundary_loss(model, params, collocation_data)
            initial_loss = _compute_initial_loss(model, params, collocation_data)
            loss = pde_loss + boundary_loss + initial_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % config.logging_interval == 0 or (epoch == 1 and stage_idx == 1):
                print(
                    f"Stage {stage_idx} epoch {epoch}/{epochs}: "
                    f"total={loss.item():.3e}, pde={pde_loss.item():.3e}, "
                    f"boundary={boundary_loss.item():.3e}, init={initial_loss.item():.3e}"
                )
    return model


class PINNCoarsePropagator:
    def __init__(self, model: PINNModel, s_grid: Iterable[float], device: torch.device):
        self.model = model
        self.device = device
        self.s_tensor = torch.tensor(list(s_grid), dtype=torch.float32, device=device).unsqueeze(1)

    def _evaluate(self, tau: float) -> np.ndarray:
        tau_tensor = torch.full((self.s_tensor.shape[0], 1), float(tau), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tau_tensor, self.s_tensor).squeeze(-1)
        return output.cpu().numpy()

    def __call__(self, u_init: np.ndarray, tau_start: float, tau_end: float) -> np.ndarray:
        pred_start = self._evaluate(tau_start)
        pred_end = self._evaluate(tau_end)
        delta = pred_end - pred_start
        return np.array(u_init, dtype=np.float64) + delta
