"""Entrypoint for PINN-based Parareal experiments on the Black-Scholes PDE."""

import argparse
from pathlib import Path

import numpy as np
import torch

from pinn_parareal import (
    BlackScholesParams,
    CrankNicolsonSolver,
    MpiFineExecutor,
    MultiprocessingFineExecutor,
    PararealIntegrator,
    PINNCoarsePropagator,
    PINNModel,
    PINNTrainingConfig,
    train_pinn_model,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_pinn(args: argparse.Namespace) -> None:
    device = _device()
    params = BlackScholesParams()
    model = PINNModel(hidden_layers=args.hidden_layers, hidden_size=args.hidden_size)
    config = PINNTrainingConfig(
        interior_points=args.interior_points,
        boundary_points=args.boundary_points,
        initial_points=args.initial_points,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        logging_interval=args.log_interval,
    )
    print(f"Training PINN on {device} (hidden={args.hidden_layers}x{args.hidden_size})")
    trained = train_pinn_model(model, params, config, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), output_path)
    print(f"Saved trained PINN to {output_path}")


def run_parareal(args: argparse.Namespace) -> None:
    device = _device()
    params = BlackScholesParams()
    solver = CrankNicolsonSolver(params, spatial_steps=args.spatial_steps)
    model = PINNModel(hidden_layers=args.hidden_layers, hidden_size=args.hidden_size)
    checkpoint = Path(args.model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"PINN checkpoint not found at {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    coarse = PINNCoarsePropagator(model, solver.s_grid, device)

    fine_propagator = solver.propagate
    parallel_executor = None
    if args.parallel_mode == "multiprocessing":
        parallel_executor = MultiprocessingFineExecutor(
            fine_func=fine_propagator,
            max_workers=args.workers,
        )
    elif args.parallel_mode == "mpi":
        parallel_executor = MpiFineExecutor(
            fine_func=fine_propagator,
            max_workers=args.workers,
        )

    time_grid = np.linspace(0.0, params.maturity, args.time_slices + 1)
    parareal = PararealIntegrator(
        time_points=time_grid,
        initial_state=solver.initial_state,
        coarse=coarse,
        fine=fine_propagator,
        fine_steps_per_slice=args.fine_steps,
        parallel_fine=parallel_executor,
    )
    solutions, errors = parareal.run(iterations=args.iterations)
    for idx, err in enumerate(errors):
        print(f"Iteration {idx}: relative final error = {err:.3e}")
    coarse_time = parareal.metrics["coarse_secs"]
    fine_time = parareal.metrics["fine_secs"]
    print(f"Coarse total time: {coarse_time:.3f}s, fine total time: {fine_time:.3f}s")
    if fine_time > 0:
        print(f"Coarse-to-fine time ratio: {coarse_time / fine_time:.2f}")
    print("Parareal history shape:", solutions[-1].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PINN and run Parareal for Black-Scholes pricing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trainer = subparsers.add_parser("train-pinn", help="Train a PINN coarse propagator.")
    trainer.add_argument("--output", default="pinn_model.pth", help="Path to save trained PINN state dict.")
    trainer.add_argument("--hidden-layers", type=int, default=10, help="Number of hidden layers.")
    trainer.add_argument("--hidden-size", type=int, default=50, help="Neurons per hidden layer.")
    trainer.add_argument("--interior-points", type=int, default=100_000)
    trainer.add_argument("--boundary-points", type=int, default=10_000)
    trainer.add_argument("--initial-points", type=int, default=10_000)
    trainer.add_argument("--stage1-epochs", type=int, default=5_000)
    trainer.add_argument("--stage2-epochs", type=int, default=800)
    trainer.add_argument("--lr-stage1", type=float, default=1e-2)
    trainer.add_argument("--lr-stage2", type=float, default=1e-3)
    trainer.add_argument("--log-interval", type=int, default=500)

    runner = subparsers.add_parser("run-parareal", help="Run Parareal with a trained PINN coarse prop.")
    runner.add_argument("--model-path", default="pinn_model.pth", help="Checkpoint of trained PINN.")
    runner.add_argument("--spatial-steps", type=int, default=200, help="Spatial discretization points for finite differences.")
    runner.add_argument("--time-slices", type=int, default=16, help="Number of Parareal time slices.")
    runner.add_argument("--fine-steps", type=int, default=20, help="Fine solver steps per slice.")
    runner.add_argument("--iterations", type=int, default=3, help="Parareal iterations to run.")
    runner.add_argument("--hidden-layers", type=int, default=10)
    runner.add_argument("--hidden-size", type=int, default=50)
    runner.add_argument(
        "--parallel-mode",
        choices=("serial", "multiprocessing", "mpi"),
        default="serial",
        help="Parallelization strategy for the fine propagator.",
    )
    runner.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for multiprocessing or MPI pool (default=CPU count).",
    )

    args = parser.parse_args()
    if args.command == "train-pinn":
        train_pinn(args)
    elif args.command == "run-parareal":
        run_parareal(args)
    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":
    main()
