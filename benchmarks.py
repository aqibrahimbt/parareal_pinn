"""Run Parareal benchmarks and plot runtime/speedup for different coarse propagators."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_parareal import (
    BlackScholesParams,
    CrankNicolsonSolver,
    PararealIntegrator,
    PINNCoarsePropagator,
    PINNModel,
)


@dataclass
class CoarseConfig:
    name: str
    mode: str
    device: torch.device | None
    coarse_steps: int
    skip_if_no_gpu: bool = False


def load_pinn_model(path: Path, hidden_layers: int, hidden_size: int, device: torch.device) -> PINNModel:
    model = PINNModel(hidden_layers=hidden_layers, hidden_size=hidden_size)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_coarse_propagator(
    solver: CrankNicolsonSolver,
    config: CoarseConfig,
    checkpoint: Path,
    hidden_layers: int,
    hidden_size: int,
    pinn_cache: Dict[str, PINNModel],
) -> Callable[[np.ndarray, float, float], np.ndarray]:
    if config.mode == "numeric":
        def numeric_coarse(state: np.ndarray, start: float, end: float) -> np.ndarray:
            return solver.propagate(state, start, end, config.coarse_steps)

        return numeric_coarse

    assert config.mode == "pinn"
    device = config.device or torch.device("cpu")
    cache_key = f"{config.name}-{device.type}"
    if cache_key not in pinn_cache:
        pinn_cache[cache_key] = load_pinn_model(checkpoint, hidden_layers, hidden_size, device)
    model = pinn_cache[cache_key]
    return PINNCoarsePropagator(model, solver.s_grid, device)


def measure_parareal(
    solver: CrankNicolsonSolver,
    coarse: Callable[[np.ndarray, float, float], np.ndarray],
    time_points: Iterable[float],
    fine_steps: int,
    iterations: int,
) -> tuple[float, float, float]:
    integrator = PararealIntegrator(
        time_points=time_points,
        initial_state=solver.initial_state,
        coarse=coarse,
        fine=lambda state, start, end, steps: solver.propagate(state, start, end, steps),
        fine_steps_per_slice=fine_steps,
    )
    start = time.perf_counter()
    _, errors = integrator.run(iterations=iterations)
    total_time = time.perf_counter() - start
    coarse_time = integrator.metrics["coarse_secs"]
    fine_time = integrator.metrics["fine_secs"]
    return total_time, coarse_time, fine_time


def compute_serial_time(solver: CrankNicolsonSolver, steps: int) -> float:
    start = time.perf_counter()
    solver.propagate(solver.initial_state, 0.0, solver.params.maturity, steps)
    return time.perf_counter() - start


def plot_results(
    time_slices: List[int],
    results: Dict[str, Dict[str, List[float]]],
    serial_times: Dict[int, float],
    output_dir: Path,
) -> None:
    fig, (ax_rt, ax_speedup) = plt.subplots(2, 1, figsize=(8, 10))
    colors = {"numeric": "tab:blue", "pinn_cpu": "tab:orange", "pinn_gpu": "tab:green"}
    for name, entry in results.items():
        runtime = entry["runtime"]
        speedup = [serial_times[s] / rt if rt > 0 else 0.0 for s, rt in zip(time_slices, runtime)]
        ax_rt.plot(time_slices, runtime, label=name, marker="o", color=entry["color"])
        ax_speedup.plot(time_slices, speedup, label=name, marker="o", color=entry["color"], linestyle="-")
        ax_speedup.set_ylabel("Speedup")
        ax_rt.set_ylabel("Runtime (s)")
        ax_speedup.set_xlabel("Time slices / cores")
    ax_rt.set_title("Parareal runtime vs number of slices")
    ax_rt.set_yscale("log")
    ax_rt.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_rt.legend()
    ax_speedup.set_title("Speedup vs number of slices")
    ax_speedup.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_speedup.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_speedup.png"
    fig.savefig(path)
    print(f"Saved runtime/speedup figure to {path}")


def run_benchmarks(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"PINN checkpoint does not exist at {checkpoint}. Train it with `python main.py train-pinn` first.")

    params = BlackScholesParams()
    solver = CrankNicolsonSolver(params, spatial_steps=args.spatial_steps)
    pinn_cache: Dict[str, PINNModel] = {}
    time_slices = args.time_slices
    serial_times: Dict[int, float] = {}
    configs = [
        CoarseConfig("Numeric coarse (CPU)", "numeric", torch.device("cpu"), args.coarse_steps),
        CoarseConfig("PINN coarse (CPU)", "pinn", torch.device("cpu"), args.coarse_steps),
    ]
    if torch.cuda.is_available():
        configs.append(CoarseConfig("PINN coarse (GPU)", "pinn", torch.device("cuda"), args.coarse_steps))
    else:
        print("CUDA not available: skipping GPU PINN benchmarks")
    results: Dict[str, Dict[str, List[float]]] = {}
    for config in configs:
        results[config.name] = {"runtime": [], "coarse": [], "fine": [], "color": "tab:orange"}
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, config in enumerate(configs):
        results[config.name]["color"] = palette[idx % len(palette)]
        for slices in time_slices:
            serial_key = (slices, args.fine_steps)
            if slices not in serial_times:
                total_steps = args.fine_steps * slices
                serial_times[slices] = compute_serial_time(solver, total_steps)
            coarse = build_coarse_propagator(
                solver, config, checkpoint, args.hidden_layers, args.hidden_size, pinn_cache
            )
            time_points = np.linspace(0.0, params.maturity, slices + 1)
            runtime, coarse_time, fine_time = measure_parareal(
                solver, coarse, time_points, args.fine_steps, args.iterations
            )
            print(
                f"{config.name}, slices={slices}: total={runtime:.3f}s (coarse={coarse_time:.3f}s / fine={fine_time:.3f}s), "
                f"serial={serial_times[slices]:.3f}s"
            )
            results[config.name]["runtime"].append(runtime)
            results[config.name]["coarse"].append(coarse_time)
            results[config.name]["fine"].append(fine_time)
    with open(args.report / "benchmark_results.json", "w") as handle:
        json.dump({name: data for name, data in results.items()}, handle, indent=2)
    print("Benchmark raw data saved.")
    plot_results(time_slices, results, serial_times, args.report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce runtime/speedup plots for PINN-Parareal experiments.")
    parser.add_argument("--checkpoint", default="models/pinn.pth", help="Path to trained PINN checkpoint.")
    parser.add_argument("--spatial-steps", type=int, default=200, help="Spatial steps for the fine solver.")
    parser.add_argument("--coarse-steps", type=int, default=10, help="Time steps for the numeric PINN coarse propagator (if used).")
    parser.add_argument("--fine-steps", type=int, default=20, help="Fine solver steps per Parareal slice.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of Parareal iterations to run.")
    parser.add_argument("--hidden-layers", type=int, default=10, help="Hidden layers for the PINN network (must match training).")
    parser.add_argument("--hidden-size", type=int, default=50, help="Hidden size for the PINN network (must match training).")
    parser.add_argument("--time-slices", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12, 14, 16], help="List of slice counts to benchmark.")
    parser.add_argument("--report", type=Path, default=Path("reports"), help="Directory to store figure and JSON results.")
    args = parser.parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
