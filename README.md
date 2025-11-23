# PINN Parareal Black-Scholes

This repository implements the workflow from *Parareal with a physics-informed neural network as coarse propagator* by combining
a PINN-based coarse propagator with a Crank–Nicolson fine model inside a Parareal driver. The focus is on modularity, GPU-friendly PINN execution,
CPU/resizable fine solvers, and benchmarking tooling for runtime/speedup plots.

## System overview

- `main.py` exposes two entry points: `train-pinn` builds and stages the PINN (10 layers, 50 neurons) while `run-parareal` runs the Parareal loop
  with optional multiprocessing or MPI acceleration for the fine propagator.
- `pinn_parareal/pinn.py` contains the model architecture, physics-informed losses (PDE residual, boundary, expiration), staged Adam training, and the
  PINN coarse propagator that adds a delta correction to the current slice.
- `pinn_parareal/solver.py` provides the Crank–Nicolson fine integrator with second-order central differences over the Black–Scholes PDE.
- `pinn_parareal/parareal.py` orchestrates the Parareal iteration and now supports running the fine propagator serially, via Python multiprocessing, or
  via `mpi4py`’s `MPIPoolExecutor`.
- `benchmarks.py` reproduces the paper’s runtime/speedup experiments, comparing numeric, CPU PINN, and (if available) GPU PINN coarse propagators across multiple
  time-slice counts and recording plots/JSON reports.

## Requirements & setup

1. Create and activate a Python 3.11+ virtual environment (the repo already contains `.venv311` for testing).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. The PINN leverages PyTorch (CPU/GPU) while the fine solver stays in NumPy, so a CUDA-capable GPU is optional but recommended for the PINN coarse run.

## Step-by-step execution

### 1. Train the PINN coarse propagator

This training uses 100k interior points plus boundary/expiration collocation sets, Adam with two-stage learning rates, and the combined physics losses.
The checkpoint is stored for reuse by the Parareal driver.

```bash
python main.py train-pinn --output models/pinn.pth
```

Optional flags include `--hidden-layers`, `--hidden-size`, point counts, epochs, and learning rates. The CLI prints loss summaries during training.

### 2. Run Parareal with the trained PINN

The fine propagator is the Crank–Nicolson solver; the coarse propagator uses the PINN checkpoint. The CLI reports relative error per iteration and the cumulative
coarse/fine runtimes.

```bash
python main.py run-parareal --model-path models/pinn.pth
```

Large-scale runs can spawn multiple workers for the fine propagator:
- **Multiprocessing**: `--parallel-mode multiprocessing --workers 8` uses `ProcessPoolExecutor` per iteration.
- **MPI**: `--parallel-mode mpi --workers 4` uses `mpi4py.futures.MPIPoolExecutor` (MPI runtime required).

### 3. Reproduce runtime/speedup plots

`benchmarks.py` reruns Parareal over a sweep of time slices, comparing numeric vs. PINN coarse propagators (CPU+GPU). It writes `reports/runtime_speedup.png` and a JSON report.

```bash
python benchmarks.py --checkpoint models/pinn.pth
```

Adjust CLI flags such as `--time-slices`, `--iterations`, `--fine-steps`, or PINN architecture parameters to match specific experiment setups.

## Notes

- `main.py` automatically runs on CUDA if available; otherwise the PINN runs on CPU. The MPI option requires an MPI implementation that supports dynamic process spawning.
- Benchmarks compute a serial runtime at each slice count and compare it to the parallel Parareal execution to derive speedup.
- The dataset for PINN training is generated once but reshuffled each epoch to stay faithful to the paper while avoiding repeated allocation.
