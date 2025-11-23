
import numpy as np
import math
from pinn_parareal import BlackScholesParams, CrankNicolsonSolver

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))
norm_cdf = np.vectorize(norm_cdf)

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

def verify():
    params = BlackScholesParams(
        r=0.05,
        sigma=0.2,
        strike=100.0,
        maturity=1.0,
        s_max=300.0
    )
    
    # Run solver
    solver = CrankNicolsonSolver(params, spatial_steps=500)
    # 100 time steps
    u_num = solver.solve_full(steps=100)
    
    # Analytical solution at T=1 (tau=1)
    s_grid = solver.s_grid
    # Avoid S=0 for log
    s_valid = s_grid[1:]
    u_exact = black_scholes_call(s_valid, params.strike, params.maturity, params.r, params.sigma)
    u_exact = np.concatenate(([0.0], u_exact))
    
    # Compare at ATM (S=100)
    idx = np.abs(s_grid - 100.0).argmin()
    print(f"S = {s_grid[idx]:.2f}")
    print(f"Numerical: {u_num[idx]:.4f}")
    print(f"Analytical: {u_exact[idx]:.4f}")
    
    error = np.abs(u_num[idx] - u_exact[idx])
    print(f"Error: {error:.4f}")
    
    if error > 1.0: # Tolerance
        print("FAIL: Significant error detected.")
    else:
        print("PASS: Solver matches analytical solution.")

if __name__ == "__main__":
    verify()
