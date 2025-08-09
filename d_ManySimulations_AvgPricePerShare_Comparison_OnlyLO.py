#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:59:36 2025

@author: jawclaes
"""

# COMPARING OPTIMAL STRATEGY VS CARTEA AND JAIMUNGAL THAT DONT CONSIDER DRIFT
# THIS TIME RUNNING IT FOR 10,000 MC SIMULATIONS AND PRINTING THE TOTAL AVERAGE PRICE PER SHARE FOR EACH STRATEGY

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
import math

# Set the parameters of our system (all values in $ and seconds)
R = 6          # Initial inventory
T = 60         # Total time
kappa = 100    # Price impact parameter
Delta = 0.01   # Spread  was 0.01
lam_start = 50/60  # Market order arrival rate
rho_u = 0 # was 0.005
theta = 0 # Risk aversion parameter
r = 0.003      # Additional cost for LO
f = 0.002      # Additional cost for MO
lam = lam_start * np.exp(kappa * (Delta + r + f) - 1)  # Adjusted arrival rate

# Define the stochastic price movement
def simulate_midprice(p0, mu, sig_p, T, N):
    dt = T / N
    dW = np.random.normal(0, np.sqrt(dt), N)
    dp = mu * dt + sig_p * dW
    prices = np.zeros(N + 1)
    prices[0] = p0
    prices[1:] = p0 + np.cumsum(dp)
    return prices

# Simulation parameters
p0 = 60        # Initial price
mu = -0.001    # Drift
sig_p = 0.01   # Volatility
N = 1000      # Number of steps
n_sim = 10000      # Number of simulations

# Simulate three midprice paths
prices = np.zeros((N+1, n_sim))
for i in range(n_sim):
    prices[:, i] = simulate_midprice(p0, mu, sig_p, T, N)

# Time points
time = np.linspace(0, T, N + 1)

# Define helper functions
def f_theta(i, theta):
    return theta * i**2

def A(i, kappa, mu, theta):
    return kappa * (i * mu - f_theta(i, theta))

# Precompute A values
A_values = np.zeros(R + 1)
for i in range(R + 1):
    A_values[i] = A(i, kappa, mu, theta)

# Base case for tau_1
TAU_1 = T
tau_cache = {1: TAU_1}

# Compute coefficient term
def compute_coeff(i, S, lam):
    if i == 0:
        return 1
    denominator = 1
    for j in range(1, i + 1):
        diff = A_values[S] - A_values[S - j]
        if diff == 0:
            raise ValueError(f"Denominator zero at S={S}, j={j}")
        denominator *= diff
    return (lam ** i) / denominator

# Define g_S
def g_S(S, t):
    if S == 0:
        return 1
    if S == 1:
        A_1 = A_values[1]
        return np.exp(A_1 * (T - t)) * (1 + lam / A_1) - lam / A_1
    tau_S = tau(S)
    A_S = A_values[S]
    term1 = g_S(S - 1, tau_S)
    for i in range(1, S + 1):
        coeff = compute_coeff(i, S, lam)
        term1 += coeff * g_S(S - i, tau_S)
    term1 *= np.exp(A_S * (tau_S - t))
    term2 = 0
    for i in range(1, S + 1):
        coeff = compute_coeff(i, S, lam)
        term2 += coeff * g_S(S - i, t)
    return term1 - term2

# Define f_S(t)
def f_S(S, t):
    if S < 2:
        raise ValueError("f_S is defined for S >= 2")
    A_S = A_values[S]
    A_S_1 = A_values[S - 1]
    denominator = A_S - A_S_1 + lam
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    if S == 2:
        return g_S(S - 1, t) - (lam / denominator)
    else:
        return g_S(S - 1, t) - (lam / denominator) * g_S(S - 2, t)

# Solve for tau_S
def solve_tau_S(S, initial_guess=0, tol=1e-6, max_iter=10000):
    if S in tau_cache:
        return tau_cache[S]
    if S==1:
        tau_S = 60
        tau_cache[S] = tau_S
        return tau_S
    
    
    def f(t):
        return f_S(S, t)
    try:
        tau_S = newton(f, initial_guess, tol=tol, maxiter=max_iter)
        tau_cache[S] = tau_S
        return tau_S
    except RuntimeError as e:
        print(f"Failed to converge for S={S}: {e}")
        return None

# Define tau(i)
def tau(i):
    if i == 1:
        return TAU_1
    elif i >= 2:
        return solve_tau_S(i)
    else:
        raise ValueError("Index i must be positive")

# Precompute tau values
for i in range(1, R + 1):
    tau(i)  # Populates tau_cache

# Compute g_all and w_all
g_all = np.zeros((N + 1, R))
w_all = np.zeros((N + 1, R))
for x in range(R):
    print(x)
    for i in range(N + 1):
        g_all[i, x] = g_S(x + 1, time[i])
        if x == 0:
            w_all[i, x] = g_all[i, x] if time[i] < tau_cache[1] else 1
        else:
            tau_x = tau_cache[x + 1]
            w_all[i, x] = g_all[i, x] if time[i] < tau_x else w_all[i, x - 1]
        if w_all[i, x] <= 1:
            w_all[i, x] = 1

# Compute optimal depths
depths_all = np.zeros((N + 1, R))
for x in range(R):
    for i in range(N + 1):
        if x == 0:
            depths_all[i, x] = 1/kappa - Delta - r - f + 1/kappa * np.log(w_all[i,x])
        else:
            depths_all[i, x] = 1/kappa - Delta - r - f + 1/kappa * np.log(w_all[i,x]/w_all[i,x-1])

# Simulation function for optimal strategy
def simulate_one_path(prices, depths_all, tau_cache, dt, lam, kappa, Delta, r, f, R):
    N = len(prices) - 1
    inventory = np.zeros(N + 1)
    inventory[0] = R
    total_cost = 0
    shares_executed = 0
    avg_price = np.zeros(N + 1)
    lo_fills = []
    mo_executions = []
    depths = np.zeros(N + 1)
    depths[0] = depths_all[0, R - 1]  # Initial depth for inventory R

    for k in range(N):
        i_k = int(inventory[k])
        # Execute MOs if time >= tau(i_k)
        while i_k > 0 and time[k] >= tau_cache[i_k]:
            price = prices[k] - Delta / 2 - f
            total_cost += price
            shares_executed += 1
            i_k -= 1
            mo_executions.append(k)
            if i_k > 0:
                depths[k + 1] = depths_all[k + 1, i_k - 1]
            else:
                depths[k + 1] = 0
        if i_k > 0:
            delta_k = depths_all[k, i_k - 1]
            prob_LO = lam_start * np.exp(-kappa * delta_k) * dt
            if np.random.uniform(0, 1) < prob_LO:
                price = prices[k] + delta_k + Delta / 2 + r
                total_cost += price
                shares_executed += 1
                i_k -= 1
                lo_fills.append(k)
                if i_k > 0:
                    depths[k + 1] = depths_all[k + 1, i_k - 1]
                else:
                    depths[k + 1] = 0
            else:
                depths[k + 1] = delta_k
        inventory[k + 1] = i_k
        avg_price[k + 1] = total_cost / shares_executed if shares_executed > 0 else prices[0]

    # At the final time step (t = T, k = N), liquidate any remaining inventory with MOs
    i_k = int(inventory[N-1])  # Use inventory before the last step to reflect the state
    if i_k > 0:
        for _ in range(i_k):
            price = prices[N] - Delta / 2 - f
            total_cost += price
            shares_executed += 1
            mo_executions.append(N-1)
        inventory[N] = 0  # Set final inventory to 0 after execution
        avg_price[N] = total_cost / shares_executed
        depths[N] = 0

    return inventory, avg_price, lo_fills, mo_executions, depths

# Run simulations
dt = time[1] - time[0]
inventories = np.zeros((n_sim, N + 1))
avg_prices = np.zeros((n_sim, N + 1))
depths_sim = np.zeros((n_sim, N + 1))
lo_fills_all = [[] for _ in range(n_sim)]
mo_executions_all = [[] for _ in range(n_sim)]
# NEW: Array to store terminal average prices for optimal strategy
avg_price_opt_T = np.zeros(n_sim)

for sim_index in range(n_sim):
    prices_sim = prices[:, sim_index]
    # Optimal strategy
    inventory, avg_price, lo_fills, mo_executions, depths = simulate_one_path(
        prices_sim, depths_all, tau_cache, dt, lam, kappa, Delta, r, f, R
    )
    inventories[sim_index, :] = inventory
    avg_prices[sim_index, :] = avg_price
    depths_sim[sim_index, :] = depths
    lo_fills_all[sim_index] = lo_fills
    mo_executions_all[sim_index] = mo_executions
    # NEW: Store terminal average price
    avg_price_opt_T[sim_index] = avg_price[-1]


# Compute g_all and w_all
g_all = np.zeros((N + 1, R))
w_all = np.zeros((N + 1, R))
for q in range(R):
    print(q)
    for t in range(N + 1):
        w_all[t, q] = 0
        for i in range(q+1):
            sommetje = 0
            for j in range(q+1-i):
                sommetje += (lam_start/(mu*kappa*np.exp(1)))**j * (1/math.factorial(j)) * np.exp(-1*kappa*rho_u* (q-i-j)**2 )
            w_all[t, q] += ( (-1)**i * np.exp((q-i)*mu*kappa*(T-t)) * (lam_start/(mu*kappa*np.exp(1)))**i ) * sommetje
        




# -- Add this function after the other simulation functions
def simulate_limit_order_only(prices, depths_all, tau_cache, dt, lam, kappa, Delta, r, f, R):
    N = len(prices) - 1
    inventory = np.zeros(N + 1)
    inventory[0] = R
    total_cost = 0
    shares_executed = 0
    avg_price = np.zeros(N + 1)
    lo_fills = []
    depths = np.zeros(N + 1)

    for k in range(N):
        i_k = int(inventory[k])
        if i_k == 0:
            inventory[k + 1] = 0
            avg_price[k + 1] = total_cost / shares_executed if shares_executed > 0 else prices[0]
            depths[k + 1] = 0
            continue

        tau_i = tau_cache[i_k]

        if time[k] < tau_i:
            delta_k = depths_all[k, i_k - 1]
        else:
            delta_k = 0.0  # Limit order at best bid/ask

        prob_LO = lam_start * np.exp(-kappa * delta_k) * dt
        if np.random.uniform(0, 1) < prob_LO:
            price = prices[k] + delta_k + Delta / 2 + r
            total_cost += price
            shares_executed += 1
            i_k -= 1
            lo_fills.append(k)

        inventory[k + 1] = i_k
        avg_price[k + 1] = total_cost / shares_executed if shares_executed > 0 else prices[0]
        depths[k + 1] = delta_k
    
    # Final liquidation at time T for remaining inventory
    i_k = int(inventory[N - 1])
    if i_k > 0:
        for _ in range(i_k):
            final_price = prices[N] - Delta / 2 - f - rho_u  # or adjust signs if needed
            total_cost += final_price
            shares_executed += 1
        inventory[N] = 0
    else:
        inventory[N] = i_k  # Already 0

    avg_price[N] = total_cost / shares_executed if shares_executed > 0 else prices[N]
    depths[N] = 0
    return inventory, avg_price, lo_fills, depths

# -- Run simulations for LO-only strategy
inventories_LO_ONLY = np.zeros((n_sim, N + 1))
avg_prices_LO_ONLY = np.zeros((n_sim, N + 1))
depths_sim_LO_ONLY = np.zeros((n_sim, N + 1))
lo_fills_all_LO_ONLY = [[] for _ in range(n_sim)]
avg_price_lo_only_T = np.zeros(n_sim)

for sim_index in range(n_sim):
    prices_sim = prices[:, sim_index]
    inventory_LO_ONLY, avg_price_LO_ONLY, lo_fills_LO_ONLY, depths_LO_ONLY = simulate_limit_order_only(
        prices_sim, depths_all, tau_cache, dt, lam, kappa, Delta, r, f, R
    )
    inventories_LO_ONLY[sim_index, :] = inventory_LO_ONLY
    avg_prices_LO_ONLY[sim_index, :] = avg_price_LO_ONLY
    depths_sim_LO_ONLY[sim_index, :] = depths_LO_ONLY
    lo_fills_all_LO_ONLY[sim_index] = lo_fills_LO_ONLY
    avg_price_lo_only_T[sim_index] = avg_price_LO_ONLY[-1]

# -- Print final comparison
print("\nAverage Average Price across all simulations Comparison:")
print(f"Optimal strategy (with MOs)       = ${np.mean(avg_price_opt_T):.4f}")
print(f"LO-only strategy (no MOs)         = ${np.mean(avg_price_lo_only_T):.4f}")
