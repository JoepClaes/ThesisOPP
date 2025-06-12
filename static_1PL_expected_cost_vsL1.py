#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:22:29 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES A PLOT OF THE EXPECTED COST VS L1 FOR 3 DIFFERENT RHO_U VALUES


import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm, expon, pareto



# Parameters
Q1 = 2000
S = 1000
h = 0.02
r = 0.002
f = 0.003
theta = 0.0005
rho_o = 0.05
delta = 0.01
n = 1  # Single price level
Q = [Q1]  # Queue size for n=1
mean_xi = 2200  # Expectation of xi
num_samples = 10000  # Number of Monte Carlo samples

rho_max = (2*h + f + r) / expon.cdf(Q1, loc=0, scale=mean_xi) - h - r - theta
rho_min = (2*h + f + r) / expon.cdf(Q1 + S, loc=0, scale=mean_xi) - h - r - theta
rho_u_values = [rho_min, 0.043, rho_max]  # Different rho_u values for plotting

# Generate samples for xi (no psi for n=1)
#np.random.seed(42)  # For reproducibility
xi_samples = np.random.exponential(mean_xi, num_samples)

# Compute OF_i for price level i
def compute_OF_i(i, xi, psi, Q, L):
    sum_Q_L_prev = sum(Q[:i-1] + L[:i-1]) if i > 1 else 0
    sum_psi = sum(psi[:i-1]) if i > 1 else 0
    total_Q_L_up_to_i = sum_Q_L_prev + Q[i-1]
    total_Q_L_incl_i = total_Q_L_up_to_i + L[i-1]
    total_flow = xi + sum_psi
    indicator_1 = 1 if total_Q_L_up_to_i < total_flow < total_Q_L_incl_i else 0
    indicator_2 = 1 if total_flow >= total_Q_L_incl_i else 0
    OF_i = (total_flow - total_Q_L_up_to_i) * indicator_1 + L[i-1] * indicator_2
    return max(0, min(OF_i, L[i-1]))

# Compute total shares acquired A
def compute_A(M, L, xi, psi, Q):
    OF = [compute_OF_i(i+1, xi, psi, Q, L) for i in range(len(L))]
    return M + sum(OF)

# Cost function v(X, xi, psi)
def cost_function(M, L, xi, psi, Q, S, h, r, f, theta, rho_u, rho_o, delta):
    n = len(L)
    OF = [compute_OF_i(i+1, xi, psi, Q, L) for i in range(n)]
    A = compute_A(M, L, xi, psi, Q)
    # Execution cost for market orders
    exec_cost_M = (h + f) * M
    # Execution cost for limit orders
    exec_cost_L = -sum((h + r + (i) * delta) * OF[i] for i in range(n))
    # Under-filling and over-filling penalties
    underfill = max(S - A, 0)
    overfill = max(A - S, 0)
    penalty_under = rho_u * underfill 
    penalty_over = rho_o * overfill  
    # Impact cost
    impact = theta * (M + sum(L) + underfill)
    return exec_cost_M + exec_cost_L + penalty_under + penalty_over + impact

# Expected cost over xi samples
def expected_cost(M, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta):
    costs = []
    for xi in xi_samples:
        # For n=1, psi is empty
        cost = cost_function(M, L, xi, [], Q, S, h, r, f, theta, rho_u, rho_o, delta)
        costs.append(cost)
    return np.mean(costs)

# Compute expected cost for range of L1 values
L1_values = np.linspace(0, S, 100)  # Range of L1 from 0 to 1000
costs = {rho_u: [] for rho_u in rho_u_values}

for rho_u in rho_u_values:
    for L1 in L1_values:
        L = [L1]  # Single price level
        M = S - L1  # Market orders
        cost = expected_cost(M, L, xi_samples, [], Q, S, h, r, f, theta, rho_u, rho_o, delta)
        costs[rho_u].append(cost)

# Plot the results
plt.figure(figsize=(10, 6))
for rho_u in rho_u_values:
    plt.plot(L1_values, costs[rho_u], label=f'ρ_u = {rho_u}')
plt.title("Expected Cost vs. Number of Limit Orders (L1) for n=1 Price Level")
plt.xlabel("Number of Limit Orders (L1)")
plt.ylabel("Expected Cost")
plt.legend()
plt.grid(True)
plt.show()