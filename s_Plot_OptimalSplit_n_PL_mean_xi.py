#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 16:03:04 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES A PLOT OF THE OPTIMAL SPLIT FOR N-PL FOR DIFFERENT VALUES OF MEAN_XI BASED ON MONTE CARLO SIMULATIONS



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Parameters
S = 1000  # Total size to allocate
h = 0.02
r = 0.002
f = 0.003
theta = 0.0005
rho_u = 0.05
delta = 0.01
num_sim = 100000  # Number of simulations
price_levels = 3  # Adjustable: set between 2 and 8
n= price_levels

# Define Q values up to Q9 to support up to 8 price levels
Q = [2000] + [1500 - 100 * i for i in range(n)]  # Q1=2000, Q2=Q3=...=Q9=1500

# Function to compute optimal allocations
def compute_optimal(mean_xi):
    # Generate samples
    xi_samples = np.random.exponential(scale=mean_xi, size=num_sim)
    psi_samples = [Q[i + 1] - np.random.exponential(scale=0.5 * Q[i + 1], size=num_sim) 
                   for i in range(price_levels)]
    
    # Cumulative sums: sum_samples[0] = xi, sum_samples[1] = xi + psi1, etc.
    sum_samples = [xi_samples]
    for i in range(price_levels):
        sum_samples.append(sum_samples[-1] + psi_samples[i])
    
    # Empirical CDF functions
    F_sum = [lambda x, i=i: np.mean(sum_samples[i] <= x) for i in range(price_levels + 1)]
    
    # Compute thresholds
    p = (2 * h + 2 * delta + f + r) / (h + 2 * delta + r + rho_u + theta)
    p_u_lower = -2
    p_u_upper = 2
    
    allocations = [0] * (price_levels + 1)  # [M, L1, L2, ..., L_price_levels]
    
    if rho_u <= p_u_lower:
        allocations[-1] = S  # All to highest limit order
    elif rho_u >= p_u_upper:
        allocations[0] = S   # All to market orders
    else:
        # Mixed allocation
        A = np.percentile(sum_samples[price_levels - 1], p * 100)
        M_star = S - A + sum(Q[:price_levels])
        M_star = max(0, min(S, M_star))
        allocations[0] = M_star
        
        remaining = S - M_star
        cumulative_L = 0
        for i in range(1, price_levels + 1):
            a_i = h + (i - 1) * delta + r + rho_u + theta
            b_i = -(h + i * delta + r + rho_u + theta)
            c_i = delta
            
            def g(L):
                arg1 = sum(Q[:i]) + cumulative_L + L
                arg2 = sum(Q[:i + 1]) + cumulative_L + L
                return (a_i * F_sum[i - 1](arg1) + 
                        b_i * F_sum[i](arg2) + c_i)
            
            try:
                sol = root_scalar(g, bracket=[0, remaining], method='brentq')
                L_i = max(0, min(remaining, sol.root))
            except ValueError:
                L_i = 0 if g(0) > 0 else remaining
            allocations[i] = L_i
            cumulative_L += L_i
            remaining -= L_i
        
        # Adjust last level to use remaining size
        total_L = sum(allocations[1:])
        if total_L < S - M_star:
            allocations[price_levels] += S - M_star - total_L
    
    return allocations

# Compute allocations over a range of mean_xi
mean_xi_range = np.linspace(5000, 11000, 200)
results = [compute_optimal(mean_xi) for mean_xi in mean_xi_range]

# Extract M and L values
M_values = [r[0] for r in results]
L_values = [[r[i + 1] for r in results] for i in range(price_levels)]

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(mean_xi_range, M_values, label='M (Market Orders)', linewidth=2)
for i in range(price_levels):
    plt.plot(mean_xi_range, L_values[i], label=f'L{i + 1} (Limit Order Level {i + 1})', linewidth=2)
plt.xlabel('Mean of ξ')
plt.ylabel('Allocation Size')
plt.title(f'Optimal Allocations vs Mean of ξ ({price_levels} Price Levels)')
plt.legend()
plt.grid(True)
plt.ylim(0, S)
plt.show()