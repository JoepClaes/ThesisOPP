#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:29:32 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES A PLOT OF THE OPTIMAL N PL SPLIT FOR DIFFERENT VALUES OF MEAN_XI BASED ON THE ANALYTICAL SOLUTION




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import sys

# ParametersS
n = 2  # Number of price levels (set to any n >= 3)
S = 1000
h = 0.01
r = 0.002
f = 0.003
theta = 0.001
rho_u = 0.05
delta = 0.01

# Generate queue sizes: Q1 = 2000, Q2 = 1500, Q3 = 1400, etc.
Q = [2000] + [1500 - 0 * i for i in range(n - 1)]

# Generate rate parameters: lambda_psi_i = 2 / Q_{i+1} for i = 0 to n-2
lambda_psi_list = [2 / Q[i + 1] for i in range(n - 1)]

# Generalized CDF function F_sum(k, x) for ξ + ψ1 + ... + ψk
def F_sum(k, x, lambda_xi, lambda_psi_list, Q):
    if k == 0:
        return 1 - np.exp(-lambda_xi * x) if x >= 0 else 0
    else:
        threshold = sum(Q[1:k + 1])  # Q2 + Q3 + ... + Q_{k+1}
        if x <= threshold:
            terms = []
            for j in range(1, k + 1):
                prod1 = np.prod([lambda_psi_list[m - 1] for m in range(1, k + 1) if m != j] + [1])
                prod2 = np.prod([lambda_psi_list[m - 1] - lambda_psi_list[j - 1] 
                                 for m in range(1, k + 1) if m != j] + [1])
                if prod2 == 0:
                    term = 0  # Handle rare case of equal rates
                else:
                    term = (lambda_xi * prod1) / ((lambda_xi + lambda_psi_list[j - 1]) * prod2) * \
                           np.exp(-lambda_psi_list[j - 1] * (threshold - x))
                terms.append(term)
            return sum(terms)
        else:
            prod_num = np.prod(lambda_psi_list[:k])
            prod_den = np.prod([lambda_xi + lambda_psi for lambda_psi in lambda_psi_list[:k]])
            return 1 - (prod_num / prod_den) * np.exp(-lambda_xi * (x - threshold))

# Compute optimal allocations for a given mean_xi
def compute_optimal(mean_xi, n, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta):
    lambda_xi = 1 / mean_xi
    
    # Compute thresholds from Proposition 4.5
    sum_Q = sum(Q)
    F_Q_sum_S = F_sum(n - 1, sum_Q + S, lambda_xi, lambda_psi_list, Q)
    F_Q_sum = F_sum(n - 1, sum_Q, lambda_xi, lambda_psi_list, Q)
    
    p_u_lower = (2 * h + (n - 1) * delta + f + r) / F_Q_sum_S - (h + (n - 1) * delta + r + theta) \
                if F_Q_sum_S > 0 else -np.inf
    p_u_upper = (2 * h + (n - 1) * delta + f + r) / F_Q_sum - (h + (n - 1) * delta + r + theta) \
                if F_Q_sum > 0 else np.inf
    
    # Determine allocation case
    if rho_u <= 0:
        print('a')
        M_star = 0
        L_star = [0] * (n - 1) + [S]
    elif rho_u >= 2:
        print('b')
        M_star = S
        L_star = [0] * n
    else:
        # Mixed allocation case
        x = (2 * h + (n - 1) * delta + f + r) / (h + (n - 1) * delta + r + rho_u + theta)
        threshold = sum(Q[1:])
        
        if x >= F_sum(n - 1, threshold, lambda_xi, lambda_psi_list, Q):
            c = np.prod(lambda_psi_list) / np.prod([lambda_xi + lambda_psi for lambda_psi in lambda_psi_list])
            A = -1 / lambda_xi * np.log((1 - x) / c) + threshold
        else:
            print('c')
            def func(A):
                return F_sum(n - 1, A, lambda_xi, lambda_psi_list, Q) - x
            try:
                sol = root_scalar(func, bracket=[0, threshold], method='brentq')
                A = sol.root
            except ValueError:
                A = threshold  # Fallback
            
        M_star = S - A + sum_Q
        M_star = max(0, min(S, M_star))
        
        # Solve for L_i_star iteratively
        L_star = []
        remaining = S - M_star
        for i in range(n - 1):
            def g(L):
                sum_up_to_i = sum(Q[:i + 1]) + sum(L_star) + L if i > 0 else Q[0] + L
                sum_up_to_i_plus1 = sum_up_to_i + Q[i + 1]
                term1 = delta + (h + i * delta + r + rho_u + theta) * F_sum(i, sum_up_to_i, lambda_xi, lambda_psi_list, Q)
                term2 = -(h + (i + 1) * delta + r + rho_u + theta) * F_sum(i+1, sum_up_to_i_plus1, lambda_xi, lambda_psi_list, Q)
                return term1 + term2
            
            try:
                sol = root_scalar(g, bracket=[0, remaining - sum(L_star)], method='brentq')
                L_i_star = sol.root
            except ValueError:
                #print(g(0))
                if g(0) > 0:
                    L_i_star = 0
                elif g(S) < 0:
                    L_i_star = remaining - sum(L_star)
                    
                else:
                    L_i_star = 500  # Default
            
            L_i_star = max(0, min(remaining - sum(L_star), L_i_star))
            L_star.append(L_i_star)
            remaining -= L_i_star
        
        L_star.append(max(0, remaining))
    
    return M_star, L_star

# Compute allocations over a range of mean_xi
mean_xi_range = np.linspace(4000, 9000, 1000)
M_values = []
L_values = [[] for _ in range(n)]  # Store L1 to Ln


M_opt, L = compute_optimal(6000, n, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta)

print(6000, n, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta)
print(M_opt,L)


for mean_xi in mean_xi_range:
    M, L_star = compute_optimal(mean_xi, n, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta)
    M_values.append(M)
    for i in range(n):
        L_values[i].append(L_star[i])

#colors = ['orange', 'purple', 'green', 'brown', 'red']
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_xi_range, M_values, label='M (Market Orders)', linewidth=2)
for i in range(n):
    plt.plot(mean_xi_range, L_values[i],  label=f'L{i + 1} (Level {i + 1})', linewidth=2)
plt.xlabel('Mean of ξ')
plt.ylabel('Optimal Allocation Size (0 to 1000)')
plt.title(f'Optimal Allocations vs Mean of ξ for n={n} Price Levels')
plt.legend()
plt.grid(True)
plt.ylim(0, 1000)
plt.show()

sys.exit()