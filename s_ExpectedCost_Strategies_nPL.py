#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 21:24:23 2025

@author: jawclaes
"""

# THIS SCRIPT PLOTS THE EXPECTED COST OF CERTAIN STRATEGIES BASED ON MONTE CARLO SIMULATIONS OF THE COST
# SAMPLING THE RVS XI AND PSI FOR N PRICE LEVELS


import numpy as np
import matplotlib.pyplot as plt

# Parameters
S = 1000
h = 0.02
r = 0.002
f = 0.003
theta = 0.0005
rho_u = 0.05
rho_o = 0.05
delta = 0.01
n = 6  # Number of price levels
num_samples = 200

# Queue sizes: Q1 = 2000, Q2 = 1500, Q3 = 1400, Q4 = 1300
Q = [2000] + [1500 - 0 * i for i in range(n - 1)]
lambda_psi_list = [2 / Q[i + 1] for i in range(n - 1)]

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
    exec_cost_M = (h + f) * M
    exec_cost_L = -sum((h + r + i * delta) * OF[i] for i in range(n))
    underfill = max(S - A, 0)
    overfill = max(A - S, 0)
    penalty_under = rho_u * underfill 
    penalty_over = rho_o * overfill  
    impact = theta * (M + sum(L) + underfill)
    return exec_cost_M + exec_cost_L + penalty_under + penalty_over + impact

# Expected cost over xi and psi samples
def expected_cost(M, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta):
    costs = []
    OFs = [[] for _ in range(n)]
    for xi, psi in zip(xi_samples, psi_samples):
        cost = cost_function(M, L, xi, psi, Q, S, h, r, f, theta, rho_u, rho_o, delta)
        costs.append(cost)
        for i in range(n):
            OFs[i].append(compute_OF_i(i+1, xi, psi, Q, L))
    return np.mean(costs), [np.mean(OF) for OF in OFs]

# Main computation
mean_xi_list = np.linspace(0, 10000, 100)
cost_allo1 = np.zeros(len(mean_xi_list))
cost_allo2 = np.zeros(len(mean_xi_list))
cost_allo3 = np.zeros(len(mean_xi_list))
cost_allo4 = np.zeros(len(mean_xi_list))
cost_allo5 = np.zeros(len(mean_xi_list))
OF1_allo2 = np.zeros(len(mean_xi_list))
OF2_allo3 = np.zeros(len(mean_xi_list))
OF3_allo4 = np.zeros(len(mean_xi_list))
OF4_allo5 = np.zeros(len(mean_xi_list))

for i in range(len(mean_xi_list)):
    # Generate samples for each mean_xi
    xi_samples = np.random.exponential(mean_xi_list[i], num_samples)
    psi_samples = np.zeros((num_samples, n-1))
    for j in range(n-1):
        psi_samples[:,j] = Q[j+1] - np.random.exponential(1/lambda_psi_list[j], num_samples)

    L = np.zeros(n)
    # Allocation 1: Market only
    L[0] = 0
    L[1] = 0
    cost_allo1[i], _ = expected_cost(S, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    
    # Allocation 2: L1 only
    L[0] = S
    L[1] = 0
    cost_allo2[i], OFs_allo2 = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    OF1_allo2[i] = OFs_allo2[0]
    
    # Allocation 3: L2 only
    L[0] = 0
    L[1] = S
    cost_allo3[i], OFs_allo3 = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    OF2_allo3[i] = OFs_allo3[1]
    
    # Allocation 4: L3 only
    L[0] = 0
    L[1] = 0
    L[2] = S
    cost_allo4[i], OFs_allo4 = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    OF3_allo4[i] = OFs_allo4[2]
    
    # Allocation 5: Half L1, Half L2
    L[0] = S/2
    L[1] = S/2 
    L[2] = 0
    cost_allo5[i], OFs_allo5 = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    
    print(i)

# Plotting costs
plt.figure(figsize=(10, 6))
plt.plot(mean_xi_list, cost_allo1, label='Only Market Orders')
plt.plot(mean_xi_list, cost_allo2, label='Only L1 Orders')
plt.plot(mean_xi_list, cost_allo3, label='Only L2 Orders')
plt.plot(mean_xi_list, cost_allo4, label='Only L3 Orders')
#plt.plot(mean_xi_list, cost_allo5, label='Half L1, Half L2 Orders')
plt.xlabel('Mean of ξ')
plt.ylabel('Expected Cost')
plt.title('Expected Cost for Different Allocations')
plt.legend()
plt.grid(True)
plt.show()

# Plotting OFs
plt.figure(figsize=(10, 6))
plt.plot(mean_xi_list, OF1_allo2, label='E[OF_1] for L1 Orders')
plt.plot(mean_xi_list, OF2_allo3, label='E[OF_2] for L2 Orders')
plt.plot(mean_xi_list, OF3_allo4, label='E[OF_3] for L3 Orders')
plt.xlabel('Mean of ξ')
plt.ylabel('Expected Orders Filled')
plt.title('Expected Orders Filled for L1 and L2 Allocations')
plt.legend()
plt.grid(True)
plt.show()





