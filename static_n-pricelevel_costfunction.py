#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:17:02 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES THE PLOT OF THE EXPECTED COST FUNCTION FOR 2PL WHERE YOU CAN SET THE MEAN_XI AND THE 3RD VARIABLE =0
# TO CHANGE WHICH VARIABLES TO PLOT ON THE X AND Y AXIS, CHANGE IT MANUALLY IN THE SCRIPT
# IT ALSO PLOTS THE COST FROM MONTE CARLO SIMULATIONS FOR L1+L2=S AND M=0 FOR DIFFERENT L1 VALUES, TO SEE WHAT HAPPENS AT THE TIPPING POINT


import numpy as np
import matplotlib.pyplot as plt
import sys

# Parameters
Q1 = 2000
Q2 = 1500
S = 1000
h = 0.02
r = 0.002
f = 0.003
theta = 0.0005
rho_u = 0.05
rho_o = 0.05
delta = 0.01
n = 4  # Number of price levels

mean_xi = 3000  # Expectation of xi

num_samples = 1000  # Number of Monte Carlo samples for expectation

# Generate queue sizes: Q1 = 2000, Q2 = 1500, Q3 = 1400, etc.
Q = [2000] + [1500 - 0 * i for i in range(n - 1)]

# Generate rate parameters: lambda_psi_i = 2 / Q_{i+1} for i = 0 to n-2
lambda_psi_list = [1 / Q[i + 1] for i in range(n - 1)]


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
    #overfill = 0
    overfill = max(A - S, 0)
    penalty_under = rho_u * underfill 
    penalty_over = rho_o * overfill  
    # Impact cost
    impact = theta * (M + sum(L) + underfill)
    return exec_cost_M + exec_cost_L + penalty_under + penalty_over + impact

# Expected cost over xi and psi samples
def expected_cost(M, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta):
    costs = []
    for xi, psi in zip(xi_samples, psi_samples):
        cost = cost_function(M, L, xi, psi, Q, S, h, r, f, theta, rho_u, rho_o, delta)
        costs.append(cost)
    return np.mean(costs)


psi_samples = np.zeros((num_samples,n-1))
for i in range(n-1):
    psi_samples[:,i] = Q[i] - np.random.exponential(1/lambda_psi_list[i], num_samples)  # psi_1 for n=2


mean_xi_list = np.linspace(0,10000,100)

L = np.zeros(n)
#L[0] = 500
#L[1] = 500

#xi=3000

#for psi in psi_samples:
#    print(compute_OF_i(1, xi, psi, Q, L),compute_OF_i(2, xi, psi, Q, L))
#    print(compute_A(0, L, xi, psi, Q))


cost_allo1 = np.zeros(len(mean_xi_list))
cost_allo2 = np.zeros(len(mean_xi_list))
cost_allo3 = np.zeros(len(mean_xi_list))


for i in range(len(mean_xi_list)):
    xi_samples = np.random.exponential(mean_xi_list[i], num_samples)
    #xi_samples = np.full(num_samples,mean_xi_list[i])
    print(i)
    L[0] = 0
    L[1] = 0
    cost_allo1[i] = expected_cost(S, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    L[0] = S
    L[1] = 0
    cost_allo2[i] = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    L[0] = 0
    L[1] = S
    cost_allo3[i] = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    
plt.plot(mean_xi_list,cost_allo1, label='Only market orders')
plt.plot(mean_xi_list,cost_allo2, label='Only L1 orders')
plt.plot(mean_xi_list,cost_allo3, label='Only L2 orders')
plt.legend()
plt.show()

sys.exit()


M_values =  np.array([0])  # Sample M values when constraint is off
L1_values = np.linspace(0, 1000, 50) # Extended range for full meshgrid
L2_values = np.linspace(0, 1000, 50) 
cost_matrix_with_constraint = np.zeros((len(L1_values), len(L2_values)))
cost_matrices_no_constraint = [np.zeros((len(L1_values), len(L2_values))) for _ in M_values]

# With constraint S = M + L1 + L2
#for i, M in enumerate(M_values):
#    for j, L1 in enumerate(L1_values):
#        L2 = S - M - L1  # To disable constraint, comment this line
#        if L2 < 0:  # To disable constraint, comment this block
#            cost_matrix_with_constraint[i, j] = np.nan
#            continue  # To disable constraint, comment this line
#        L = [L1, L2]
#        cost_matrix_with_constraint[i, j] = expected_cost(M, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)


# Plot Expected Cost vs M with constraint L1 + M = S (L2 = 0)
cost_vs_L1 = []
i=0
for L1 in L1_values:
    i += 1
    print(i)
    L2 = S - L1  # Enforce constraint M + L1 = S
    L = [L1, L2]  # L2 = 0 since we're in 1-price-level case
    cost = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    cost_vs_L1.append(cost)

plt.figure(figsize=(10, 6))
plt.plot(L1_values, cost_vs_L1, label='Expected Cost', linewidth=2, color='blue')
plt.xlabel('L1 Orders (L1)')
plt.ylabel('Expected Cost ($)')
plt.title('Expected Cost vs L1 Orders (L1) with L1 + L2 = S, M = 0')
plt.grid(True)
plt.legend()
plt.show()

sys.exit()


# Without constraint: loop over L2 values
for idx, M in enumerate(M_values):
    for i, L1 in enumerate(L1_values):
        for j, L2 in enumerate(L2_values):
            L = [L1, L2]
            cost_matrices_no_constraint[idx][i, j] = expected_cost(M, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)

# Plot with constraint
#plt.figure(figsize=(10, 8))
L1_grid, L2_grid = np.meshgrid(L1_values, L2_values)
#contour = plt.contourf(M_grid, L1_grid, cost_matrix_with_constraint.T, levels=20, cmap='viridis')
#plt.colorbar(contour, label='Expected Cost ($)')
#plt.xlabel('Market Orders (M)')
#plt.ylabel('Limit Orders at Level 1 (L1)')
#plt.title('Expected Cost with Constraint S = M + L1 + L2')
#plt.plot()

# Plot without constraint for each L2 value
for idx, M in enumerate(M_values):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(L1_grid, L2_grid, cost_matrices_no_constraint[idx].T, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Expected Cost ($)')
#    plt.xlabel('Market Orders (M)')
    plt.xlabel('Limit Orders at Level 1 (L1)')
    plt.ylabel('Limit Orders at Level 2 (L2)')
    plt.title(f'Expected Cost without Constraint (M = {M} and ξ = {mean_xi})')
    plt.plot()
