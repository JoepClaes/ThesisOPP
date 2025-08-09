#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 11:19:01 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES A PLOT OF THE EXPECTED COST FOR DIFFERENT STRATEGIES FOR DIFFERENT VALUES OF MEAN_XI BASED ON THE ANALYTICAL SOLUTION
# IT CAN USE THE OPTIMAL ALLOCATION ADJUSTED TO KNOW HOW MANY PRICE LEVELS TO CONSIDER
# IT COMPARES TO ONLY MO OR LO STATS, BUT ALSO CONT AND KUKANOV 1 PL STRAT



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.stats import expon

# Parameters
S = 1000
h = 0.01
r = 0.002
f = 0.003
theta = 0.001
rho_u = 0.05
rho_o = 0.05
delta = 0.01
n = 5  # Number of price levels
num_samples = 200 

# Queue sizes: Q1 = 2000, Q2 = 1500, Q3 = 1400, Q4 = 1300, Q5 = 1200, Q6 = 1100
Q = [2000] + [1500 - 0 * i for i in range(n - 1)]
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
    if mean_xi == 0:
        # Handle mean_xi = 0 (no market flow) -> place all in highest level
        return 0, [0] * (n - 1) + [S]
    
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
        M_star = 0
        L_star = [0] * (n - 1) + [S]
    elif rho_u >= 2:
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
                if g(0) > 0:
                    L_i_star = 0
                elif g(remaining - sum(L_star)) < 0:
                    L_i_star = remaining - sum(L_star)
                else:
                    L_i_star = 0  # Default fallback
            
            L_i_star = max(0, min(remaining - sum(L_star), L_i_star))
            L_star.append(L_i_star)
            remaining -= L_i_star
        
        L_star.append(max(0, remaining))
    
    return M_star, L_star





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
mean_xi_list = np.linspace(100, 10000, 100)
cost_allo1 = np.zeros(len(mean_xi_list))
cost_allo2 = np.zeros(len(mean_xi_list))
cost_allo3 = np.zeros(len(mean_xi_list))
cost_allo4 = np.zeros(len(mean_xi_list))
cost_allo5 = np.zeros(len(mean_xi_list))
cost_optimal = np.zeros(len(mean_xi_list))  # For optimal strategy
OF1_allo2 = np.zeros(len(mean_xi_list))
OF2_allo3 = np.zeros(len(mean_xi_list))
OF3_allo4 = np.zeros(len(mean_xi_list))
OF4_allo5 = np.zeros(len(mean_xi_list))

for i in range(len(mean_xi_list)):
    
    # Generate samples for each mean_xi
    xi_samples = np.random.exponential(mean_xi_list[i], num_samples) if mean_xi_list[i] > 0 else np.zeros(num_samples)
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
    
    # Allocation 3: L5 only
    L[0] = 0
    L[4] = S
    cost_allo3[i], OFs_allo3 = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    OF2_allo3[i] = OFs_allo3[4]
    
    # Allocation 4: L3 only
    L[0] = 0
    L[1] = 0
    L[2] = S
    L[4] = 0
    cost_allo4[i], OFs_allo4 = expected_cost(0, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    OF3_allo4[i] = OFs_allo4[2]
    
    # Allocation 5: Cont and Kukanov
    frac = (2*h+f+r)/(h+r+rho_u+theta)
    M = S - expon.ppf(frac, loc=0, scale=mean_xi_list[i]) + Q[0]

    M = max(0, min(S,M))
    L[0] = S-M
    L[1] = 0 
    L[2] = 0
    cost_allo5[i], OFs_allo5 = expected_cost(M, L, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    
    # Optimal Allocation
    print(i)
    
    critical_point1 = (h+r+rho_u+theta) / (lambda_psi_list[0]*delta)
    critical_point2 = (h+delta+r+rho_u+theta) / (lambda_psi_list[1]*delta)
    critical_point3 = (h+2*delta+r+rho_u+theta) / (lambda_psi_list[2]*delta)
    critical_point4 = (h+3*delta+r+rho_u+theta) / (lambda_psi_list[3]*delta)
    #print(critical_point1)
    if mean_xi_list[i] <= critical_point1-10:
        M_opt, L_opt = compute_optimal(mean_xi_list[i], 1, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta)
        
        frac = (2*h+f+r)/(h+r+rho_u+theta)
        M = S - expon.ppf(frac, loc=0, scale=mean_xi_list[i]) + Q[0]

        M_opt = max(0, min(S,M))
        L_opt = np.zeros(n)
        L_opt[0] = S-M_opt
        
    elif critical_point1-1 <=mean_xi_list[i] <= critical_point2-1:
        Qn=[Q[0],Q[1]]
        lambda_psi_listn = [lambda_psi_list[0]]
        M_opt, L = compute_optimal(mean_xi_list[i], 2, Qn, lambda_psi_listn, S, h, r, f, theta, rho_u, delta)
        L_opt = np.zeros(n)
        L_opt[0] = L[0]
        L_opt[1] = L[1]
    elif critical_point2-1 <=mean_xi_list[i] <= critical_point3-1:
        Qn=[Q[0],Q[1],Q[2]]
        lambda_psi_listn = [lambda_psi_list[0],lambda_psi_list[1]]
        M_opt, L = compute_optimal(mean_xi_list[i], 3, Qn, lambda_psi_listn, S, h, r, f, theta, rho_u, delta)
        #print(mean_xi_list[i], 3, Qn, lambda_psi_listn, S, h, r, f, theta, rho_u, delta)
        #print(M_opt,L)
        L_opt = np.zeros(n)
        L_opt[0] = L[0]
        L_opt[1] = L[1]
        L_opt[2] = L[2]
    elif critical_point3-1 <=mean_xi_list[i] <= critical_point4-1:
        Qn=[Q[0],Q[1],Q[2],Q[3]]
        lambda_psi_listn = [lambda_psi_list[0],lambda_psi_list[1],lambda_psi_list[2]]
        M_opt, L = compute_optimal(mean_xi_list[i], 4, Qn, lambda_psi_listn, S, h, r, f, theta, rho_u, delta)
        L_opt = np.zeros(n)
        L_opt[0] = L[0]
        L_opt[1] = L[1]
        L_opt[2] = L[2]
        L_opt[3] = L[3]
    else:
        M_opt, L_opt = compute_optimal(mean_xi_list[i], n, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta)
#    M_opt, L_opt = compute_optimal(mean_xi_list[i], n, Q, lambda_psi_list, S, h, r, f, theta, rho_u, delta)
    cost_optimal[i], _ = expected_cost(M_opt, L_opt, xi_samples, psi_samples, Q, S, h, r, f, theta, rho_u, rho_o, delta)
    
    #print(f"Index {i}, mean_xi = {mean_xi_list[i]:.2f}")

# Plotting costs
plt.figure(figsize=(10, 6))
plt.plot(mean_xi_list, cost_allo1, label='Only Market Orders')
plt.plot(mean_xi_list, cost_allo2, label='Only L1 Orders')
plt.plot(mean_xi_list, cost_allo4, label='Only L2 Orders')
plt.plot(mean_xi_list, cost_allo3, label='Only L3 Orders')
plt.plot(mean_xi_list, cost_allo5, label='Cont and Kukanov')
plt.plot(mean_xi_list, cost_optimal, label='Static Optimal Strategy (Adjusted)', linewidth=2, linestyle='--')
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
plt.title('Expected Orders Filled for L1, L2, and L3 Allocations')
plt.legend()
plt.grid(True)
plt.show()