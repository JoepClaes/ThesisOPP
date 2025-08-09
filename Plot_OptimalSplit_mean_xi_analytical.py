#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:36:29 2025

@author: jawclaes
"""


# THIS SCRIPT CREATES THE PLOT FOR THE OPTIMAL SPLIT FOR 2 PL USING THE ANALYITCAL SOLUTION FOR DIFFERENT MEAN_XI OR MEAN_PSI

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.optimize import root_scalar
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
delta = 0.01
num_sim = 100  # Number of simulations

aa_rho_u = 0

# Function to compute optimal allocations for a given mean_xi
def compute_optimal(mean_xi, mean_psi):
    lambda_xi = 1 / mean_xi
    lambda_psi = 1 / mean_psi  # Rate parameter for psi
    # Generate samples
    xi_samples = np.random.exponential(scale=mean_xi, size=num_sim)
    psi_samples = Q2 - np.random.exponential(scale=mean_psi, size=num_sim)
    sum_samples = xi_samples + psi_samples

    # Empirical CDF functions
    def F_xi(x):
        return 1 - np.exp(-lambda_xi * x)
    
    def F_xi_psi(x):
        convo = 0
        if x <= Q2:
            convo = lambda_xi/(lambda_xi+lambda_psi) * np.exp(-lambda_psi * (Q2-x))
        elif x > Q2:
            convo = 1 - lambda_psi/(lambda_xi+lambda_psi) * np.exp(-lambda_xi * (x-Q2))
        else:
            print("huh")
        return convo

    # Compute thresholds from Proposition 4.5
    F_Q1_Q2_S = F_xi_psi(Q1 + Q2 + S)
    F_Q1_Q2 = F_xi_psi(Q1 + Q2)
    p = (2 * h + delta + f + r) / (h + delta + r + rho_u + theta)
    p_u_lower = aa_rho_u
    #p_u_lower = (2 * h + delta + f + r) / F_Q1_Q2_S - (h + r + theta) if F_Q1_Q2_S > 0 else np.inf
    #p_u_upper = 2
    p_u_upper = (2 * h + delta + f + r) / F_Q1_Q2 - (h + r + theta) if F_Q1_Q2 > 0 else np.inf
    
    # Determine case based on rho_u
    if rho_u <= p_u_lower:
        M_star = 0
        L1_star = 0
        L2_star = S
    elif rho_u >= p_u_upper:
        M_star = S
        L1_star = 0
        L2_star = 0
    else:
        # Case (iii): Mixed allocation

        F_Q2 = lambda_xi / (lambda_xi + lambda_psi)  # F_xi_psi at Q2
        if p <= F_Q2:
            A = Q2 + (1 / lambda_psi) * np.log(lambda_xi / (p * (lambda_xi + lambda_psi)))
        else:
            A = Q2 - (1 / lambda_xi) * np.log((1 - p) * (lambda_xi + lambda_psi) / lambda_psi)
        M_star = S - A + Q1 + Q2
        M_star = max(0, min(S, M_star))  # Constrain M_star first

        # Solve for L1_star
        def g(L1):
            term1 = -(h + delta + r + rho_u + theta) * F_xi_psi(Q1 + Q2 + L1)
            term2 = (h + r + rho_u + theta) * F_xi(Q1 + L1)
            term3 = delta
            return term1 + term2 + term3

        try:
            sol = root_scalar(g, bracket=[0, S - M_star], method='brentq')
            #L1_star = max(0, min(S - M_star, sol.root))  # Constrain root
            L1_star = max(0,  sol.root)# Constrain root
            print(f"mean_xi={mean_xi}, L1_star={L1_star}")
        except ValueError:
            if g(0) > 0:
                L1_star = 0
            elif g(S) < 0:
                L1_star = S - M_star
            else:
                L1_star = 500
            #print(f"mean_xi={mean_xi}, No root, L1_star={L1_star}")

        L2_star = max(0, S - M_star - L1_star)

    return M_star, L1_star, L2_star

# Range of mean_xi to explore
mean_xi_range = np.linspace(2000, 6000, 1000)
mean_xi = 2800

mean_psi_range = np.linspace(1, 1500, 1000)
mean_psi = 0.5*Q2 # Fixed mean for psi

# Store results
M_values = []
L1_values = []
L2_values = []

# Compute optimal allocations for each mean_xi
for mean_psi in mean_psi_range:
    M, L1, L2 = compute_optimal(mean_xi,mean_psi)
    M_values.append(M)
    L1_values.append(L1)
    L2_values.append(L2)
    

magic_value = 1/mean_psi * delta/(h+r+rho_u+theta)

print(1/magic_value)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_psi_range, M_values, label='M (Market Orders)', linewidth=2)
plt.plot(mean_psi_range, L1_values, label='L1 (Limit Orders Level 1)', linewidth=2)
plt.plot(mean_psi_range, L2_values, label='L2 (Limit Orders Level 2)', linewidth=2)
plt.xlabel('1/$\\lambda_{\\psi_1}$')
plt.ylabel('Optimal Allocation Size (0 to 1000)')
plt.title('Optimal Allocations vs Mean of $\\psi_1$')
plt.legend()
plt.grid(True)
plt.ylim(0, 1000)  # Set y-axis range as requested
plt.show()

sys.exit()

# Plot g(L1) for specified mean_xi values
mean_xi_list = [3400, 4500, 4600, 6700]
L1_range = np.linspace(0, S, 100)  # Range of L1 values from 0 to S
plt.figure(figsize=(10, 6))

for mean_xi in mean_xi_list:
    lambda_xi = 1 / mean_xi
    # Generate samples for this mean_xi
    xi_samples = np.random.exponential(scale=mean_xi, size=num_sim)
    psi_samples = np.random.exponential(scale=mean_psi, size=num_sim)
    sum_samples = xi_samples + psi_samples

    # Empirical CDF functions
    def F_xi(x):
        return np.mean(xi_samples <= x)
    
    def F_xi_psi(x):
        return np.mean(sum_samples <= x)

    # Compute p for g(L1)
    p = (2 * h + delta + f + r) / (h + delta + r + rho_u + theta)

    # Compute g(L1) for each L1 in L1_range
    g_values = []
    for L1 in L1_range:
        term1 = -(h + delta + r + rho_u + theta) * F_xi_psi(Q1 + Q2 + L1)
        term2 = (h + r) * F_xi(Q1 + L1)
        term3 = delta - (rho_u + theta) + 2 * (rho_u + theta) * p
        g_values.append(term1 + term2 + term3)

    # Plot g(L1) for this mean_xi
    plt.plot(L1_range, g_values, label=f'mean_ξ = {mean_xi}', linewidth=2)

# Add a horizontal line at y=0 to show where g(L1) crosses zero
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('L1')
plt.ylabel('g(L1)')
plt.title('g(L1) for Different Mean Values of ξ')
plt.legend()
plt.grid(True)
plt.show()