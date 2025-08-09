#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 22:11:56 2025

@author: jawclaes
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Q1 = 2000
Q2 = 1500
S = 1000
h = 0.02
r = 0.002
rho_u = 0.05
theta = 0.0005
delta = 0.01
f = 0.003
mean_psi = 0.5 * Q2  # 750
lambda_psi = 1 / mean_psi  # 1/750

# Compute p
p = (2 * h + delta + f + r) / (h + delta + r + rho_u + theta)

# Range of mean_xi (mean of xi = 1/lambda_xi)
mean_xi_range = np.linspace(2000, 6000, 1000)
lambda_xi_range = 1 / mean_xi_range

# Compute M_star for each lambda_xi
M_star_values = []

for lambda_xi in lambda_xi_range:
    # Evaluate the conditions for M_star
    lambda_ratio = lambda_xi / (lambda_xi + lambda_psi)
    threshold1 = lambda_ratio * (h + delta + r + rho_u + theta)  # First condition
    threshold2 = (lambda_xi + lambda_psi) / lambda_psi * (h + delta + r + rho_u + theta)  # Third condition
    
    term1 = 2 * h + delta + f + r
    term2 = h + delta + r + rho_u + theta
    
    if term1/term2 <= lambda_ratio:
        # First case
        M_star = S - (1 / lambda_psi) * np.log(lambda_ratio * term2/term1) + Q1
    elif term1/term2 > lambda_ratio:
        # Second case
        M_star = S - (1 / lambda_xi) * np.log((1-lambda_ratio) * term2/(term2-term1)) + Q1
    else:
        print("huh")
    
    # Constrain M_star between 0 and S
    M_star = max(0, min(S, M_star))
    M_star_values.append(M_star)

# Plot M_star vs mean_xi
plt.figure(figsize=(10, 6))
plt.plot(mean_xi_range, M_star_values, label='M* (Market Orders)', linewidth=2, color='blue')
plt.xlabel('Mean of ξ (1/λ_ξ)')
plt.ylabel('Optimal M* (Market Orders)')
plt.title('Optimal M* vs Mean of ξ')
plt.legend()
plt.grid(True)
plt.ylim(0, 1000)  # As per your previous plots
plt.show()

# Mark the specific lambda_xi = 1/5437.5
mean_xi_specific = 5437.5
lambda_xi_specific = 1 / mean_xi_specific

# Compute M_star at this specific point
lambda_ratio = lambda_xi_specific / (lambda_xi_specific + lambda_psi)
threshold1 = lambda_ratio * (h + delta + r + rho_u + theta)
threshold2 = (lambda_xi_specific + lambda_psi) / lambda_psi * (h + delta + r + rho_u + theta)

if (2 * h + delta + f + r) <= threshold1:
    M_star_specific = S - (1 / lambda_psi) * np.log(
        (lambda_xi_specific * (lambda_psi + lambda_xi_specific) * (2 * h + delta + f + r)) / 
        ((lambda_psi + lambda_xi_specific)**2 * (h + delta + r + rho_u + theta))
    ) + Q1
elif (2 * h + delta + f + r) <= threshold2:
    M_star_specific = S + (1 / lambda_xi_specific) * np.log(
        (lambda_psi * (lambda_psi + lambda_xi_specific) * (h + delta + r + rho_u + theta)) / 
        ((lambda_psi + lambda_xi_specific)**2 * (2 * h + delta + f + r))
    ) + Q1
else:
    M_star_specific = S - (1 / lambda_xi_specific) * np.log(
        (lambda_psi * (h + delta + r + rho_u + theta)) / 
        ((lambda_psi + lambda_xi_specific) * (h + r + rho_u + theta))
    ) + Q1

M_star_specific = max(0, min(S, M_star_specific))
print(f"At mean_xi = 5437.5 (lambda_xi = {lambda_xi_specific:.6f}), M_star = {M_star_specific:.2f}")