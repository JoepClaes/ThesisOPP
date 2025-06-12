#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:34:29 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES THE PLOT SHOWING THE OPTIMAL SPLIT IN THE 1PL CASE FOR DIFFERENT RHO_U VALUES AND DIFFERENT DISTRIBUTIONS FOR XI
# CURRENTLY WE SET THE EXPONENTIAL NORMAL AND PARETO DISTRIBUTION, ALL SO THAT THEIR AVERAGES EQUAL 2200


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, pareto

# Define variables
h = 0.02
r = 0.002
f = 0.003
delta = 0.01  # Unused in this script
theta = 0.0005

# Limit order book state
Q1 = 2000

# Queue outflows parameters
l_xi = 2200    # Mean for exponential and normal
sig_xi = 300   # Standard deviation for normal

# Our target
S = 1000

# Distribution parameters
exp_scale = l_xi          # Exponential scale (mean = 2200)
norm_mean = l_xi          # Normal mean
norm_std = sig_xi         # Normal standard deviation
pareto_alpha = 2          # Pareto shape parameter
pareto_xm = 1100          # Pareto scale, mean = xm * alpha / (alpha - 1) = 2200

# Define rho_u range based on exponential distribution
rho_max = (2*h + f + r) / expon.cdf(Q1, loc=0, scale=exp_scale) - h - r - theta
rho_min = (2*h + f + r) / expon.cdf(Q1 + S, loc=0, scale=exp_scale) - h - r - theta
rho_u = np.linspace(0.02, 0.065, 1000)

# Initialize arrays for M and L for each distribution
M_exp = np.zeros(1000)
L_exp = np.zeros(1000)
M_norm = np.zeros(1000)
L_norm = np.zeros(1000)
M_pareto = np.zeros(1000)
L_pareto = np.zeros(1000)

# Calculate optimal splits
for i in range(len(rho_u)):
    frac = (2 * h + f + r) / (h + r + rho_u[i] + theta)
    
    # Exponential distribution
    xi_inv_exp = expon.ppf(frac, loc=0, scale=exp_scale) if 0 < frac < 1 else (np.inf if frac >= 1 else 0)
    M_exp[i] = S - xi_inv_exp + Q1
    M_exp[i] = max(0, min(S, M_exp[i]))
    L_exp[i] = S - M_exp[i]
    
    # Normal distribution
    xi_inv_norm = norm.ppf(frac, loc=norm_mean, scale=norm_std) if 0 < frac < 1 else (np.inf if frac >= 1 else 0)
    M_norm[i] = S - xi_inv_norm + Q1
    M_norm[i] = max(0, min(S, M_norm[i]))
    L_norm[i] = S - M_norm[i]
    
    # Pareto distribution
    xi_inv_pareto = pareto.ppf(frac, b=pareto_alpha, scale=pareto_xm) if 0 < frac < 1 else (np.inf if frac >= 1 else 0)
    M_pareto[i] = S - xi_inv_pareto + Q1
    M_pareto[i] = max(0, min(S, M_pareto[i]))
    L_pareto[i] = S - M_pareto[i]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(rho_u, M_exp, label='M (Exponential)', color='blue', linestyle='--')
plt.plot(rho_u, L_exp, label='L (Exponential)', color='blue', linestyle='-')
plt.plot(rho_u, M_norm, label='M (Normal)', color='green', linestyle='--')
plt.plot(rho_u, L_norm, label='L (Normal)', color='green', linestyle='-')
plt.plot(rho_u, M_pareto, label='M (Pareto)', color='red', linestyle='--')
plt.plot(rho_u, L_pareto, label='L (Pareto)', color='red', linestyle='-')

plt.title("Optimal split (M,L) for 1 price level with different distributions")
plt.xlabel("ρ_u")
plt.ylabel("Number of orders")
plt.legend()
plt.grid(True)
plt.show()