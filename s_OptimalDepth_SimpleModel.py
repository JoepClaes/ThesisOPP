#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:05:13 2025

@author: jawclaes
"""

# THIS SCRIPT PLOTS THE OPTIMAL DEPTH FOR THETA = 0 USING THE SIMPLE MODEL WITHOUT DRIFT OR RUNNING INVENTORY PENALTY

import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
R = 6
T = 60
kappa = 100
Delta = 0.01
lam_start = 50 / 60
rho_u = 0.005 
theta = 0
r = 0
f = 0

# Adjusted lambda
lam = lam_start * np.exp(kappa * (Delta + r + f) - 1)
lam_tilde = lam

# Time vector (include T=60 explicitly)
t_vals = np.linspace(0, T, 1000)
if T not in t_vals:
    t_vals = np.append(t_vals, T)
t_vals = np.sort(t_vals)

# Lower bound for delta*
delta_min = (1 / kappa) - r - f - Delta

# Function to compute delta*
def delta_star(t, S, T, kappa, Delta, r, f, lam_tilde):
    # Avoid division by zero when t == T
    T_minus_t = T - t
    numerator = (lam_tilde**S / math.factorial(S)) * T_minus_t**S
    denominator = sum((lam_tilde**j / math.factorial(j)) * T_minus_t**j for j in range(S)) 
    log_term = np.log(numerator / denominator + 1)
    delta = (1/kappa) - Delta - r - f + (1/kappa) * log_term
    return max(delta, delta_min)  # Clip from below

# Plotting
plt.figure(figsize=(10, 6))

for S in range(1, 9):  # S = 1 to 8
    delta_vals = [delta_star(t, S, T, kappa, Delta, r, f, lam_tilde) for t in t_vals]
    plt.plot(t_vals, delta_vals, label=f'S = {S}')

#plt.axhline(delta_min, color='gray', linestyle='--', linewidth=1, label=r'$\delta_{\min} = \frac{1}{\kappa} - r - f$')
plt.xlabel('Time ($t$)')
plt.ylabel(r'Optimal Depth ($\delta^*_t$)')
plt.title(r'Optimal Placement Depth at Different Inventories')
plt.legend()
#plt.grid(True)
#plt.tight_layout()
plt.show()
