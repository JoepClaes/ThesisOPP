#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:25:07 2025

@author: jawclaes
"""

# THIS SCRIPT PLOTS THE OPTIMAL DEPTH AND G-FUNCTIONS FOR DIFFERENT INVENTORY SIZES
# IT USES THE SIMPLE NO RUNNING/REMAINING PENALTY MODEL FROM SECTION 5.2

import numpy as np
import matplotlib.pyplot as plt

# Set the parameters of our system (all values in $ and seconds)
T = 60
kappa = 100
h = 0.005
lam_start = 50 / 60
r = 0.003
f = 0.002
lambda_tilde = lam_start * np.exp(kappa * (2*h+r+f) - 1)

print(str(2*h+f+r)+" should be smaller than" + str(1/kappa))

# Time array
timesteps = 500
t = np.linspace(0, T*(timesteps)/timesteps, timesteps)
# Function to compute g_S(t)
def g_S(t, S):
    result = 0
    for j in range(S + 1):
        result += (lambda_tilde ** j / np.math.factorial(j)) * (T - t) ** j
    return result

# Function to compute delta^*(t, S)
def delta_star(t, S):
    if S <= 0:
        return np.nan
    g_S_val = g_S(t, S)
    g_S_minus_1_val = g_S(t, S - 1) if S > 1 else 1
    log_term = (1 / kappa) * np.log(g_S_val / g_S_minus_1_val)
    return (1 / kappa) + log_term - 2 * h - r - f

# Compute delta for different S values
S_values = range(1, 9)  # S = 1, 2, ..., 8
delta_curves = {S: delta_star(t, S) for S in S_values}
g_curves = {S: g_S(t, S) for S in S_values}

# Compute y = e^(lambda_tilde * (T - t))
exp_curve = np.exp(lambda_tilde * (T - t))

# Plotting g_S(t) with the exponential curve
plt.figure(figsize=(10, 6))
for S in S_values:
    plt.plot(t, g_curves[S], label=f'S = {S}')
plt.plot(t, exp_curve, label='y = e^(\u03BB̃(T-t))', linestyle='--', color='black')
plt.xlabel('Time (t) [seconds]')
plt.ylabel('g_S(t)')
plt.yscale('log')
plt.title('g_S(t) vs Time for Different Inventory Levels S')
plt.legend()
#plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(10, 6))
for S in S_values:
    plt.plot(t, delta_curves[S], label=f'S = {S}')
thresh = (1 / kappa) - 2*h - r - f
#plt.axhline(y=thresh, color='r', linestyle='--')
plt.title("Optimal Placement Depth at Different Inventories")
plt.xlabel('Time (t)')
plt.ylabel('Optimal Depth ($δ_t^*$)')
plt.legend()
#plt.grid(True)
plt.show()