#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 11:36:37 2025

@author: jawclaes
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters from your thesis Section 5.2
T = 60  # Terminal time in seconds
kappa = 100  # Order book decay rate
h = 0.002  # Half-spread
f = 0.002  # Fee
r = 0.003  # Rebate or reference adjustment
lam_start = 50 / 60  # Initial intensity (events per second)
lambda_tilde = lam_start * np.exp(kappa * r - 1)  # Adjusted intensity
lambda_mu = lambda_tilde  # Using lambda_tilde as lambda_mu
tilde_lambda_mu = (lambda_mu / np.e) * np.exp(kappa * (2*h+r+f))  # Corrected tilde_lambda_mu

# Time array
t = np.linspace(0, T, 1000)

# Function to compute g_S(t)
def g_S(t, S):
    result = 0
    for j in range(S + 1):
        term = (tilde_lambda_mu ** j) / np.math.factorial(j) * (T - t) ** j
        result += term
    return result

# Compute g_S(t) for S = 1 to 8
S_values = range(1, 9)  # S = 1, 2, ..., 8
g_curves = {S: g_S(t, S) for S in S_values}

# Plotting
plt.figure(figsize=(10, 6))
for S in S_values:
    plt.plot(t, g_curves[S], label=f'S = {S}')
plt.xlabel('Time (t) [seconds]')
plt.ylabel('g_S(t)')
plt.title('g_S(t) vs Time for Different Inventory Levels S')
plt.legend()
plt.grid(True)
plt.show()