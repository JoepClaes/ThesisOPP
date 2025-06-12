#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 22:14:11 2025

@author: jawclaes
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.optimize import brentq

# Set the parameters of our system (all values in $ and seconds)
R = 8
T = 60
kappa = 100
Delta = 0.01
lam_start = 50/60 
rho_u = 0.005 + 0.001
theta = 0.00001
r = 0
f = 0
lam = lam_start * np.exp(kappa * (Delta + r + f) - 1)

# Define the stochastic price movement
def simulate_midprice(p0, mu, sig_p, T, N):
    dt = T / N
    dW = np.random.normal(0, np.sqrt(dt), N)
    dp = mu * dt + sig_p * dW
    prices = np.zeros(N + 1)
    prices[0] = p0
    prices[1:] = p0 + np.cumsum(dp)
    return prices

# Example usage
p0 = 60
sig_p = 0.01
N = 10000
n_sim = 3

# Define functions
def f_theta(i, theta):
    return theta * i**2

def A(i, kappa, mu, theta):
    return kappa * (i * mu - f_theta(i, theta))

def f_S(S, t, A_values, lam):
    if S < 2:
        raise ValueError("f_S is defined for S >= 2")
    A_S = A_values[S]
    A_S_1 = A_values[S - 1]
    denominator = A_S - A_S_1 + lam
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    
    if S == 2:
        f = g_S(S - 1, t, A_values, lam) - (lam / denominator)
    else:
        f = g_S(S - 1, t, A_values, lam) - (lam / denominator) * g_S(S - 2, t, A_values, lam)
    
    return f

def solve_tau_S(S, A_values, lam, tol=1e-8):
    def f(t):
        return f_S(S, t, A_values, lam)
    try:
        tau_S = brentq(f, 0, T, xtol=tol)
        return tau_S
    except ValueError:
        # No root in [0, T], or f(0) and f(T) have same sign
        return 0  # Or T, depending on context

#def solve_tau_S(S, A_values, lam, initial_guess=0, tol=1e-6, max_iter=10000):
#    if S in tau_cache:
#        return tau_cache[S]
#    
#    def f(t):
#        return f_S(S, t, A_values, lam)
#    
#    try:
#        tau_S = newton(f, initial_guess, tol=tol, maxiter=max_iter)
#        tau_cache[S] = tau_S
#        return tau_S
#    except RuntimeError as e:
#        print(f"Failed to converge for S={S}: {e}")
#        return None

def tau(i, A_values, lam):
    if i == 1:
        return TAU_1
    elif i >= 2:
        return solve_tau_S(i, A_values, lam)
    else:
        print("tau: "+str(i))
        raise ValueError("Index i must be positive")

def compute_coeff(i, S, lam, A_values):
    if i == 0:
        return 1
    denominator = 1
    for j in range(1, i + 1):
        diff = A_values[S] - A_values[S - j]
        if diff == 0:
            raise ValueError(f"Denominator zero at S={S}, j={j}")
        denominator *= diff
    return (lam ** i) / denominator

def g_S(S, t, A_values, lam):
    if S == 0:
        return 1
    if S == 1:
        A_1 = A_values[1]
        return np.exp(A_1 * (T - t)) * (1 + lam / A_1) - lam / A_1
    
    tau_S = tau(S, A_values, lam)
    A_S = A_values[S]
    term1 = g_S(S - 1, tau_S, A_values, lam)
    for i in range(1, S + 1):
        coeff = compute_coeff(i, S, lam, A_values)
        term1 += coeff * g_S(S - i, tau_S, A_values, lam)
    term1 *= np.exp(A_S * (tau_S - t))
    
    term2 = 0
    for i in range(1, S + 1):
        coeff = compute_coeff(i, S, lam, A_values)
        term2 += coeff * g_S(S - i, t, A_values, lam)
    
    return term1 - term2

def depth_i1(t, kappa, r, f, mu, theta, T, lam):
    delta = 1/kappa + r + f + 1/kappa * np.log(np.exp(A(1, kappa, mu, theta) * (T-t)) * (1 + lam/A(1, kappa, mu, theta)) - lam/A(1, kappa, mu, theta))
    return delta

# Define mu values to test
mu_stopping = [-0.0003, -0.00025, -0.0002, -0.00015, -0.0001, -0.00005, 0]
inventory = np.array(range(1, R + 1))
tau_all_mu = np.zeros((len(mu_stopping), R))  # Store stopping times for each mu


# Compute stopping times for each mu
for mu_idx, mu in enumerate(mu_stopping):
    # Reset tau_cache for each mu
    TAU_1 = T
    tau_cache = {1: TAU_1}
    
    # Compute A_values for current mu
    A_values = np.zeros(20)
    for i in range(20):
        A_values[i] = A(i, kappa, mu, theta)
    
    # Compute stopping times for each inventory
    for i in inventory:
        tau_i = tau(i, A_values, lam)
        if tau_i is None or tau_i <= 0:
            tau_all_mu[mu_idx, i-1] = 0
        else:
            tau_all_mu[mu_idx, i-1] = tau_i
        print(f"mu={mu}, Inventory={i}, Stopping Time={tau_i}")

# Plotting
plt.figure(figsize=(10, 6))
for mu_idx, mu in enumerate(mu_stopping):
    plt.plot(tau_all_mu[mu_idx, :], inventory, 'o', label=f'μ={mu}')
plt.title("Optimal Stopping Times at Different Inventories for Various μ")
plt.xlabel('Time (t)')
plt.ylabel('Inventory')
plt.legend()
plt.grid(True)
plt.show()