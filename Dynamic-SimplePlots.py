#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:44:46 2025

@author: jawclaes
"""


# THIS SCRIPT PLOTS THE OPTIMAL DEPTHS FOR ANY INVENOTRY SIZE
# BELOW THE SYS.EXIT WE CAN ALSO PLOT THE OPTIMAL DEPTH FOR ANY DRIFT
# PLOT THE F(S,T) YOU ARE INTERESTED IN IF THE NUMERICAL ROOT FINDER DOESNT WORK TO FIND THE ROOT MANUALLY

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.optimize import brentq


# Set the parameters of our system (all values in $ and seconds)
R = 6
T = 60
kappa = 100
Delta = 0.01
lam_start = 50/60 
rho_u = 0.005 + 0.001
theta = 0.00001 # Was 0.00001
r = 0 # Was 0.003
f = 0 # Was 0.002
lam = lam_start * np.exp(kappa * (Delta + r + f) - 1)


# Example usage
p0 = 60        # Initial price
mu = 0    # Drift (example value)
sig_p = 0.01      # Total time (e.g., 1 unit)
N = 1000       # Number of steps
n_sim = 3

S=6

# Define the stochastic price movement

def simulate_midprice(p0, mu, sig_p, T, N):
    """
    Simulate midprice path with p0 = initial price, mu = drift, sig_p = volatility,
    T = total time, N = number of steps.
    
    Parameters:
    - p0 (float): Initial midprice (e.g., 30)
    - mu (float): Drift parameter
    - sig_p (float): Volatility parameter
    - T (float): Total time horizon
    - N (int): Number of time steps
    
    Returns:
    - prices (numpy array): Array of simulated midprices over time
    """
    dt = T / N  # Time step size
    # Generate N Wiener process increments: N(0, dt)
    dW = np.random.normal(0, np.sqrt(dt), N)
    # Compute price changes: drift + volatility * stochastic term
    dp = mu * dt + sig_p * dW
    # Initialize price array, starting with p0
    prices = np.zeros(N + 1)
    prices[0] = p0
    # Cumulative sum of price changes, added to initial price
    prices[1:] = p0 + np.cumsum(dp)
    return prices


# Simulate the midprice path
prices = np.zeros((N+1,n_sim))
for i in range(n_sim):
    prices[:,i] = simulate_midprice(p0, mu, sig_p, T, N)

# Time points for plotting
time = np.linspace(0, T, N + 1)


def f_theta(i, theta):
    return theta * i**2


def A(i, kappa, mu, theta):
    A = kappa * (i*mu - f_theta(i,theta))
    return A

A_values = np.zeros(20)
for i in range(20):
    A_values[i] = A(i, kappa, mu, theta)


# Base case for tau_1
TAU_1 = T  # Adjust this to your known \(\tau_1\)
tau_cache = {1: TAU_1}

# Define f_S(t) based on the assumed equation
def f_S(S, t):
    if S < 2:
        raise ValueError("f_S is defined for S >= 2")
    A_S = A(S, kappa, mu, theta)
    A_S_1 = A(S - 1, kappa, mu, theta)
    denominator = A_S - A_S_1 + lam
    if denominator == 0:
        raise ValueError("Denominator cannot be zero")
    
    if S == 2:
        f = g_S(S - 1, t) - (lam / denominator) 
    else:
        f = g_S(S - 1, t) - (lam / denominator) * g_S(S - 2, t)
    
    return f



# Solve for tau_S using Newton-Raphson
def solve_tau_S(S, initial_guess = 0, tol=1e-6, max_iter=10000):
    if S in tau_cache:
        return tau_cache[S]
    if S==1:
        tau_S = 60
        tau_cache[S] = tau_S
        return tau_S
    
    
    def f(t):
        return f_S(S, t)
    
    if f(0)<0:
        tau_S = 0
        tau_cache[S] = tau_S
        return tau_S
    else:
        
        try:
            tau_S = newton(f, initial_guess, tol=tol, maxiter=max_iter)
            if tau_S >= tau_cache[S-1]:
                try:
                    tau_S = brentq(f, 0, tau_cache[S-1], xtol=tol)
                except ValueError as e:
                    tau_S = tau_cache[S-1]
                    tau_cache[S] = tau_S
                return tau_S
            tau_cache[S] = tau_S
            return tau_S
        except RuntimeError as e:
            print(f"Failed to converge for S={S}: {e}")
        return None

# Define tau(i)
def tau(i):
    if i == 1:
        return TAU_1
    elif i >= 2:
        return solve_tau_S(i)
    else:
        print("tau: "+str(i))
        raise ValueError("Index i must be positive")
        

# Compute the coefficient term \frac{\bar{\lambda}_v^i}{\prod_{j=1}^i (A_S - A_{S-j})}
def compute_coeff(i, S, lam):
    if i == 0:
        return 1
    denominator = 1
    for j in range(1, i + 1):
        diff = A_values[S] - A_values[S - j]
        if diff == 0:
            raise ValueError(f"Denominator zero at S={S}, j={j}")
        denominator *= diff
    return (lam ** i) / denominator

# Forward declaration of g_S
def g_S(S, t):
    if S == 0:
        return 1  # Boundary condition: g_0(t) = 1
    if S == 1:
        A_1 = A_values[1]
        return np.exp(A_1 * (T - t)) * (1 + lam / A_1) - lam / A_1
    
    tau_S = tau(S)
    A_S = A_values[S]
    #g_so_far = g_all[:,:S]
    
    # First term: e^{A_S (\tau_S - t)} ( g_{S-1}(t) + \sum_{i=1}^S coeff_i g_{S-i}(\tau_S) )
    term1 = g_S(S - 1, tau_S)
    for i in range(1, S + 1):
        coeff = compute_coeff(i, S, lam)
        term1 += coeff * g_S(S - i, tau_S)
    term1 *= np.exp(A_S * (tau_S - t))
    
    # Second term: - \sum_{i=1}^S coeff_i g_{S-i}(t)
    term2 = 0
    for i in range(1, S + 1):
        coeff = compute_coeff(i, S, lam)
        term2 += coeff * g_S(S - i, t)
        #term2 += coeff * g_so_far[time.tolist().index(t),S-i]
    
    return term1 - term2


def depth_i1(t, kappa, r, f, mu, theta, T, lam):
    delta = 0
    delta = 1/kappa - Delta - r - f + 1/kappa * np.log( np.exp(A(1, kappa, mu, theta) * (T-t)) * (1 + lam/A(1, kappa, mu, theta)) - lam/A(1, kappa, mu, theta) )
    
    # INSERT OPTIMAL DEPTH FORMULA HERE
    
    return delta

def TWAP(t,R,T):
    return R* (T-t)/T

twap_schedule = np.zeros(N+1)
for i in range(N):
    twap_schedule[i] = TWAP(i*T/N,R,T)




# Compute f_S(S, t) for each t value
f_values = [f_S(S, t) for t in time]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(time, f_values, label=f'f_S({S}, t)')
plt.axhline(0, color='black', linestyle='--', label='y=0')  # Line to spot roots
plt.xlabel('Time (t)')
plt.ylabel(f'f_S({S}, t)')
plt.title(f'Plot of f_S for S={S}')
plt.legend()
plt.grid(True)
plt.show()


#sys.exit()

g_all = np.zeros((N+1,R))
for x in range(R):
    for i in range(N+1):
        g_all[i,x] = g_S(x+1, time[i])
    print(x)

w_all = np.zeros((N+1,R))
for x in range(R):
    for i in range(N+1):
        if x == 0:
            w_all[i,x] = g_all[i,x]
        else:
            tau_x = tau_cache[x+1]
            if time[i] < tau_x:
                w_all[i,x] = g_all[i,x]
            else:  # t[i] >= tau_3
                w_all[i,x] = w_all[i,x-1]

depths_all = np.zeros((N+1,R))
for x in range(R):
    for i in range(N+1):
        if x ==0:
            depths_all[i,x] = 1/kappa - Delta - r - f + 1/kappa * np.log(w_all[i,x])
        else:
            depths_all[i,x] = 1/kappa - Delta - r - f + 1/kappa * np.log(w_all[i,x]/w_all[i,x-1])

plt.figure(figsize=(10, 6))
for x in range(R):
    #plt.plot(time,w_all[:,x], label='i='+str(x+1))
    plt.plot(time,depths_all[:,x], label='i='+str(x+1))

plt.title("Optimal placement depth at different inventories")
plt.legend()
plt.show()

#sys.exit()


plt.figure(figsize=(10, 6))
depths =np.zeros((7,N+1))
mu_val = [-0.0003, -0.0002, -0.0001, 0,  0.0001, 0.0002, 0.0003]
for x in range(len(mu_val)):
    mu = mu_val[x]
    print(mu)
    for i in range(N+1):
        depths[x,i] = depth_i1(time[i], kappa, r, f, mu, theta, T, lam)

    plt.plot(time, depths[x,:], label=f'$\mu$={mu}')


plt.xlabel('Time (t)')
plt.ylabel('Optimal Depth ($\delta_t^*$)')
plt.title("Optimal Placement Depth at Different Price-Drifts")
plt.legend()
plt.show()
