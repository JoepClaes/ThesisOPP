#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 16:40:03 2025

@author: jawclaes
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.optimize import brentq

# WE WISH TO PLOT THE REMAINING INVENTORY OVER A PERIOD OF TIME INDICATING WHEN A LO IS FILLED AND WHEN AN MO IS EXECUTED

# ALSO FOR 3 MONTE CARLO PRICE SIMULATIONS (WITH OR WITHOUT DRIFT) THE OPTIMAL DEPTH AS A FUNCITON OF TIME
# FOR THE ABOVE A PLOT SHOWING THE INVENTORY AS A FUNCTION OF TIME FOR ALL 3 CASES
# FOR THE ABOVE A PLOT SHOWING THE AVERAGE PRICE/SHARE PAID OVER TIME STARTS BAD THEN IMPROVES AS LOs ARE FILLED

# Set the parameters of our system (all values in $ and seconds)
R = 10
T = 60
kappa = 1000
Delta = 0.005 # was 0.005
lam_start = 50/60 
rho_u = 0.005 + 0.001
theta = 0.00005 # Was 0.00001
r = 0.003 # Was 0.003
f = 0.002 # Was 0.002
lam = lam_start * np.exp(kappa * (Delta + r + f) - 1)

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

# Example usage
p0 = 30        # Initial price
mu = 0.0001       # Drift (example value)
sig_p = 0.01      # Total time (e.g., 1 unit)
N = 1000       # Number of steps

# Simulate the midprice path
prices = simulate_midprice(p0, mu, sig_p, T, N)

# Time points for plotting
t = np.linspace(0, T, N + 1)

## Plot the simulated path
#plt.plot(t, prices, label=f'mu={mu}, sigma={sig_p}')
#plt.xlabel('Time')
#plt.ylabel('Midprice')
#plt.title('Simulated Midprice Path')
#plt.legend()
#plt.grid(True)
#plt.show()


# Set our start inventory
Inv_t = np.zeros(N)
Inv_t[0] = R

def f_theta(i, theta):
    return theta * i**2


def A(i, kappa, mu, theta):
    A = kappa * (i*mu - f_theta(i,theta))
    return A

A_values = np.zeros(20)
for i in range(10):
    A_values[i] = A(i, kappa, mu, theta)
    

frac_tau = np.zeros(20)
for x in range(1,20):
    frac_tau[x] = lam/(A_values[x]-A_values[x-1]+lam)

#plt.plot(A_values)
#plt.plot(frac_tau)
#plt.show()
#sys.exit()

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

def df_S(S, t):
    A_S = A(S, kappa, mu, theta)
    A_S_1 = A(S - 1, kappa, mu, theta)
    denominator = A_S - A_S_1 + lam
    
    if S == 2:
        df = deriv_g_S(S - 1, t) 
    else:
        df = deriv_g_S(S - 1, t) - (lam / denominator) * deriv_g_S(S - 2, t)
    
    return df

# Solve for tau_S using Newton-Raphson
def solve_tau_S(S, initial_guess = 0, tol=1e-6, max_iter=10000):
    if S in tau_cache:
        return tau_cache[S]
    
    def f(t):
        return f_S(S, t)
    
    def df(t):
        return df_S(S, t)
    
    try:
        tau_S = newton(f, initial_guess, tol=tol, maxiter=max_iter)
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
        
def deriv_g_S(i, t):
    dg_t = 0
    
    A_i = A(i, kappa, mu, theta)
    A_i_1 = A(i-1, kappa, mu, theta)
    tau_i = tau(i)
    
    if i == 1:
        dg_t = -A_i * np.exp(A_i * (T-t)) * (1 + lam/A_i) 
    elif i >= 1:
        term1 = -A_i * np.exp(A_i * (tau_i-t)) * ((1 + lam/(A_i-A_i_1)) * g_S(i-1, tau_i) + lam**2/(A_i * (A_i-A_i_1))) 
        term2 = -lam/(A_i-A_i_1) * deriv_g_S(i-1, t) 
        
        dg_t = term1 + term2
    else:
        print(i)
    
    return dg_t


def g_S(i, t):
    g_t = 0
    
    A_i = A(i, kappa, mu, theta)
    A_i_1 = A(i-1, kappa, mu, theta)
    tau_i = tau(i)
    
    if i == 1:
        g_t = np.exp(A_i * (T-t)) * (1 + lam/A_i) - lam/A_i
    elif i >= 1:
        term1 = np.exp(A_i * (tau_i-t)) * ((1 + lam/(A_i-A_i_1)) * g_S(i-1, tau_i) + lam**2/(A_i * (A_i-A_i_1))) 
        term2 = -(lam/(A_i-A_i_1) * g_S(i-1, t) + lam**2/(A_i * (A_i-A_i_1)))
        
        g_t = term1 + term2
    else:
        print(i)
    
    return g_t

def depth_i1(t, Delta, kappa, r, f, mu, theta, T, lam):
    delta = 0
    delta = 1/kappa + Delta + r + f + 1/kappa * np.log( np.exp(A(1, kappa, mu, theta) * (T-t)) * (1 + lam/A(1, kappa, mu, theta)) - lam/A(1, kappa, mu, theta) )
    
    # INSERT OPTIMAL DEPTH FORMULA HERE
    
    return delta

expressive = T 


termA = lam/(A(2, kappa, mu, theta)-A(1, kappa, mu, theta)) + lam/A(1, kappa, mu, theta)
termC = - np.log(termA)/A(1, kappa, mu, theta)
termD = lam/(A(1, kappa, mu, theta)) + 1
termB = - np.log(termD)/A(1, kappa, mu, theta)
expressive += termA + termB

check1 = mu + lam/kappa - f_theta(1,theta)
check2 = mu + lam/kappa + f_theta(1,theta) - f_theta(2,theta)
 
#print(T, tau(2), tau(3))


g1 = np.zeros(N+1)
for i in range(N+1):
    g1[i] = g_S(1, t[i])
    
#g2 = np.zeros(N+1)
#for i in range(N+1):
#    g2[i] = g_S(2, t[i])
    
#g3 = np.zeros(N+1)
#for i in range(N+1):
#    g3[i] = g_S(3, t[i])

#depths_i2 = np.zeros(N+1)
#for i in range(N+1):
#    depths_i2[i] = 1/kappa + r + f + 1/kappa * np.log(g2[i]/g1[i])

#f_tau = np.zeros(N+1)
#for i in range(N+1):
#    f_tau[i] = f_S(3, t[i])

#plt.plot(t,g1, label="g1")
#plt.plot(t,g2, label="g2")
#plt.plot(t,g3)
#plt.plot(t,f_tau, label="f_tau")
#plt.plot(t,depths_i2)
#plt.legend()
#plt.show()

#sys.exit()

depths =np.zeros((6,N+1))
mu_val = [-0.00003, -0.00002, -0.00001, 0.00001, 0.00002, 0.00003]
for x in range(len(mu_val)):
    mu = mu_val[x]
    print(mu)
    for i in range(N+1):
        depths[x,i] = depth_i1(t[i], Delta, kappa, r, f, mu, theta, T, lam)

    plt.plot(t, depths[x,:], label=f'mu={mu}')

plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('Optimal Depth (δ_t^*)')
plt.show()
