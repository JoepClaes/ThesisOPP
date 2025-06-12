#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:52:38 2025

@author: jawclaes
"""

import numpy as np
import math
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar


# Define variables
h = 0.02
r = 0.002
f = 0.003
delta = 0.01
theta = 0.01

# Limit order book state
Q1 = 2000
Q2 = 2000

# Queue outflows
l_xi = 1000
sig_xi = 300
l_psi1 = 1500
sig_psi1 = 100

def convolution_expon_CDF(x, rate1, rate2):
    return 1/(rate1-rate2) * (rate1 * np.exp(-rate2 * x) - rate2 * np.exp(-rate1 * x)) + 1

# Our target
S = 1000

# Define number of price levels + 1, X = (M,L1,L2,...)
n = 3
X = np.zeros(n)

# Define rho_u
#rho_max = (2*h+delta+f+r)/(norm.cdf(Q1+Q2,l_xi+l_psi1,sig_xi+sig_psi1))-h-r-delta-theta
#rho_min = (2*h+delta+f+r)/(norm.cdf(Q1+Q2+S,l_xi+l_psi1,sig_xi+sig_psi1))-h-r-delta-theta
rho_max = (2*h+delta+f+r)/(convolution_expon_CDF(Q1+Q2,l_xi, l_psi1))-h-r-delta-theta
rho_min = (2*h+delta+f+r)/(convolution_expon_CDF(Q1+Q2+S,l_xi, l_psi1))-h-r-delta-theta
#rho_u = (rho_max+rho_min)/2
rho_u = 0.03

check = convolution_expon_CDF(Q1,l_xi, l_psi1)

# Fraction inside the inverse CDF, should be <1
frac = (2*h + delta + f + r)/(h + delta + r + rho_u + theta)

# Formula for optimal split M*
M = S - norm.ppf(frac, l_xi+l_psi1, sig_xi+sig_psi1) + Q1 + Q2
M = max(0, min(S,M))

print(M)
M_prob = norm.ppf(frac, l_xi+l_psi1, sig_xi+sig_psi1)


# Define the equation to solve for L1
def solve_L1(L1):
#    F_xi_L1 = norm.cdf(Q1 + L1, l_xi, scale=sig_xi)
#    F_xi_plus_psi_L1 = norm.cdf(Q1 + Q2 + L1, l_xi + l_psi1, sig_xi + sig_psi1)
    F_xi_L1 = expon.cdf(Q1 + L1, loc=0, scale=l_xi)
    F_xi_plus_psi_L1 = convolution_expon_CDF(Q1 + Q2 + L1, l_xi, l_psi1)
    
    # Equation for dV/dL1
    lhs = -(h + delta + r + rho_u + theta) * F_xi_plus_psi_L1
    lhs += (h + r + (rho_u + theta) * frac) * F_xi_L1
    lhs += delta
    return lhs

# Check the signs of f(0) and f(S)
f0 = solve_L1(0)
fS = solve_L1(S)

if np.sign(f0) * np.sign(fS) < 0:
    # If signs are different, we can find a root
    root_result = root_scalar(solve_L1, bracket=[0, S], method='brentq')
    if root_result.converged:
        L1 = root_result.root
        print(f"Root found. Solution for L1: {L1}")
    else:
        print("Root finding failed unexpectedly.")
else:
    # If no root exists, minimize |equation(L1)| within [0, S]
    minimize_result = minimize_scalar(lambda L1: abs(solve_L1(L1)), bounds=(0, S), method='bounded')
    if minimize_result.success:
        L1 = minimize_result.x
        print(f"No root found in [0, S]. L1 that minimizes the equation: {L1}")
    else:
        print("Minimization failed unexpectedly.")

L2 = S-M-L1

X[0] = M
X[1] = L1
X[2] = L2

print(X)

l1 = np.linspace(-2000,3*S,1000)
dV_dL1 = np.zeros(1000)
for i in range(len(l1)):
    dV_dL1[i] = solve_L1(l1[i])
    
plt.plot(l1,dV_dL1)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label="y=0")  # Horizontal red dashed line
plt.title('Cost function derivative')
plt.xlabel('L1')
plt.ylabel('dV/dL1')
plt.show()




