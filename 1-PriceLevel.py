#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:47:26 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES THE PLOT SHOWING THE OPTIMAL SPLIT IN THE 1PL CASE FOR DIFFERENT RHO_U VALUES


import numpy as np
import math
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize_scalar


linspace_size = 1000

# Define variables
h = 0.02
#r = np.linspace(-0.02,0.02,linspace_size)
r = 0.002
#f = np.linspace(-0.005,0.01,linspace_size)
f = 0.003
delta = 0.01
theta = 0.0005


# Limit order book state
Q1 = 2000

# Queue outflows
l_xi = np.linspace(0,4000,linspace_size)
#l_xi = 2200
sig_xi = 300

# Our target
#S = np.linspace(0,2000,linspace_size)
S = 1000

# Define rho_u
#rho_max = (2*h+f+r)/(norm.cdf(Q1,l_xi,sig_xi))-h-r-theta
#rho_min = (2*h+f+r)/(norm.cdf(Q1+S,l_xi,sig_xi))-h-r-theta
rho_max = (2*h+f+r)/(expon.cdf(Q1,loc=0, scale=l_xi))-h-r-theta
rho_min = (2*h+f+r)/(expon.cdf(Q1+S,loc=0, scale=l_xi))-h-r-theta
#rho_u = (rho_max+rho_min)/2
#rho_u = np.linspace(0.03,0.06,1000)
rho_u = 0.043

M = np.zeros(linspace_size)
L = np.zeros(linspace_size)

for i in range(linspace_size):
    # Fraction inside the inverse CDF, should be <1
    frac = (2*h+f+r)/(h+r+rho_u+theta)
    
    # Formula for optimal split M*
#    M[i] = S - norm.ppf(frac, l_xi, sig_xi) + Q1 
    M[i] = S - expon.ppf(frac, loc=0, scale=l_xi[i]) + Q1
    print(M[i])
    M[i] = max(0, min(S,M[i]))
#    M_prob = norm.ppf(frac, l_xi, sig_xi)
    M_prob = expon.ppf(frac, loc=0, scale=l_xi[i])
    
    L[i] = S-M[i]
#print("M:",M)
#print("L:",L)
plt.plot(l_xi,M, label = 'M')
plt.plot(l_xi,L,label='L')
plt.title("Optimal split (M,L) for 1 Price-Level at Different Mean Values of ξ")
plt.xlabel("Mean of ξ")
plt.grid("True")
plt.legend()
plt.show()

