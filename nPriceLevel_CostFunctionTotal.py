#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:43:38 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES A 3D PLOT OF THE COST FUNCTION W.R.T. 2 VARIABLES, FOR N PRICE LEVELS


import sys
import numpy as np
import matplotlib.pyplot as plt


# Set how many price levels you want to consider
n_price_levels = 1

def ind(i, xi, psi_n, Q_n, L_n):
    # Make all indicator functions
    ind_i = np.zeros(2)
    
    # Handle the case where n_price_levels is 1
    if i == 0:
        ind_i[0] = int(Q_n[0] <= xi <= Q_n[0] + L_n[0])
        ind_i[1] = int(Q_n[0] + L_n[0] <= xi)
    else:
        ind_i[0] = int(sum(Q_n[:i]) + sum(L_n[:i]) <= xi + sum(psi_n[:i]) <= sum(Q_n[:i]) + sum(L_n[:i+1]))
        ind_i[1] = int(sum(Q_n[:i]) + sum(L_n[:i+1]) <= xi + sum(psi_n[:i]))
    
    return ind_i

def OF(i, xi, psi_n, Q_n, L_n):
    filled = 0
    
     # Handle the case where n_price_levels is 1
    if int(i) == 0:
        filled = (xi - Q_n[i]) * ind(i, xi, psi_n, Q_n, L_n)[0]
        filled += (L_n[i]) * ind(i, xi, psi_n, Q_n, L_n)[1]
    else:
        filled = (xi + sum(psi_n[:i]) - sum(Q_n[:i]) - sum(L_n[:i])) * ind(i, xi, psi_n, Q_n, L_n)[0]
        filled += (L_n[i]) * ind(i, xi, psi_n, Q_n, L_n)[1]
    
    
    return filled

def A_n(n, xi, psi_n, M, Q_n, L_n):
    A = M
    for i in range(n):
        A += OF(i, xi, psi_n, Q_n, L_n)
    
    return A

#Define the cost function
def V(n_price_levels, M, L_n, Q_n, S, h, f, r, delta, rho_u, rho_o, theta, l_xi, l_psi_n, num_sim=500):
    
    # Generate random samples for xi and psi
    xi_samples = np.random.exponential(scale=l_xi, size=num_sim)
    psi_samples_n = np.zeros((n_price_levels, num_sim))
    for i in range(n_price_levels):
        psi_samples_n[i,:] = np.random.exponential(scale=l_psi_n[i], size=num_sim)

    # Initialize cost components
    term1 = (h + f) * M

    # Monte Carlo estimation of expectation terms
    term2, term3, term4, term5 = 0, 0, 0, 0
    for x in range (num_sim):
        xi = xi_samples[x] 
        psi = psi_samples_n[:,x]
        for i in range(n_price_levels):
            term2 += -(h + (i) * delta + r) * OF(i, xi, psi, Q_n, L_n)

        term3 += rho_u * (S - A_n(n_price_levels, xi, psi, M, Q_n, L_n)) * (1 - ind(n_price_levels-1, xi, psi, Q_n, L_n)[1])
        term4 += rho_o * (A_n(n_price_levels, xi, psi, M, Q_n, L_n)-S) * ind(n_price_levels-1, xi, psi, Q_n, L_n)[1]
        term5 += theta * (M + sum(L_n[:n_price_levels]) + (S - A_n(n_price_levels, xi, psi, M, Q_n, L_n)) * (1 - ind(n_price_levels-1, xi, psi, Q_n, L_n)[1]))
        
    # Average the Monte Carlo results
    term2 /= num_sim
    term3 /= num_sim
    term4 /= num_sim
    term5 /= num_sim
    
    #print("Average xi: "+str(sum(xi_samples)/num_sim))
    #print("Average psi_1: "+str(sum(psi_samples_n[0,:])/num_sim))
    
    # Combine terms
    cost = term1 + term2 + term3 + term4 + term5
    return cost

# Define variables
h = 0.02
r = 0.002
f = 0.003
delta = 0.01
theta = 0.0005

# Limit order book state
Q_n = np.array([2000,1000,1000,1000])

# Queue outflows
l_xi = 2200
l_psi_n = np.array([2000,1000,1000,1000])

# Our target
S = 1000

rho_u = 0.045
rho_o = 0

## PLOT THE EXPECTED COST FUNCTION FOR THE SPLIT (M, S-M) VS. M FOR 1 PRICE LEVEL
#meshgrid_points = 100

#M = np.linspace(0, S, meshgrid_points)
#L_n = np.array([730,100,0,0])

#cost = np.zeros(meshgrid_points)

#for i in range(len(M)):
#    L_n[0] = S-M[i]
#    print(i)
#    cost[i] = V(n_price_levels, M[i], L_n, Q_n, S, h, f, r, delta, rho_u, rho_o, theta, l_xi, l_psi_n)

#plt.plot(M,cost)
#plt.title("Expected cost for split (M,S-M)")
#plt.xlabel("Expected cost")
#plt.ylabel("M")
#plt.show()

#sys.exit()

# Calculate V(M, L1) while varying M and L1 
meshgrid_points = 100

M = np.linspace(0, S, meshgrid_points)
L1 = np.linspace(0, S, meshgrid_points)

cost = np.zeros((len(M),len(L1)))
for i in range(meshgrid_points):
    for j in range(meshgrid_points):
        L_n = np.array([L1[j],100,0,0])
        cost[i,j] = V(n_price_levels, M[i], L_n, Q_n, S, h, f, r, delta, rho_u, rho_o, theta, l_xi, l_psi_n)
    print(f"Completed row {i+1}")

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Use scatter to plot only valid points (each (M[i], L2[j]) with corresponding cost)
M_grid, L2_grid = np.meshgrid(M, L1)  # Create a 2D grid from M and L2 for plotting
ax.scatter(M_grid, L2_grid, cost, c=cost, cmap='viridis', marker='o', s=2)

# Add labels and title
ax.set_title("3D Plot of V(M, L2)")
ax.set_xlabel("M")
ax.set_ylabel("L2")
ax.set_zlabel("V(M, L2)")

# Show the plot
plt.show()


