
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:34:11 2025

@author: jawclaes
"""

# THIS SCRIPT CREATES A PLOT OF THE OPTIMAL SPLIT FOR 3 PL FOR DIFFERENT VALUES OF MEAN_XI 
# BASED ON THE ANALYTICAL EXPRESSION FOR THE CONVOLUTION OF XI PSI AND PSI2



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.optimize import root_scalar
import sys

# Parameters
Q1 = 2000
Q2 = 1500
Q3 = 1400
S = 1000
h = 0.02
r = 0.002
f = 0.003
theta = 0.0005
rho_u = 0.05
delta = 0.01
mean_psi = 0.5*Q2 # Fixed mean for psi
lambda_psi = 1 / mean_psi  # Rate parameter for psi
mean_psi2 = 0.5*Q3 # Fixed mean for psi
lambda_psi2 = 1 / mean_psi2  # Rate parameter for psi
num_sim = 1000  # Number of simulations

x = (lambda_psi2 * delta) / (h+delta+r+rho_u+theta)
print(1/x)

# Function to compute optimal allocations for a given mean_xi
def compute_optimal(mean_xi):
    
    lambda_xi = 1 / mean_xi
    lambda_psi = 1 / mean_psi
    lambda_psi2 = 1 / mean_psi2
    # Generate samples
    xi_samples = np.random.exponential(scale=mean_xi, size=num_sim)
    psi_samples = Q2 - np.random.exponential(scale=mean_psi, size=num_sim)
    sum_samples = xi_samples + psi_samples

    # Empirical CDF functions
    def F_xi(x):
        return 1 - np.exp(-lambda_xi * x)
    
    def F_xi_psi(x):
        convo = 0
        if x <= Q2:
            convo = lambda_xi/(lambda_xi+lambda_psi) * np.exp(-lambda_psi * (Q2-x))
        elif x > Q2:
            convo = 1 - lambda_psi/(lambda_xi+lambda_psi) * np.exp(-lambda_xi * (x-Q2))
        else:
            print("huh")
        return convo
    
    def F_xi_psi_psi2(x):
        convo2 = 0
        if x <= Q2+Q3:
            convo2 += (lambda_xi*lambda_psi)/((lambda_xi+lambda_psi2)*(lambda_psi-lambda_psi2)) * np.exp(-lambda_psi2 * (Q2+Q3-x))
            convo2 += -(lambda_xi*lambda_psi2)/((lambda_xi+lambda_psi)*(lambda_psi-lambda_psi2)) * np.exp(-lambda_psi * (Q2+Q3-x))
        elif x > Q2+Q3:
            convo2 = 1 - (lambda_psi*lambda_psi2)/((lambda_xi+lambda_psi)*(lambda_xi+lambda_psi2)) * np.exp(-lambda_xi * (x-(Q2+Q3)))
        else:
            print("huh") 
        
        return convo2
    

    # Compute thresholds from Proposition 4.5
    F_Q1_Q2_S = F_xi_psi(Q1 + Q2 + S)
    F_Q1_Q2 = F_xi_psi(Q1 + Q2)
    p = (2 * h + delta + f + r) / (h + delta + r + rho_u + theta)
    p_u_lower = -2
    #p_u_lower = (2 * h + delta + f + r) / F_Q1_Q2_S - (h + r + theta) if F_Q1_Q2_S > 0 else np.inf
    #p_u_upper = 2
    p_u_upper = (2 * h + delta + f + r) / F_Q1_Q2 - (h + r + theta) if F_Q1_Q2 > 0 else np.inf
    
    # Determine case based on rho_u
    if rho_u <= p_u_lower:
        M_star = 0
        L1_star = 0
        L2_star = 0
        L3_star = S
    elif rho_u >= p_u_upper:
        M_star = S
        L1_star = 0
        L2_star = 0
        L3_star = 0
    else:
        # Case (iii): Mixed allocation
        
        x = (2*h+2*delta+f+r) / (h+2*delta+r+rho_u+theta)
        
        #print(x)
        #print(F_xi_psi_psi2(Q2 + Q3))
        if x >= F_xi_psi_psi2(Q2 + Q3):
            A = 1/lambda_xi*np.log((lambda_psi*lambda_psi2)/((lambda_xi+lambda_psi)*(lambda_xi+lambda_psi2)*(1-x))) + Q2 + Q3
        else:
            print("Weird A value")
            A = Q2+Q3
            
        M_star = S - A + Q1 + Q2 + Q3
        M_star = max(0, min(S, M_star))  # Constrain M_star first

        # Solve for L1_star
        def g(L1):
            term1 = -(h + delta + r + rho_u + theta) * F_xi_psi(Q1 + Q2 + L1)
            term2 = (h + r + rho_u + theta) * F_xi(Q1 + L1)
            term3 = delta
            return term1 + term2 + term3

        try:
            sol = root_scalar(g, bracket=[0, S - M_star], method='brentq')
            L1_star = max(0, min(S - M_star, sol.root))  # Constrain root
            print(f"mean_xi={mean_xi}, L1_star={L1_star}")
        except ValueError:
            if g(0) > 0:
                L1_star = 0
            elif g(S) < 0:
                L1_star = S - M_star
            else:
                L1_star = 500
            #print(f"mean_xi={mean_xi}, No root, L1_star={L1_star}")
        
        def g2(L2):
            term1 = (h + delta + r + rho_u + theta) * F_xi_psi(Q1 + Q2 + L1_star + L2)
            term2 = -(h + 2*delta + r + rho_u + theta) * F_xi_psi_psi2(Q1 + Q2 + Q3 + L1_star + L2)
            term3 = delta
            return term1 + term2 + term3

        try:
            sol = root_scalar(g2, bracket=[0, S - M_star - L1_star], method='brentq')
            L2_star = max(0, min(S - M_star - L1_star, sol.root))  # Constrain root
            print(f"mean_xi={mean_xi}, L2_star={L2_star}")
        except ValueError:
            if g2(0) > 0:
                L2_star = 0
            elif g2(S) < 0:
                L2_star = S - M_star - L1_star
            else:
                L2_star = 500
            #print(f"mean_xi={mean_xi}, No root, L1_star={L1_star}")
        
        L3_star = max(0, S - M_star - L1_star - L2_star)

    return M_star, L1_star, L2_star, L3_star

# Range of mean_xi to explore
mean_xi_range = np.linspace(2000, 8000, 1000)

# Store results
M_values = []
L1_values = []
L2_values = []
L3_values = []

# Compute optimal allocations for each mean_xi
for mean_xi in mean_xi_range:
    M, L1, L2, L3 = compute_optimal(mean_xi)
    M_values.append(M)
    L1_values.append(L1)
    L2_values.append(L2)
    L3_values.append(L3)
    print(mean_xi)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(mean_xi_range, M_values, label='M (Market Orders)', linewidth=2)
plt.plot(mean_xi_range, L1_values, label='L1 (Limit Orders Level 1)', linewidth=2)
plt.plot(mean_xi_range, L2_values, label='L2 (Limit Orders Level 2)', linewidth=2)
plt.plot(mean_xi_range, L3_values, label='L3 (Limit Orders Level 3)', linewidth=2)
plt.xlabel('Mean of ξ')
plt.ylabel('Optimal Allocation Size (0 to 1000)')
plt.title('Optimal Allocations vs Mean of ξ')
plt.legend()
plt.grid(True)
plt.ylim(0, 1000)  # Set y-axis range as requested
plt.show()

sys.exit()

