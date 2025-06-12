#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 15:00:53 2025

@author: jawclaes
"""



import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.optimize import brentq



# Set the parameters of our system (all values in $ and seconds)
R = 10
T = 60
kappa = 100
Delta = 0.01
lam_start = 50/60 
rho_u = 0.005 + 0.001
theta = 0.0005 # Was 0.00001
r = 0.003 # Was 0.003
f = 0.002 # Was 0.002
lam = lam_start * np.exp(kappa * (Delta + r + f) - 1)


# Example usage
p0 = 60        # Initial price
mu = 0     # Drift (example value)
sig_p = 0.01      # Total time (e.g., 1 unit)
N = 1000       # Number of steps
n_sim = 3

inventory = np.array([1,2,3,4,5,6,7,8,9,10])

mu0 = np.array([60.     , 59.9072 , 59.1117 , 56.9188 , 47.10942,  0.     ,
        0.     ,  0.     ,  0.     ,  0.     ])
mu000010 = np.array([60.        , 59.91479966, 59.15718   , 57.0942    , 49.2331    ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
mu000020 = np.array([60.        , 59.92214709, 59.2017    , 57.2559    , 50.664     ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
mu000040 = np.array([60.       , 59.9361717,  59.28428720610962      ,  57.54535       ,  52.53963835639213,
        0.       ,  0.       ,  0.       ,  0.       ,  0.       ])
mu000060 = np.array([60.        , 59.94936227, 59.36184605, 57.79835162, 53.76198108,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ])
mu000080 = np.array([60.        , 59.96178389, 59.43459347, 58.02280288, 54.64856212,
       13.04451229,  0.        ,  0.        ,  0.        ,  0.        ])
#mu=0.001 kan niet want die geeft nul in demoninator
mu000120 = np.array([60.        , 59.98454885, 59.56865868, 58.4069527 , 55.89056048,
       47.26223242,  0.        ,  0.        ,  0.        ,  0.        ])
mu000160 = np.array([60.        , 60.        , 59.69234839, 58.72927187, 56.7521342 ,
       51.86848625,  0.        ,  0.        ,  0.        ,  0.        ])
mu000180 = np.array([60.        , 60.        , 59.75213323, 58.87384398, 57.09903669,
       53.10351227,  6.9044097 ,  0.        ,  0.        ,  0.        ])
mu000240 = np.array([60.        , 60.        , 59.94752214, 59.2665539 , 57.9365873 ,
       55.39450925, 48.4146389 ,  0.        ,  0.        ,  0.        ])
mu000280 = np.array([60.        , 60.        , 60.        , 59.51666864, 58.38856507,
       56.36925298, 51.98784761,  4.39624304,  0.        ,  0.        ])
mu000380 = np.array([60.        , 60.        , 60.        , 60.        , 59.36996305,
       58.04974172, 55.85957929, 51.2435238 ,  3.10214082,  0.        ])
mu000480 = np.array([60.        , 60.        , 60.        , 60.        , 60.        ,
       59.31630447, 57.85907735, 55.54558745, 50.77970143,  2.37165779])
mu000580 = np.array([60.        , 60.        , 60.        , 60.        , 60.        ,
       60.        , 59.30467253, 57.77566861, 55.38009819, 50.51695358])

# Time points for plotting
time = np.linspace(0, T, N + 1)

def TWAP(t,R,T):
    return R* (T-t)/T

twap_schedule = np.zeros(N+1)
for i in range(N):
    twap_schedule[i] = TWAP(i*T/N,R,T)





plt.figure(figsize=(10, 6))
plt.plot(time,twap_schedule,'--', color='orange', label='TWAP')
plt.plot(mu0, inventory, '-o', label='μ=0')
plt.plot(mu000080, inventory, '-o', label='μ=0.0008')
plt.plot(mu000180, inventory, '-o', label='μ=0.0018')
plt.plot(mu000280, inventory, '-o', label='μ=0.0028')
plt.plot(mu000380, inventory, '-o', label='μ=0.0038')
plt.plot(mu000480, inventory, '-o', label='μ=0.0048')
plt.plot(mu000580, inventory, '-o', label='μ=0.0058')
plt.title("Optimal Stopping Times at Different Inventories for Various μ")
plt.xlabel('Time (t)')
plt.ylabel('Inventory')
plt.legend()
plt.xlim(-1,T+1)
plt.ylim(-0.2,R+0.2)
plt.grid(True)
plt.show()