# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 17:40:40 2024

@author: mbluu
"""
import numpy as np
import math 
import matplotlib.pyplot as plt


def HUD_evasion(evasion_list):
    product = 1
    for ii in range(len(evasion_list)):
        product = product*(1-evasion_list[ii])
        
    return 1 - product

def C_product(C,N):
    product = 1
    for ii in range(N-1):
        product = product*( (1/(ii+1)) - C)
    
    return product
# =============================================================================
# evasion_list = [0.35]*6
# evasion_hud = HUD_evasion(evasion_list)
# print(evasion_hud)
# 
# 
# =============================================================================

def pseudorand(target):
    C = 0.30210
    prob = 0.5
    
    while abs(target-prob) > 10**-8:
        if C < 0.1:
            C = C + (target-prob)/8
        else:
            C = C + (target-prob)/2
        mat_size = int(np.ceil(1/C))
        P = np.zeros((mat_size,mat_size))
        
        for n in range(P.shape[0]):
            print(n)
            if (n+1)*C < 1:
                P[n,0] = (n+1)*C 
            else:
                P[n,0] = 1
        
        for n in range(P.shape[0] - 1):
            P[n,n+1] = 1- P[n,0]
        
        [v,d] = np.linalg.eig(P.T)
        stationary = d[:,0]/sum(d[:,0])
        prob = np.real(np.dot(stationary,P[:,0]))
        
    return C, prob

#%
prob = HUD_evasion([0.17])

# =============================================================================
# 17 % --> 20 %
# 20 % --> 25 %
# 25 % --> 34 %
# =============================================================================

#% Pseudo Random distribution 

[C,prob] = pseudorand(1-prob)

N_tot = 50
P = np.zeros(N_tot)
N = [ii+1 for ii in range(N_tot)]
for n in range(N_tot):
    P[n] = math.factorial(n) * C * C_product(C,n)
    if P[n] < 0:
        break
plt.bar(N,P)
plt.ylim([0,max(P)+0.1])
print(sum(P))

#%%
P_linear =[]
N = []
for n in range(1000):
    P_linear.append(C*n)
    N.append(n)
    if P_linear[n] > 1:
        break
plt.figure(231)
plt.plot(N,P_linear)
print(N[-1])

#%% Dota bash simulator!!!
def dota_bash_simulator():
    import random
    [C,prob] = pseudorand(0.25)
    
    N = []
    vector_slots = np.zeros(12)
    
    for ii in range(100):
        for n in range(1,100):
            num = random.uniform(0, 1)
    
            prob = C*n
            if num < prob:
                N.append(n)
                break
    
        vector_slots[N[ii]-1] += 1

    return vector_slots, N
vector_slots, N = dota_bash_simulator()
N_pmd = [ii+1 for ii in range(12)]

plt.bar(N_pmd,vector_slots)

#%% Geometric distribution of uniform probability
p = 0.25

N_tot = 50
P = np.zeros(N_tot)
N = [ii for ii in range(N_tot)]
for n in range(N_tot):
    P[n] = (1-p)**(n) * p
    if P[n] < 0:
        break
plt.bar(N,P)
plt.ylim([0,max(P)+0.1])
print(sum(P))

