#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os


# # Question 3c

# S_0 = 100, K = 100, r = 0.05, sigma = 0.2, T = 1.0 and 4 steps for call

# In[2]:


S0 = 100                # spot stock price
K = 100                 # strike
T = 1.0                 # maturity 
r = 0.05                # risk free rate 
sigma = 0.2             # diffusion coefficient or volatility
N = 4                   # number of periods or number of time steps  
payoff = "call"         # payoff 


# In[3]:


dT = float(T) / N                             # Delta t
u = np.exp(sigma * np.sqrt(dT))               # up factor
d = 1.0 / u                                   # down factor 


# In[4]:


S = np.zeros((N + 1, N + 1))
S[0, 0] = S0
z = 1
for t in range(1, N + 1):
    for i in range(z):
        S[i, t] = S[i, t-1] * u
        S[i+1, t] = S[i, t-1] * d
    z += 1


# In[5]:


S


# In[6]:


a = np.exp(r * dT)    # risk free compound return
p = (a - d)/ (u - d)  # risk neutral up probability
q = 1.0 - p           # risk neutral down probability
p


# Finding option value at each final node

# In[7]:


S_T = S[:,-1]
V = np.zeros((N + 1, N + 1))
if payoff =="call":
    V[:,-1] = np.maximum(S_T-K, 0.0)
V


# Finding option value at earlier nodes

# In[8]:


# for European Option
for j in range(N-1, -1, -1): # Column. looping backwards. From N-1 to 0
    for i in range(j+1):  # Row. looping forwards. From 0 to j
        V[i,j] = np.exp(-r*dT) * (p * V[i,j + 1] + q * V[i + 1,j + 1]) #the theoretical value at each node.
V


# In[9]:


print('European ' + payoff, str( V[0,0]))


# In[ ]:




