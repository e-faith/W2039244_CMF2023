#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import yfinance as yf
import os


# ### Historical Volatility

# The histotical price of AMZN, Inc.

# In[2]:


AMZN = yf.download("AMZN", start="2022-12-19", end="2023-12-19")


# In[3]:


S = AMZN['Adj Close'][-1]
print('The spot price is $', round(S,2), '.')


# In[4]:


log_return = np.log(AMZN['Adj Close'] / AMZN['Adj Close'].shift(1))
vol_h = np.sqrt(252) * log_return.std()
print('The annualised volatility is', round(vol_h*100,2), '%')


# The Newton-Raphson Method to estimate impolied volatility

# In[5]:


def newton_vol_call(S, K, T, C, r):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #C: Call value
    #r: risk free rate
    #sigma: volatility of underlying asset
   
    MAX_ITERATIONS = 1000
    tolerance = 0.000001
    
    sigma = 0.25
    
    for i in range(0, MAX_ITERATIONS):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)

        diff = C - price

        if (abs(diff) < tolerance):
            return sigma
        else: 
            sigma = sigma + diff/vega
        
        # print(i,sigma,diff)
        
    return sigma


# Download the AMZN option data.

# In[6]:


AMZN = yf.Ticker("AMZN")
opt = AMZN.option_chain('2024-01-19')
opt.calls


# In[7]:


impvol = newton_vol_call(S, 165, 4/52, float(opt.calls.lastPrice[opt.calls.strike == 165.00]), 0.013675)
print('The implied volatility is', round(impvol*100,2) , '% for the one-month call with strike $ 165.00' ) 


# In[ ]:




