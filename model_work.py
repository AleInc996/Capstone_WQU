# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:04:35 2025

@author: aless
"""

# importing necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime

# downloading monthly prices of the SPY ETF, as VIX data will be monthly and therefore we keep returns as monthly
spy_prices = yf.download('SPY', start = '2000-01-01', end = '2024-12-31', interval = '1mo')
spy_prices = spy_prices['Adj Close']

spy_rets = spy_prices.pct_change().dropna()

vix_data = pd.read_excel('VIX_term_structure_20250117.xlsx', header = 0)
vix_data = vix_data.drop(vix_data.index[0]) # removing first unnecessary row

# creating functions for the three indicators which will compose the innovative part of our approach

def rolling_std(series, time_interval): # defining a function for volatility, which we consider as rolling standard deviation
    return series.rolling(window = time_interval).std()

def rolling_correlation(series1, series2, time_interval): # defining a function for the rolling correlation
    return series1.rolling(window = time_interval).corr(series2)

def ROC(series): # defining a function for the Rate Of Change
    return series.pct_change()

def constant_maturity_term_structure(data, target_maturity):
    """
    This function computes the linear interpolation of VIX futures prices 
    for generating a constant maturity term structure.
    
    It takes the VIX dataframe with prices and days to expiration,
    together with a target maturity, expressed in days, as inputs,
    and will return the interpolated prices of the VIX futures.
    """
    constant_prices = [] # pre-allocating memory for the prices calculated via constant maturity term structure
    
    for target in target_maturity: # we will be looping over all maturities included in the table
        before = data[data["Days to expiration"] <= target] # looking for the nearest earlier contract for interpolation
        after = data[data["Days to expiration"] > target] # looking for the nearest later contract for interpolation

        # for different maturities, there will be cases where we won't have the directly close value, therefore
        if before.empty:  # if the price on the earlier place is empty
            price = after.iloc[0]["Last Price"]  # we will take the earliest available price
        elif after.empty:  # if the price on the later place is empty
            price = before.iloc[-1]["Last Price"]  # use the latest available
        else:
            # otherwise
            before = before.iloc[-1] # the earlier price is normally identified
            after = after.iloc[0] # the later price is normally identified
            price = before["Last Price"] + (
                (after["Last Price"] - before["Last Price"]) /
                (after["Days to expiration"] - before["Days to expiration"]) *
                (target - before["Days to expiration"])
            ) # here we calculate linear interpolation through a commonly accepted and recognized formula
        
        constant_prices.append(price) # for each iteration of the loop, we append the newly calculated price to the originally created variable
    
    return constant_prices # the function will return the newly calculated prices

target_maturity = [30, 60, 90, 120, 150, 180, 210, 240, 270]  # here we define a variable for the target maturities and it will be all of them

constant_maturity = constant_maturity_term_structure(vix_data, target_maturity) 

# Add the constant maturity prices to a DataFrame
constant_maturity_df = pd.DataFrame({
    "Constant Maturity (Days)": target_maturity,
    "Price": constant_maturity
})

print("Constant Maturity Term Structure:") # printing the new term structure, made of prices at constant maturity

vix_slope = constant_maturity_df["Price"].diff() / constant_maturity_df["Constant Maturity (Days)"].diff() # computing the slope of the term structure
vix_slope = vix_slope.dropna()

vol_indicator = rolling_std(spy_rets, time_interval = 10).dropna() # calculating volatility indicator on the returns of SPY
correlation_indicator = rolling_correlation(spy_rets, vix_slope, time_interval = 30)#.dropna()
roc_indicator = ROC(vix_slope).dropna() # calculating rate of change of the VIX futures constant maturity term structure slope

print(vol_indicator)

print(correlation_indicator)

print(roc_indicator)