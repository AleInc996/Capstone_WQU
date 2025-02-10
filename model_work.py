# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:04:35 2025

@author: AleInc996
"""

#import pdb; pdb.set_trace() # eventually for debugging

# importing necessary libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime
from sklearn.metrics import mean_squared_error

### 1. Data preparation and manipulation

# downloading monthly prices of the SPY ETF, as VIX data will be monthly and therefore we keep returns as monthly
spy_prices = yf.download('SPY', start = '2005-07-01', end = '2024-12-31', interval = '1mo') # starting since when we have availability for the VIX futures historical term structure
spy_prices = spy_prices['Adj Close'] # taking only adjusted close prices

spy_rets = spy_prices.pct_change().dropna() # computing returns and dropping NAs (most importantly, dropping the first observation)
spy_rets.rename('SPY returns', inplace = True) # renaming the column as now we have returns and not prices

vix_data_ahead = pd.read_excel('VIX_term_structure_20250117.xlsx', header = 0) # uploading VIX futures ahead term structure data
vix_data_ahead = vix_data_ahead.drop(vix_data_ahead.index[0]) # removing first unnecessary row

for i in range(2, len(vix_data_ahead) + 1): # turning the Period column into datetime, needed for further analyses
    vix_data_ahead['Period'][i] = datetime.strptime(vix_data_ahead['Period'][i], '%m/%Y')

vix_data_hist = pd.read_excel('hist_vix_term_structure.xlsx', header = 0) # uploading VIX futures ahead term structure data
vix_data_hist = vix_data_hist.drop(vix_data_hist.index[0]) # removing first unnecessary row

for i in range(1, len(vix_data_hist) + 1): # turning the Period column into datetime, needed for further analyses
    vix_data_hist['Period'][i] = datetime.strptime(vix_data_hist['Period'][i], '%m/%Y')
    
vix_data_hist = vix_data_hist.sort_values(by = 'Period') # in the historical data, futures prices are not ordered properly in ascending or descending order

## creating functions for the three indicators which will compose the innovative part of our approach

def rolling_std(series, time_interval): # defining a function for volatility, which we consider as rolling standard deviation
    return series.rolling(window = time_interval).std()

def rolling_correlation(series1, series2, time_interval): # defining a function for the rolling correlation
    return series1.rolling(window = time_interval).corr(series2)

def ROC(series): # defining a function for the rate of change, that will be primarly used for the VIX slope
    return series.pct_change()

def constant_mat_term_structure(vix_term_structure):
    """
    This function computes the linear interpolation of VIX futures prices 
    for generating a constant maturity term structure.
    
    It takes the VIX dataframe, be it either the historical one or the future one, as input,
    and will return the interpolated prices of the VIX futures.
    """
    
    constant_maturity_prices = [] # allocating memory for the prices computed with constant maturity approach
    for i in range(1, len(vix_term_structure)): # looping over all observations of the dataframe fed to the function
        maturity1 = vix_term_structure.loc[i - 1, "Days to expiration"] # first maturity
        maturity2 = vix_term_structure.loc[i, "Days to expiration"] # second maturity
        price1 = vix_term_structure.loc[i - 1, "Last Price"] # first price
        price2 = vix_term_structure.loc[i, "Last Price"] # second price
        
        target_maturity = (maturity1 + maturity2) / 2 # the target maturity is identified as the middle point between maturity 1 and 2
        
        # formula decided to be used for price interpolation
        interpolated_price = price1 * (maturity2 - target_maturity) / (maturity2 - maturity1) + price2 * (target_maturity - maturity1) / (maturity2 - maturity1)
        constant_maturity_prices.append(interpolated_price) # appending each result of the loop in the initially created variable
    
    constant_maturity_prices.insert(0, None) # adding NaN for the first row since it doesn't have a previous contract
    vix_term_structure["Constant Maturity Price"] = constant_maturity_prices # adding the newly computed prices to the original table
    return vix_term_structure

vix_data_ahead = vix_data_ahead.reset_index(drop = True) # resetting the index of the ahead term structure for data manipulation
constant_maturity_ahead = constant_mat_term_structure(vix_data_ahead) # computing constant maturity ahead term structure

vix_data_hist = vix_data_hist.reset_index(drop = True) # resetting the index of the ahead term structure for data manipulation
vix_data_hist.rename(columns = {'Days past': 'Days to expiration'}, inplace = True) # column renaming for data manipulation (the function admits 'Days to expiration', we could have changed the function but it would require more than one line as here)
constant_maturity_hist = constant_mat_term_structure(vix_data_hist) # computing constant maturity historical term structure
vix_data_hist.rename(columns = {'Days to expiration': 'Days past'}, inplace = True) # column renaming for data manipulation (the function admits 'Days to expiration', we could have changed the function but it would require more than one line as here)

print("Constant Maturity Term Structure ahead:") # printing the new ahead term structure, made of prices at constant maturity
constant_maturity_ahead[['Period', 'Constant Maturity Price']]

print("Constant Maturity Term Structure historical:") # printing the new historical term structure, made of prices at constant maturity
constant_maturity_hist[['Period', 'Constant Maturity Price']]

# computing the slope of the ahead term structure
constant_maturity_ahead['vix_slope'] = constant_maturity_ahead["Constant Maturity Price"].diff() / constant_maturity_ahead["Days to expiration"].diff()
#constant_maturity_ahead['vix_slope']= constant_maturity_ahead['vix_slope'].dropna()

# computing the slope of the historical term structure
constant_maturity_hist["vix_slope"] = constant_maturity_hist["Constant Maturity Price"].diff() / constant_maturity_hist["Days past"].diff() # maybe a negative sign in front of it?
#constant_maturity_hist["vix_slope"] = constant_maturity_hist["vix_slope"].dropna()
#constant_maturity_hist.rename(columns = {'Period': 'Date'}, inplace = True)

# now the idea is to merge the historical VIX dataframe with returns from SPY, so that we have aligned data and we can compute correlation
constant_maturity_hist = constant_maturity_hist.set_index('Period', drop = True) # setting the dates as index
correlation_dataset = constant_maturity_hist.join(spy_rets, how = 'left') # adding the returns from SPY to the historical VIX dataframe
correlation_dataset = correlation_dataset.drop(['Tenor', 'Ticker', 'Last Price', 'Days past', 'Constant Maturity Price'], axis = 1) # dropping unnecessary columns for correlation analysis

vol_indicator = rolling_std(spy_rets, time_interval = 5).dropna() # calculating volatility indicator on the returns of SPY
correlation_indicator = rolling_correlation(correlation_dataset['vix_slope'], correlation_dataset['SPY returns'], time_interval = 5).dropna() # computing correlation indicator between SPY returns and historical VIX slope
roc_indicator = ROC(correlation_dataset['vix_slope']).dropna() # calculating rate of change of the historical VIX futures constant maturity term structure slope
roc_indicator.replace([np.inf, -np.inf], np.nan, inplace = True) # replacing infinite values with nan, as there are a couple of zeros in the slope columns which generate inf
roc_indicator.dropna(inplace = True) # dropping again rows with nan values

print(vol_indicator)

print(correlation_indicator)

print(roc_indicator)

# %%

### 2. Training Momentum Transformer model

correlation_dataset = correlation_dataset.dropna() # dropping the first NAs due to rolling window
spy_rets_training = correlation_dataset['SPY returns'] # identifying the historical SPY returns for training
vix_slope_training = correlation_dataset['vix_slope'] # identifying the historical VIX slope for training

vol_indicator_training = rolling_std(spy_rets_training, time_interval = 5).dropna() # computing volatility indicator for the training part
correlation_indicator_training = rolling_correlation(spy_rets_training, vix_slope_training, time_interval = 5).dropna() # computing correlation indicator for the training part
roc_indicator_training = ROC(vix_slope_training).dropna() # computing roc indicator for the training part
roc_indicator_training.replace([np.inf, -np.inf], np.nan, inplace = True) # replacing infinite values with nan, as there are a couple of zeros in the slope columns which generate inf
roc_indicator_training.dropna(inplace = True) # dropping again rows with nan values

transformer_train_data = pd.DataFrame({ # combining historical vix slope and SPY returns data for training
    'Historical VIX futures slope': vix_slope_training,
    'Stock Returns': spy_rets_training
}).dropna()

scaler_transformer = StandardScaler() # activating the scaler for standardizing the two variables
scaled_features_transformer = scaler_transformer.fit_transform(transformer_train_data) # standardizing

X_transformer, y_transformer = [], [] # pre-allocating memory for appending standardized values
sequence_length = 5 # instead of considering single data points, deciding for the length of a sequence of consecutive observations to fed the model with, in order to try to capture temporal dependencies

for i in range(sequence_length, len(scaled_features_transformer)): # appending
    X_transformer.append(scaled_features_transformer[i-sequence_length:i]) # TAKE VIX_SLOPE AND NOT SCALED_FEATURES???
    y_transformer.append(spy_rets_training.iloc[i])

X_transformer = np.array(X_transformer) # turning the list into a numpy array
y_transformer = np.array(y_transformer) # turning the list into a numpy array

X_train_transformer, X_test_transformer, y_train_transformer, y_test_transformer = train_test_split(
    X_transformer, y_transformer, test_size = 0.2, random_state = 42
) # training-testing dataframes split, for now going with 80/20

transformer_model = Sequential([ # grouping layers into a model with Sequential, so that we have one output for each input in a layer
    LSTM(64, input_shape = (X_transformer.shape[1], X_transformer.shape[2]), return_sequences = True), # first Long-Short Term Memory approach
    Dropout(0.2), # first dropout rate
    LSTM(32, return_sequences = False), # second Long-Short Term Memory approach
    Dropout(0.2), # second dropout rate
    Dense(1, activation = 'sigmoid') # the output is a single value, try to see what happens if you change sigmoid with classification
])

transformer_model.compile(optimizer = 'adam', loss = 'mean_squared_error') # compiling with adam optimizer and mean-squared error loss
res = transformer_model.fit(X_train_transformer, y_train_transformer, # fitting the model on the testing part of the dataframes
                            epochs = 20, batch_size = 16, 
                            validation_data = (X_test_transformer, y_test_transformer)) # validation part (IS THIS REALLY NEEDED HERE?)

forecasts = transformer_model.predict(X_test_transformer) # we can eventually check the forecasts on the testing part of the x variables
rmse = np.sqrt(mean_squared_error(y_test_transformer, forecasts)) # calculating root mean squared error as error measure between testing part of returns and forecasts
print(rmse)

# %%

### 3. Using trained model for forecasting 2025 scenarios

# for 2025 SPY returns estimates, instead of running another model on our own, we decided to make use of the estimates from:
# https://usdforecast.com/ , the website of the Economy Forecast Agency (EFA)
# https://longforecast.com/spy-stock

spy_prices_pred_2025 = pd.read_excel('spy_pred_2025.xlsx') # uploading predicted prices of SPY for 2025
spy_prices_pred_2025 = spy_prices_pred_2025[['Month', 'Close']] # taking only the month and the close prices

forward_vix_slope = constant_maturity_ahead[['Period', 'vix_slope']] # storing the ahead vix slope in a new variable
forward_vix_slope = constant_maturity_ahead['vix_slope'] # storing the ahead vix slope in a new variable





# %%

### 4. Backtesting

