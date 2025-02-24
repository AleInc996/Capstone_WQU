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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, LayerNormalization, GlobalAveragePooling1D, Embedding

### 1. Data preparation and manipulation

# downloading monthly prices of the SPY ETF, as VIX data will be monthly and therefore we keep returns as monthly
spy_prices = yf.download('SPY', start = '2005-07-01', end = '2024-12-31', interval = '1mo', multi_level_index = False, auto_adjust = False) # starting since when we have availability for the VIX futures historical term structure
#if isinstance(spy_prices.columns, pd.MultiIndex):
    #spy_prices = spy_prices.xs(key="SPY", axis=1, level=1)
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

### 2. Training Momentum Transformer model development

correlation_dataset = correlation_dataset.dropna() # dropping the first NAs due to rolling window
spy_rets_training = correlation_dataset['SPY returns'] # identifying the historical SPY returns for training
vix_slope_training = correlation_dataset['vix_slope'] # identifying the historical VIX slope for training

vol_indicator_training = rolling_std(spy_rets_training, time_interval = 5).dropna() # computing volatility indicator for the training part
correlation_indicator_training = rolling_correlation(spy_rets_training, vix_slope_training, time_interval = 5).dropna() # computing correlation indicator for the training part
roc_indicator_training = ROC(vix_slope_training).dropna() # computing roc indicator for the training part
roc_indicator_training.replace([np.inf, -np.inf], np.nan, inplace = True) # replacing infinite values with nan, as there are a couple of zeros in the slope columns which generate inf
roc_indicator_training.dropna(inplace = True) # dropping again rows with nan values


## first, we are going to check results with a double LSTM, as normally momentum transformer replaces LSTM because of vanishing gradients
doubleLSTM_train_data = pd.DataFrame({ # combining historical vix slope and SPY returns data for training
    'Historical VIX futures slope': vix_slope_training,
    'Stock Returns': spy_rets_training
}).dropna()

scaler_doubleLSTM = StandardScaler() # activating the scaler for standardizing the two variables
scaled_features_doubleLSTM = scaler_doubleLSTM.fit_transform(doubleLSTM_train_data) # standardizing

X_doubleLSTM, y_doubleLSTM = [], [] # pre-allocating memory for appending standardized values
sequence_length = 5 # instead of considering single data points, deciding for the length of a sequence of consecutive observations to fed the model with, in order to try to capture temporal dependencies

for i in range(sequence_length, len(scaled_features_doubleLSTM)): # appending
    X_doubleLSTM.append(scaled_features_doubleLSTM[i-sequence_length:i]) # TAKE VIX_SLOPE AND NOT SCALED_FEATURES???
    y_doubleLSTM.append(spy_rets_training.iloc[i])

X_doubleLSTM = np.array(X_doubleLSTM) # turning the list into a numpy array
y_doubleLSTM = np.array(y_doubleLSTM) # turning the list into a numpy array

X_train_doubleLSTM, X_test_doubleLSTM, y_train_doubleLSTM, y_test_doubleLSTM = train_test_split(
    X_doubleLSTM, y_doubleLSTM, test_size = 0.2, random_state = 42
) # training-testing dataframes split, for now going with 80/20

doubleLSTM_model = Sequential([ # grouping layers into a model with Sequential, so that we have one output for each input in a layer
    LSTM(64, input_shape = (X_doubleLSTM.shape[1], X_doubleLSTM.shape[2]), return_sequences = True), # first Long-Short Term Memory approach
    Dropout(0.2), # first dropout rate
    LSTM(32, return_sequences = False), # second Long-Short Term Memory approach
    Dropout(0.2), # second dropout rate
    Dense(1, activation = 'sigmoid') # the output is a single value, try to see what happens if you change sigmoid with classification
])

doubleLSTM_model.compile(optimizer = 'adam', loss = 'mean_squared_error') # compiling with adam optimizer and mean-squared error loss
res = doubleLSTM_model.fit(X_train_doubleLSTM, y_train_doubleLSTM, # fitting the model on the testing part of the dataframes
                            epochs = 20, batch_size = 16, 
                            validation_data = (X_test_doubleLSTM, y_test_doubleLSTM)) # validation part (IS THIS REALLY NEEDED HERE?)

forecasts_doubleLSTM = doubleLSTM_model.predict(X_test_doubleLSTM) # we can eventually check the forecasts on the testing part of the x variables
rmse_doubleLSTM = np.sqrt(mean_squared_error(y_test_doubleLSTM, forecasts_doubleLSTM)) # calculating root mean squared error as error measure between testing part of returns and forecasts
print("Double LSTM RMSE:", rmse_doubleLSTM)


## building the proper Momentum Transformer model
spy_rets_training = correlation_dataset['SPY returns'] # identifying the historical SPY returns for training
vix_slope_training = correlation_dataset['vix_slope'] # identifying the historical VIX slope for training

class transformer_core(Layer): # main class for the transformer model
    def __init__(self, embed_dim, num_heads, ff_dim, rate = 0.1):
        super(transformer_core, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim) # applying self-attention mechanism over the input sequence
        self.ffn = tf.keras.Sequential([ # after self-attention mechanism, two dense layers are used to process each token in an isolated way
            Dense(ff_dim, activation = 'relu'), # ReLu activation function  
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon = 1e-6) # the input is normalized so that the training is faster and more stable
        self.layernorm2 = LayerNormalization(epsilon = 1e-6) # done for both layers
        self.dropout1 = Dropout(rate) # first dropout rate
        self.dropout2 = Dropout(rate) # second dropout rate

    def call(self, inputs, training): # here we define a class for designing how the input is fed to the layers
        attn_output = self.att(inputs, inputs) # self-attention
        attn_output = self.dropout1(attn_output, training = training) # the dropout makes the neurons alive in a random fashion
        out1 = self.layernorm1(inputs + attn_output) # as well as inputs, here we normalize the output
        ffn_output = self.ffn(out1) # transforming the normalized output
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)
    
class positionalencoding(Layer): # defining another class for the positional encoding step of the transformer
    def __init__(self, sequence_length, embed_dim):
        super(positionalencoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, embed_dim) # the positional encoding will learn temporal dependencies with the length of sequence for input data that we decided

    def get_angles(self, pos, i, d_model): # defining a function to calculate sin and cosin of the angle
        angle_degree = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_degree

    def positional_encoding(self, position, d_model): # here we define a function for proper positional encoding, which takes as inputs the maximum length of a sequence (position) and the dimension of each embedding (d_model)
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], # creating an array of integers showing the positions in the sequence
                                     np.arange(d_model)[np.newaxis, :], # creating an array for the dimensions of the embeddings
                                     d_model) 
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # applying the sine function to all columns with an even number
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) # applying the cosine function to all columns with an odd number
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs): # defining a function to add everything built in the positional encoding to the initial input data
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
def momentum_transformer(sequence_length, feature_dim): # defining a function for a transformer based neural network, that takes as inputs the window size and the features
    embed_dim = 32  # setting the size of embedding dimensions for each token
    num_heads = 4  # setting the number of heads in the multi-head attention mechanism
    ff_dim = 128  # setting the number of hidden neurons for the feed forward network
    inputs = Input(shape = (sequence_length, feature_dim)) # storing memory for input data, where the shape is given by the length of each sequence and the feature dimension
    x = Dense(embed_dim)(inputs) # fully connected (dense) layer is applied to transform the input features into a higher-dimensional space of size given by the embedding dimension
    x = positionalencoding(sequence_length, embed_dim)(x) # applying positional encoding to the inputs
    x = transformer_core(embed_dim, num_heads, ff_dim)(x, training = True) # launching the core transformer class
    x = GlobalAveragePooling1D()(x) # computing the mean across for all points in time for future dimension, after this the sequence becomes a single vector
    x = Dropout(0.1)(x) # applying dropout, here with a probability of 10% to prevent overfitting (it means that 10% of the elements in x are being put equal to zero)
    x = Dense(20, activation = 'relu')(x) # applying a dense layer with 20 neurons and ReLu activation function
    x = Dropout(0.1)(x) # a second dropout rate is applied, again 10%
    outputs = Dense(1, activation = 'sigmoid')(x) # the last dense layer for output is applied with 1 neuron and sigmoid activation function
    model = Model(inputs = inputs, outputs = outputs) # creating a model object, using the defined inputs and outputs
    model.compile(optimizer = 'adam', loss = 'mean_squared_error') # compiling the model, using adam optimizer and mean squared error as a loss measure, also to compare with double LSTM approach
    return model

sequence_length = 5 # the length of the sequence is set the same as for double LSTM approach
scaler_transformer = StandardScaler() # activating standard scaler
transformed_data = scaler_transformer.fit_transform(pd.DataFrame({ # grouping the vix slope and spy rets data together, as before
    'Historical VIX futures slope': vix_slope_training,
    'Stock Returns': spy_rets_training
}).dropna()) # dropping NAs

X_transformer, y_transformer = [], [] # as before, pre-allocating memory for x and y dataframes
for i in range(sequence_length, len(transformed_data)): # as before, appending data using the length of sequence
    X_transformer.append(transformed_data[i-sequence_length:i])
    y_transformer.append(spy_rets_training.iloc[i])

X_transformer = np.array(X_transformer) # turning into a numpy array
y_transformer = np.array(y_transformer) # turning into a numpy array

X_train_transformer, X_test_transformer, y_train_transformer, y_test_transformer = train_test_split(
    X_transformer, y_transformer, test_size = 0.2, random_state = 42) # same type of split

transformer_model = momentum_transformer(sequence_length, X_transformer.shape[2]) # running the momentum transformer model
transformer_model.fit(X_train_transformer, y_train_transformer, epochs = 20, batch_size = 16, validation_data = (X_test_transformer, y_test_transformer)) # fitting the momentum transformer on training data and validating

forecasts_transformer = transformer_model.predict(X_test_transformer) # predicting forecasts using testing data of the vix slope
rmse_transformer = np.sqrt(mean_squared_error(y_test_transformer, forecasts_transformer)) # computing mean-squared error
print("Transformer RMSE:", rmse_transformer)

# %%

### 3. Using trained model for forecasting 2025 scenarios

# for 2025 SPY returns estimates, instead of running another model on our own, we decided to make use of the estimates from:
# https://usdforecast.com/ , the website of the Economy Forecast Agency (EFA)
# https://longforecast.com/spy-stock

spy_prices_pred_2025 = pd.read_excel('spy_pred_2025.xlsx') # uploading predicted prices of SPY for 2025
spy_prices_pred_2025 = spy_prices_pred_2025[['Month', 'Close']] # taking only the month and the close prices
spy_prices_pred_2025 = spy_prices_pred_2025[:-3]

forward_vix_slope = constant_maturity_ahead[['Period', 'vix_slope']] # storing the ahead vix slope in a new variable
forward_vix_slope = forward_vix_slope.dropna().reset_index(drop = True)

# Generate initial alpha (momentum probabilities) using the Transformer model
sequence_length = 5 # the length of the sequence is set the same as for double LSTM approach

transformer_data = pd.DataFrame({ # grouping the vix slope and spy rets data together, as before
    '2025 VIX futures slope': forward_vix_slope['vix_slope'],
    'Stock Returns': spy_prices_pred_2025['Close']
}).dropna() # dropping NAs

X_pred_transformer, y_pred_transformer = [], [] # as before, pre-allocating memory for x and y dataframes
for i in range(sequence_length, len(transformer_data)): # as before, appending data using the length of sequence
    X_pred_transformer.append(transformer_data[i-sequence_length:i])
    y_pred_transformer.append(spy_prices_pred_2025.iloc[i])

X_pred_transformer = np.array(X_pred_transformer) # turning into a numpy array
y_pred_transformer = np.array(y_pred_transformer) # turning into a numpy array
initial_alpha = transformer_model.predict(X_pred_transformer).flatten()

# Define momentum and mean-reversion trading signals based on alpha
def momentum_signal(vix_slope):
    return np.sign(vix_slope)

def mean_reversion_signal(stock_returns):
    return -np.sign(stock_returns)

# Compute initial trading signals (before adjustment) - the problem is that we have initial alpha with 3 and the slope with 8 obs
momentum_component = initial_alpha * momentum_signal(forward_vix_slope['vix_slope']) 
mean_reversion_component = (1 - initial_alpha) * mean_reversion_signal(spy_rets.iloc[-len(forward_vix_slope):])
initial_trading_signal = momentum_component + mean_reversion_component

# Backtest the initial strategy
def backtest_trading_signals(signals, stock_prices):
    daily_returns = stock_prices.pct_change().dropna()
    strategy_returns = signals.shift(1).fillna(0) * daily_returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    return cumulative_returns

# Load 2025 stock prices for backtesting
stock_prices_2025 = pd.read_csv('SPY_2025_prices.csv', index_col='Date', parse_dates=True)['Adj Close']
initial_results = backtest_trading_signals(pd.Series(initial_trading_signal, index=stock_prices_2025.index), stock_prices_2025)

# Plot initial cumulative returns (using matplotlib or other visualization library)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(initial_results, label='Initial Strategy Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Initial Momentum Transformer Strategy Backtest')
plt.legend()
plt.show()




# %%

### 4. Backtesting

