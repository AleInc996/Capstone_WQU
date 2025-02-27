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
import seaborn as sns
import matplotlib.pyplot as plt

### 1. Data preparation and manipulation

# downloading monthly prices of the SPY ETF, as VIX data will be monthly and therefore we keep returns as monthly
spy_prices_hist = yf.download('SPY', start = '2005-07-01', end = '2024-12-31', interval = '1mo', multi_level_index = False, auto_adjust = False) # starting since when we have availability for the VIX futures historical term structure
#if isinstance(spy_prices.columns, pd.MultiIndex):
    #spy_prices = spy_prices.xs(key="SPY", axis=1, level=1)
spy_prices_hist = spy_prices_hist['Adj Close'] # taking only adjusted close prices

spy_rets_hist = spy_prices_hist.pct_change().dropna() # computing returns and dropping NAs (most importantly, dropping the first observation)
spy_rets_hist.rename('SPY returns', inplace = True) # renaming the column as now we have returns and not prices

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
correlation_dataset = constant_maturity_hist.join(spy_rets_hist, how = 'left') # adding the returns from SPY to the historical VIX dataframe
correlation_dataset = correlation_dataset.drop(['Tenor', 'Ticker', 'Last Price', 'Days past', 'Constant Maturity Price'], axis = 1) # dropping unnecessary columns for correlation analysis

vol_indicator = rolling_std(spy_rets_hist, time_interval = 5).dropna() # calculating volatility indicator on the returns of SPY
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
    'Historical SPY Returns': spy_rets_training
}).dropna()

scaler_doubleLSTM = StandardScaler() # activating the scaler for standardizing the two variables
scaled_features_doubleLSTM = scaler_doubleLSTM.fit_transform(doubleLSTM_train_data) # standardizing

X_doubleLSTM, y_doubleLSTM = [], [] # pre-allocating memory for appending standardized values
sequence_length = 5 # instead of considering single data points, deciding for the length of a sequence of consecutive observations to fed the model with, in order to try to capture temporal dependencies

for i in range(sequence_length, len(scaled_features_doubleLSTM)): # appending
    X_doubleLSTM.append(scaled_features_doubleLSTM[i-sequence_length:i]) 
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
    'Historical SPY Returns': spy_rets_training
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
spy_prices_pred_2025 = spy_prices_pred_2025[:-3] # removing October, November and December 2025 to align with VIX futures data

forward_vix_slope = constant_maturity_ahead[['Period', 'vix_slope']] # storing the ahead vix slope in a new variable
forward_vix_slope = forward_vix_slope.dropna().reset_index(drop = True) # dropping NAs and resetting index as it does not start from 0

transformer_data = pd.DataFrame({ # grouping the vix slope and spy rets data together, as before
    '2025 VIX futures slope': forward_vix_slope['vix_slope'],
    '2025 SPY Prices': spy_prices_pred_2025['Close']
}).dropna() # dropping NAs

transformer_data = np.array(transformer_data) # turning the dataframe into a numpy array, as it is needed by the transformer model

# now the vix slope variable needs to be transformed into an array of shape similar to the one used for train
# therefore I am setting the final parameters I want:
num_obs = 8 # Number of observations (because we have 8 observations for the 2025 forward VIX futures slope)
timesteps = 5 # sequence used to train the model
features = 2 # variables 

transformer_expanded_data = np.zeros((num_obs, timesteps, features))# storing memory for the expanded data

for i in range(num_obs): # expanding the data repeating values but making sure that each value appears at least once
    for t in range(timesteps):
        transformer_expanded_data[i, t] = transformer_data[(i + t) % num_obs]

initial_mom_prob = transformer_model.predict(transformer_expanded_data).flatten() # creating initial momentum probabilities using the trained transformer model

def momentum_signal(vix_slope): # defining a function for the momentum signal
    return np.sign(vix_slope)  # momentum signal is activated when VIX future slope is decreasing (contango steepening)

def mean_reversion_signal(stock_returns): # defining a function for the mean reversion signal
    return -np.sign(stock_returns)  # mean reversion signal is activated when returns are extreme

# generating trading signals
momentum_component = initial_mom_prob * momentum_signal(forward_vix_slope['vix_slope']) # momentum signals are given by the momentum probabilities multiplied by whether the vix slope gives us momentum vibes
mean_reversion_component = (1 - initial_mom_prob) * mean_reversion_signal(spy_prices_pred_2025['Close'].pct_change()) # mean reversion signals are given by the momentum probabilities multiplied by whether the returns give us mean reversion vibes
trading_signal = momentum_component + mean_reversion_component # summing

strategy_type = np.where(momentum_component > mean_reversion_component, "Momentum", "Mean Reversion") # showing the active strategy at each time step

def backtest_trading_signals(signals, stock_prices, initial_capital = 1000): # function for an initial backtest with a simulated initial capital of 1,000 euro
    monthly_returns = stock_prices.pct_change().fillna(0) # monthly returns are calculated as percentage difference
    strategy_returns = signals.shift(1).fillna(0) * monthly_returns  # shift to avoid lookahead bias
    cumulative_returns = (1 + strategy_returns).cumprod() # cumulative multiplication of returns
    portfolio_value = initial_capital * cumulative_returns  # portfolio evolution over time
    return portfolio_value, strategy_returns

# running backtest on the trading signal previously generated and the spy predicted prices for 2025
portfolio_value, strategy_returns = backtest_trading_signals(pd.Series(trading_signal, index = spy_prices_pred_2025.index), 
                                                              spy_prices_pred_2025['Close'])

initial_results_df = pd.DataFrame({ # grouping everything into a dataframe to show strategy switches and returns
    'Date': forward_vix_slope['Period'], # we take the months from the forward vix slope dataframe
    '2025 SPY prices': spy_prices_pred_2025['Close'], # SPY prices forecasted for 2025
    '2025 VIX slope': forward_vix_slope['vix_slope'], # forward VIX futures term structure slope
    'Trading signal': trading_signal, # trading signal previously generated
    'Active strategy': strategy_type, # switches between momentum and mean reversion
    'Daily strategy return': strategy_returns
})
initial_results_df.set_index('Date', inplace = True) # setting dates as index

# plotting portfolio value over the next 7/8 months
plt.figure(figsize = (12, 6)) # size of the figure
plt.plot(forward_vix_slope['Period'], portfolio_value, label = 'Portfolio value evolution', color = 'blue') 
plt.xlabel('Date') # setting label of x-axis
plt.ylabel('Portfolio Value') # setting label of y-axis
plt.title('Portfolio value evolution - Momentum vs. Mean Reversion') # title of the plot
plt.legend() # adding the legend
plt.show()

plt.figure(figsize = (12, 6)) # plotting the table with strategy selection
sns.heatmap(pd.DataFrame(strategy_type, index = spy_prices_pred_2025['Month'], columns = ['Strategy']).T == "Momentum", 
            cmap = ['red', 'green'], cbar = False) # we are displaying the Strategy column of strategy_type variable
plt.title("Evolution of strategy selection in 2025 (Green = Momentum, Red = Mean Reversion)") # title of the plot
plt.show()


# %%

### 4. Innovative double-check and Backtesting

historical_df = correlation_dataset # storing historical data of vix slope and spy returns in a new dataframe only for ease of understanding with names of the variables
df_2025 = pd.DataFrame({ # grouping the 2025 vix slope and spy rets data together, as before
    '2025 VIX futures slope': forward_vix_slope['vix_slope'],
    '2025 SPY prices': spy_prices_pred_2025['Close']
}).dropna() # dropping NAs

## following the same reasoning of the 3 functions initially created, we decided to 
## compute rolling indicators but using an Exponentially Weighted Moving Average (EWMA),
## in order to give more relevance to recent observations (volatility of 2024 is much more relevant than 2007 for predicting 2025)
rolling_window = 5  # the idea is to reproduce initial functions with 5-day rolling window but with more weight on recent data

vol_indicator_hist = historical_df['SPY returns'].ewm(span = rolling_window).std() # calculating historical rolling volatility indicator on the historical returns of SPY
vol_indicator_2025 = df_2025['2025 SPY prices'].pct_change().ewm(span = rolling_window).std() # calculating rolling volatility indicator on the 2025 returns of SPY

# the same for rolling correlation
correlation_indicator_hist = historical_df['SPY returns'].ewm(span = rolling_window).corr(historical_df['vix_slope'])
correlation_indicator_2025 = df_2025['2025 SPY prices'].pct_change().ewm(span = rolling_window).corr(df_2025['2025 VIX futures slope'])

roc_indicator_hist = historical_df['vix_slope'].diff() # for the moment, we considered here to calculate rate of change (ROC) of VIX slope as absolute difference to avoid having inf and nan
roc_indicator_2025 = df_2025['2025 VIX futures slope'].diff() # rate of change for 2025 vix slope

volatility_mean = vol_indicator_hist.mean() # setting volatility threshold to assess if we should apply momentum or mean reversion

# Determine confirmation signals based on thresholds
def confirm_strategy(returns, vix_slope, vol, corr, roc):
    conditions_momentum = (vol < volatility_mean) & (corr > 0.3) & (roc < 0)
    conditions_mean_reversion = (vol > volatility_mean) & (corr < -0.3) & (roc > 0)
    
    strategy = np.where(conditions_momentum, 'Momentum',
                np.where(conditions_mean_reversion, 'Mean Reversion', 'Momentum'))
    return strategy

# Determine confirmation signals based on thresholds
def confirm_strategy(returns, vix_slope, vol, corr, roc):
    # Favor Momentum if volatility is low
    momentum_strong = (vol < volatility_mean)
    momentum_confirmed = (momentum_strong & (corr > 0.3)) | (momentum_strong & (roc < 0))
    
    # Favor Mean Reversion if volatility is high
    mean_reversion_strong = (vol > volatility_mean)
    mean_reversion_confirmed = (mean_reversion_strong & (corr < -0.3)) | (mean_reversion_strong & (roc > 0))

    # Assign strategy based on conditions
    strategy = np.where(momentum_confirmed, 'Momentum',
                np.where(mean_reversion_confirmed, 'Mean Reversion', 'Momentum'))  # Default to Momentum
    
    return strategy


# Apply strategy confirmation
strategy_2025 = confirm_strategy(df_2025['2025 SPY prices'], df_2025['2025 VIX futures slope'], vol_indicator_2025, correlation_indicator_2025, roc_indicator_2025)

# Plot the results
fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
ax[0].plot(df_2025.index, portfolio_value, label='Portfolio Value', color='black')
ax[0].set_title('Portfolio Value')
ax[0].legend()

ax[1].plot(df_2025.index, vol_indicator_2025, label='Rolling Volatility', color='blue')
ax[1].axhline(volatility_mean, linestyle='--', color='red', label='Median Volatility')
ax[1].set_title('Rolling Volatility')
ax[1].legend()

ax[2].plot(df_2025.index, correlation_indicator_2025, label='Rolling Correlation (SPY vs VIX Slope)', color='purple')
ax[2].axhline(0.3, linestyle='--', color='green', label='Momentum Threshold')
ax[2].axhline(-0.3, linestyle='--', color='red', label='Mean Reversion Threshold')
ax[2].set_title('Rolling Correlation')
ax[2].legend()

ax[3].plot(df_2025.index, roc_indicator_2025, label='Rate of Change of VIX Slope', color='orange')
ax[3].axhline(0, linestyle='--', color='black', label='Zero Line')
ax[3].set_title('Rate of Change (ROC) of VIX Slope')
ax[3].legend()

plt.tight_layout()
plt.show()






# Assuming strategy_type, forward_vix_slope, and other needed data are already available

# Define thresholds based on historical data
volatility_threshold = volatility_mean
correlation_threshold_low = -0.3
correlation_threshold_high = 0.3

# Confirmation check based on the three indicators
confirmation_check = []
for i in range(len(vol_indicator_2025)):
    if vol_indicator_2025[i] < volatility_threshold:
        vol_signal = "Momentum"
    else:
        vol_signal = "Mean Reversion"
    
    if correlation_indicator_2025[i] < correlation_threshold_low:
        corr_signal = "Mean Reversion"
    elif correlation_indicator_2025[i] > correlation_threshold_high:
        corr_signal = "Momentum"
    else:
        corr_signal = None  # Neutral case, doesn't strongly favor either
    
    if roc_indicator_2025[i] > 0:
        roc_signal = "Mean Reversion"
    else:
        roc_signal = "Momentum"
    
    # Final confirmation check based on majority vote
    signals = [vol_signal, corr_signal, roc_signal]
    signals = [s for s in signals if s is not None]  # Remove neutral cases
    
    if signals.count("Momentum") > signals.count("Mean Reversion"):
        confirmation_check.append("Momentum")
    else:
        confirmation_check.append("Mean Reversion")

# Convert to DataFrame
confirmation_check = pd.Series(confirmation_check, index=forward_vix_slope['Period'])

# Compare with original strategy selection
strategy_comparison = pd.DataFrame(strategy_type, index=forward_vix_slope['Period'], columns=['Strategy'])
strategy_comparison['Confirmation'] = confirmation_check


# Hybrid strategy selection combining transformer with confirmation indicators
def hybrid_strategy_selection(transformer_signal, vol, corr, roc):
    # Volatility is the primary factor
    momentum_strong = vol < volatility_mean
    mean_reversion_strong = vol > volatility_mean
    
    # Secondary confirmations
    momentum_confirmed = momentum_strong & ((corr > 0.3) | (roc < 0))
    mean_reversion_confirmed = mean_reversion_strong & ((corr < -0.3) | (roc > 0))

    # Hybrid selection: Transformer signal + Confirmation
    strategy = np.where(momentum_confirmed, 'Momentum',
                np.where(mean_reversion_confirmed, 'Mean Reversion', transformer_signal))  # Default to transformer
    
    return strategy

# Function to evaluate performance
def evaluate_performance(cumulative_returns):
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (1 / (len(cumulative_returns) / 252)) - 1
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

# Apply the hybrid strategy on 2025 data
final_strategy = hybrid_strategy_selection(strategy_comparison['Strategy'], vol_indicator_2025, correlation_indicator_2025, roc_indicator_2025)

def compute_portfolio_returns(strategy, returns):
    """
    Computes portfolio returns based on strategy selection.
    - If 'Momentum', we assume we follow the market return.
    - If 'Mean Reversion', we assume we take the opposite sign of returns.
    """
    portfolio_returns = np.where(strategy == 'Momentum', returns, -returns)
    return pd.Series(portfolio_returns, index=returns.index)

# Compute portfolio returns based on the final strategy
final_portfolio_returns = compute_portfolio_returns(final_strategy, df_2025['2025 SPY prices'].pct_change())

# Convert returns to cumulative returns
final_cumulative_returns = (1 + final_portfolio_returns).cumprod()

# Evaluate final strategy performance
print("Final Strategy Performance:")
evaluate_performance(final_cumulative_returns)

plt.figure(figsize=(12, 6))
plt.plot((1 + compute_portfolio_returns(strategy_comparison['Strategy'], df_2025['2025 SPY prices'].pct_change())).cumprod(), label="Momentum Transformer Only", linestyle='dashed')
plt.plot(final_cumulative_returns, label="Hybrid Strategy (Transformer + Indicators)", linewidth=2)
plt.legend()
plt.title("Cumulative Returns: Transformer vs Hybrid Strategy")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid()
plt.show()
