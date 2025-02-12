# Capstone_WQU
## Repository for code part of capstone project of group 7851 at WQU

This repository contains the work of Alessio Incelli and Tebogo Isia Dintoe on the project research "Stock Market Returns based on the State of the VIX Futures Term Structure".
The goal is to develop the following:

### 1. Data retrieval and manipulation
Downloading VIX futures term structure data (for different maturities, historical and for the year 2025) from Bloomberg and having them on two Excel files to then upload it in Python and manipulate them in order to get constant maturity VIX futures term structures. The final term structures will be characterized by monthly data, that will then be used to calculate term structure slope and volatility (rate of change).
Monthly stock market prices, for the SPY, are then retrieved, and turned into returns. Data manipulation and exploratory data analysis follows, together with definition and calculation of the three innovative indicators to be used. Data is prepared for being fed as input to the momentum transformer model.

### 2. Momentum transformer model development
A double LSTM approach is explored, as one of its pitfalls is the fact that it generates vanishing or exploding gradients, leading the way for the use of momentum transformer. Then, atransformer to switch between momentum and mean reversion strategies is built, using historical VIX futures term structure slope and past SPY returns for training. The historical dataframes are standardized and divided into sequences of observations before running the model. The first model consists of 2 LSTM approaches: the first one with 64 hidden neurons to initially learn from the data, and the second one with 32 hidden neurons to redefine the learning process. We make use of the Adam optimizer and of the Mean Squared Error as a measure of loss. The momentum transformer is developed with 4-head self-attention mechanism, 32 embeddings dimension, 128 hidden neurons for the feed forward network and two dense layers.

### 3. Forecasting 2025 scenarios and innovative double check
Generating momentum and mean-reversion signals based on the momentum transformer. Furthermore, rolling volatility and correlation, between VIX futures term structure slope and SPY returns, will be used in this phase to double check that the switches between strategies make sense.

### 4. Backtesting
The whole strategy will be tested on past data.
