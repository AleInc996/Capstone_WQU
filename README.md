### Capstone_WQU
## Repository for code part of capstone project of group 7851 at WQU

This repository contains the work of Alessio Incelli and Tebogo Isia Dintoe on the project research "Stock Market Returns based on the State of the VIX Futures Term Structure".
The goal is to develop the following:

# 1. Data retrieval and manipulation
Downloading VIX futures term structure data (for different maturities) from Bloomberg and having them on an Excel file to then upload it in Python and manipulate them in order to get a constant maturity VIX futures term structure.
This means that we will have monthly data, that will then be used to calculate term structure slope and volatility.
This also means that stock market prices will then be retrieved, they must be monthly at this point, and turned into returns.
Data manipulation and exploratory data analysis will follow.
Data will be prepared for being fed as input to the momentum transformer model.

# 2. Momentum transformer model development
A transformer to switch between momentum and mean reversion strategies will be built.

# 3. Innovative double check
Volatility and correlations will be used in this phase to double check that the switches between strategies make sense.

# 4. Backtesting
The whole strategy will be tested on past data.
