import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np

pd.set_option('use_inf_as_na', True)

# Downloading the ETH-USD quote from yfinance for each day between 2014-01-01 and 2023-07-03
eth_data = yf.download("ETH-USD", start="2014-01-01", end="2023-07-03")

# Computing the RSI of the ETH-USD price
eth_data['RSI'] = ta.rsi(eth_data['Close'], length=7)

# Computing the CCI of the ETH-USD price
cci = ta.cci(eth_data['High'], eth_data['Low'], eth_data['Close'], length=7)
eth_data['CCI'] = cci

# Computing the CMF of the ETH-USD price
cmf = ta.cmf(eth_data['High'], eth_data['Low'],
             eth_data['Close'], eth_data['Volume'])
eth_data['CMF'] = cmf

# Computing the VWAP of the ETH-USD price
vwap = ta.vwap(eth_data['High'], eth_data['Low'],
               eth_data['Close'], eth_data['Volume'])
eth_data['VWAP'] = vwap

# Computing the ADX, DMP and DMN of the ETH-USD price
adx = ta.adx(eth_data['High'], eth_data['Low'], eth_data['Close'], length=7)
eth_data['ADX'] = adx['ADX_7']
eth_data['DMP'] = adx['DMP_7']
eth_data['DMN'] = adx['DMN_7']

# Computing the ATR of the ETH-USD price
atr = ta.atr(eth_data['High'], eth_data['Low'], eth_data['Close'], length=7)
eth_data['ATR'] = atr

# Computing the MACD of the ETH-USD price
macd = ta.macd(eth_data['Close'])
eth_data['MACD_LINE'] = macd['MACD_12_26_9']
eth_data['MACD_HISTOGRAM'] = macd['MACDh_12_26_9']
eth_data['MACD_SIGNAL'] = macd['MACDs_12_26_9']

# Computing the Bollinger Bands of the ETH-USD price
bbands = ta.bbands(eth_data['Close'], length=7)
eth_data['BBL'] = bbands['BBL_7_2.0']
eth_data['BBM'] = bbands['BBM_7_2.0']
eth_data['BBU'] = bbands['BBU_7_2.0']
eth_data['BBB'] = bbands['BBB_7_2.0']
eth_data['BBP'] = bbands['BBP_7_2.0']

# Computing the EMA of the ETH-USD price
eth_data['EMAF'] = ta.ema(eth_data['Close'], length=7)
eth_data['EMAM'] = ta.ema(eth_data['Close'], length=21)
eth_data['EMAS'] = ta.ema(eth_data['Close'], length=63)
eth_data['OBV'] = ta.obv(eth_data['Close'], eth_data['Volume'])

# Computing the Stochastic Oscillators of the ETH-USD price
stoch = ta.stoch(
    eth_data['High'], eth_data['Low'], eth_data['Close'])
eth_data['STOCHk'] = stoch['STOCHk_14_3_3']
eth_data['STOCHd'] = stoch['STOCHd_14_3_3']

# Computing the Aroon Indicator of the ETH-USD price
aroon = ta.aroon(
    eth_data['High'], eth_data['Low'], length=7)
eth_data['AROONd'] = aroon['AROOND_7']
eth_data['AROONu'] = aroon['AROONU_7']
eth_data['AROONosc'] = aroon['AROONOSC_7']

# Computing the Ichimoku Cloud of the ETH-USD price
ichimoku = ta.ichimoku(
    eth_data['High'], eth_data['Low'], eth_data['Close'], lookahead=False)
visible = ichimoku[0]
forward = ichimoku[1]

eth_data["VISIBLE_ISA"] = visible["ISA_9"]
eth_data["VISIBLE_ISB"] = visible["ISB_26"]
eth_data["VISIBLE_ITS"] = visible["ITS_9"]
eth_data["VISIBLE_IKS"] = visible["IKS_26"]


# Computing the target of each date, which is the next day closing price
eth_data['Target'] = eth_data["Close"].shift(-1)

eth_data = eth_data.drop(
    ['Adj Close'], axis=1)
# Removing any date that contains a null value for any of its feature
eth_data = eth_data.dropna()
# Resetting the index of the ETH-USD Dataframe
eth_data = eth_data.reset_index()
eth_data.to_csv("../coin_data/eth_data.csv", index=False)
