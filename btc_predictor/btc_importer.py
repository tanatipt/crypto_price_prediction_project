import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np

pd.set_option('use_inf_as_na', True)

# Downloading the BTC-USD quote from yfinance for each day between 2013-01-01 and 2023-07-03
btc_data = yf.download("BTC-USD", start="2013-01-01", end="2023-07-03")

# Computing the RSI of the BTC-USD price
btc_data['RSI'] = ta.rsi(btc_data['Close'], length=7)

# Computing the CCI of the BTC-USD price
cci = ta.cci(btc_data['High'], btc_data['Low'], btc_data['Close'], length=7)
btc_data['CCI'] = cci

# Computing the CMF of the BTC-USD price
cmf = ta.cmf(btc_data['High'], btc_data['Low'],
             btc_data['Close'], btc_data['Volume'])
btc_data['CMF'] = cmf

# Computing the VWAP of the BTC-USD price
vwap = ta.vwap(btc_data['High'], btc_data['Low'],
               btc_data['Close'], btc_data['Volume'])
btc_data['VWAP'] = vwap

# Computing the ADX, DMP and DMN of the BTC-USD price
adx = ta.adx(btc_data['High'], btc_data['Low'], btc_data['Close'], length=7)
btc_data['ADX'] = adx['ADX_7']
btc_data['DMP'] = adx['DMP_7']
btc_data['DMN'] = adx['DMN_7']

# Computing the ATR of the BTC-USD price
atr = ta.atr(btc_data['High'], btc_data['Low'], btc_data['Close'], length=7)
btc_data['ATR'] = atr

# Computing the MACD of the BTC-USD price
macd = ta.macd(btc_data['Close'])
btc_data['MACD_LINE'] = macd['MACD_12_26_9']
btc_data['MACD_HISTOGRAM'] = macd['MACDh_12_26_9']
btc_data['MACD_SIGNAL'] = macd['MACDs_12_26_9']

# Computing the Bollinger Bands of the BTC-USD price
bbands = ta.bbands(btc_data['Close'], length=7)
btc_data['BBL'] = bbands['BBL_7_2.0']
btc_data['BBM'] = bbands['BBM_7_2.0']
btc_data['BBU'] = bbands['BBU_7_2.0']
btc_data['BBB'] = bbands['BBB_7_2.0']
btc_data['BBP'] = bbands['BBP_7_2.0']

# Computing the EMA of the BTC-USD price
btc_data['EMAF'] = ta.ema(btc_data['Close'], length=7)
btc_data['EMAM'] = ta.ema(btc_data['Close'], length=21)
btc_data['EMAS'] = ta.ema(btc_data['Close'], length=63)
btc_data['OBV'] = ta.obv(btc_data['Close'], btc_data['Volume'])

# Computing the Stochastic Oscillators of the BTC-USD price
stoch = ta.stoch(
    btc_data['High'], btc_data['Low'], btc_data['Close'])
btc_data['STOCHk'] = stoch['STOCHk_14_3_3']
btc_data['STOCHd'] = stoch['STOCHd_14_3_3']

# Computing the Aroon Indicator of the BTC-USD price
aroon = ta.aroon(
    btc_data['High'], btc_data['Low'], length=7)
btc_data['AROONd'] = aroon['AROOND_7']
btc_data['AROONu'] = aroon['AROONU_7']
btc_data['AROONosc'] = aroon['AROONOSC_7']

# Computing the Ichimoku Cloud of the BTC-USD price
ichimoku = ta.ichimoku(
    btc_data['High'], btc_data['Low'], btc_data['Close'], lookahead=False)
visible = ichimoku[0]
forward = ichimoku[1]

btc_data["VISIBLE_ISA"] = visible["ISA_9"]
btc_data["VISIBLE_ISB"] = visible["ISB_26"]
btc_data["VISIBLE_ITS"] = visible["ITS_9"]
btc_data["VISIBLE_IKS"] = visible["IKS_26"]


# Computing the target of each date, which is the next day closing price
btc_data['Target'] = btc_data["Close"].shift(-1)

btc_data = btc_data.drop(
    ['Adj Close'], axis=1)
# Removing any date that contains a null value for any of its feature
btc_data = btc_data.dropna()
# Resetting the index of the BTC-USD Dataframe
btc_data = btc_data.reset_index()
btc_data.to_csv("../coin_data/btc_data.csv", index=False)
