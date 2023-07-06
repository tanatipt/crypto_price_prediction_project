import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np

pd.set_option('use_inf_as_na', True)

# Downloading the ETH-USD quote from yfinance for each day between 2014-01-01 and 2023-06-19
data_frame = None


for pair, symbol in [("BTC-USD", "BTC"), ("ETH-USD", "ETH"), ("BNB-USD", "BNB"), ("XRP-USD", "XRP"), ("ADA-USD", "ADA")]:
    symbol_data = yf.download(pair,  start="2013-01-01", end="2023-07-05")
    symbol_data = symbol_data.drop(["Adj Close"], axis=1)
    symbol_data.columns = [symbol + "_Open",
                           symbol + "_High", symbol + "_Low", symbol + "_Close", symbol + "_Volume"]

    print(symbol_data)
    if data_frame is None:
        data_frame = symbol_data
    else:
        data_frame = data_frame.merge(symbol_data, how="left", on="Date")

print(data_frame)


def compute_label(y):
    if y >= 0:
        return 1
    elif y < 0:
        return 0
    else:
        return y


target = data_frame['ETH_Close'].diff().apply(compute_label)
data_frame = data_frame
# Computing the target of each date, which is the next day closing price
data_frame['Target'] = target.shift(-1)

# Removing any date that contains a null value for any of its feature
data_frame = data_frame.dropna()
data_frame = data_frame.reset_index()
print(data_frame)
data_frame.to_csv("../coin_data/eth_data.csv", index=False)
