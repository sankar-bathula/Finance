import pandas as pd
import yfinance as yf
from datetime import datetime
import seaborn as plt

ticker = yf.Ticker("RELIANCE.NS")
print(ticker.info['sharesOutstanding'])
# get stock info
ticker.info
# get historical market data
ticker.history(period="max")
# show actions (dividends, splits)
ticker.actions
# show dividends
ticker.dividends
# show splits
ticker.splits
#show current analyst reccomednations
print(ticker.recommendations)
#see major holders
ticker.major_holders
df = ticker.institutional_holders
print(df)