import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "Data", "Stocks.xlsx")

portfolio = pd.read_excel(file_path)
tickers = portfolio["ticker"].tolist()
#print(tickers)
start_date = "2022-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")
stock_data = yf.download(tickers, start=start_date, end=end_date).Close
print(stock_data)

portfolio_returns = pd.DataFrame(stock_data)
portfolio_returns = portfolio_returns.pct_change().sum(axis=1)
print(portfolio_returns)

#portfolio_returns.to_excel("Data\portfolio_returns.xlsx")

