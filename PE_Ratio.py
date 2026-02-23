"""
EPS: Earnings Per Share = Not Income /Total outstanding shares
P/E Ratio: Price to Earnings Ratio = Current Stock Price /Earnings Per Share (EPS)

"""
import yfinance as yf

# Ticker symbol (e.g., "INFY.NS" for Infosys on NSE)
ticker_symbol = "INFY.NS"

# Fetch data
stock = yf.Ticker(ticker_symbol)

# Get current stock price and EPS
hist = stock.history(period="5d")
current_price = hist["Close"].iloc[-1]

#current_price = stock.history(period="1d")["Close"][0]
eps = stock.info.get("trailingEps")

if eps:
    pe_ratio = current_price / eps
    print(f"Current Price: {current_price}")
    print(f"Earnings Per Share (EPS): {eps}")
    print(f"P/E Ratio: {pe_ratio:.2f}")
else:
    print("EPS data not available.")



# List of stocks (example)
tickers = ["INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS"]

valid_stocks = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    pe = stock.info.get("trailingPE")
    eps = stock.info.get("trailingEps")
    
    if pe and eps and eps > 0:
        # Define your rule
        if 10 < pe < 25:
            valid_stocks.append((ticker, pe))

print("Valid stocks based on PE filter:")
for stock in valid_stocks:
    print(stock)

