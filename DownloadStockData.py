import yfinance as yf
import datetime

# Choose the BSE ticker for Reliance
ticker_symbol = "RELIANCE.BO"

# Calculate dates for last 1 year
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)

# Download data
reliance = yf.Ticker(ticker_symbol)
df = reliance.history(start=start_date.strftime('%Y-%m-%d'),
                      end=end_date.strftime('%Y-%m-%d'),
                      interval="1d")

# Save to CSV
df.to_csv("Data\\"+ticker_symbol.split(".")[0]+"_last_1year.csv")
print("Saved "+ticker_symbol+" last 1 year data ")

