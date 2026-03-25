import yfinance as yf
import pandas as pd
from datetime import datetime
import pyotp
import creds
from SmartApi import SmartConnect
import pandas as pd
import os



# -------------------------------
# ANGEL ONE LOGIN
# -------------------------------
def Generate_TOTP():
	totp = pyotp.TOTP(creds.totp_code)
	return totp.now()


def Generate_Session():
	smartApi = SmartConnect(creds.api_key)
	data = smartApi.generateSession(creds.client_code, creds.client_pin, Generate_TOTP())
	authToken = data['data']['jwtToken']
	refreshToken = data['data']['refreshToken']
	return(smartApi, refreshToken)


smartApi, refreshToken = Generate_Session()
print("Login Successful")

#-------------------------------
#STOCK LIST (NSE Symbols)
#Add stocks you want to scan
#-------------------------------
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "Data", "Stocks.xlsx")

portfolio = pd.read_excel(file_path)
stock_list = portfolio["ticker"].tolist()   

#stock_list = ["TATAMOTORS.NS", "INFY.NS", "IDEA.NS", "IRCTC.NS"]

filtered_stocks = [] 

for stock in stock_list:
    print(stock) 
    try:
        ticker = yf.Ticker(stock)
        info = ticker.info
        promoter_holding = info.get("heldPercentInsiders", 0) * 100
        market_cap = info.get("marketCap", 0) / 1e7   # Convert to Crores
        profit_margin = info.get("profitMargins", 0) * 100
        dividend_yield = info.get("dividendYield", 0)
        pe_ratio = info.get("trailingPE", 0)
        debt_to_equity = info.get("debtToEquity", 0)
        book_value = info.get("bookValue", 0)
        fifty_two_high = info.get("fiftyTwoWeekHigh", 0)
        fifty_two_low = info.get("fiftyTwoWeekLow", 0)
        current_price = info.get("currentPrice")

        # Listing age calculation
        hist = ticker.history(period="max")
        listing_year = hist.index[0].year
        years_listed = datetime.now().year - listing_year

        # Apply Filters
        if (
            promoter_holding > 35 and
            market_cap > 5000 and
            years_listed >= 5 and
            profit_margin >= 5 and
          #  dividend_yield and
            pe_ratio > 5 and
            debt_to_equity < 1
        ):
            filtered_stocks.append({
                "Stock": stock,
                "Promoter %": promoter_holding,
                "Market Cap (Cr)": market_cap,
                "Profit Margin %": profit_margin,
              #  "Dividend Yield": dividend_yield,
                "PE Ratio": pe_ratio,
                "Debt/Equity": debt_to_equity,
                "Book Value": book_value,
                "52W High": fifty_two_high,
                "52W Low": fifty_two_low,
                "Current Price" : current_price
            })

    except Exception as e:
        print(f"Error in {stock}: {e}")

# -------------------------------
# Output Results
# -------------------------------
df = pd.DataFrame(filtered_stocks)
print("\nFiltered Stocks:")
print(df) 
FilterdStocks = df.to_excel("Data\\output.xlsx", index=False)
