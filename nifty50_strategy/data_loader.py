import yfinance as yf
import pandas as pd

def fetch_nifty_data(period="5d", interval="5m"):
    """
    Fetches Nifty 50 data from Yahoo Finance.
    
    Args:
        period (str): Data period to download (e.g., "1d", "5d", "1mo").
                      Max 60d for 5m interval.
        interval (str): Data interval (e.g., "1m", "5m", "15m", "1d").
        
    Returns:
        pd.DataFrame: DataFrame containing the historical data.
    """
    ticker = "^NSEI"
    print(f"Fetching data for {ticker} with period={period} and interval={interval}...")
    data = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)
    
    if data.empty:
        print("No data fetched. Please check your internet connection or ticker symbol.")
        return pd.DataFrame()

    # Drop rows with NaN values if any
    data.dropna(inplace=True)
    
    # Ensure index is timezone aware (IST usually, but yfinance gives UTC or exchange time)
    # We'll leave it as is for now, but good to note.
    
    print(f"Fetched {len(data)} rows.")
    return data

if __name__ == "__main__":
    df = fetch_nifty_data()
    print(df.head())
    print(df.tail())
