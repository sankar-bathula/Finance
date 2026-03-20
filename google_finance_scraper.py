import requests
from bs4 import BeautifulSoup

def get_google_finance_data(ticker, exchange="NASDAQ"):
    """
    Scrapes basic stock data from Google Finance.
    Example: ticker='AAPL', exchange='NASDAQ'
    """
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve data for {ticker}:{exchange}")
        return None
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting the current price
    price_element = soup.find('div', class_='YMlKec fxKbKc')
    current_price = price_element.text.strip() if price_element else "N/A"
    
    # Extracting the previous close
    # The class names in Google Finance change frequently, this is a common one for stats
    stats_elements = soup.find_all('div', class_='P6K39c')
    previous_close = stats_elements[0].text.strip() if len(stats_elements) > 0 else "N/A"
    
    # Extracting company name
    name_element = soup.find('div', class_='zzDege')
    company_name = name_element.text.strip() if name_element else ticker
    
    return {
        "Company": company_name,
        "Ticker": ticker,
        "Exchange": exchange,
        "Current Price": current_price,
        "Previous Close": previous_close,
        "URL": url
    }

if __name__ == "__main__":
    # Example usage
    stock_info = get_google_finance_data("RELIANCE", "NSE")
    print("--- Google Finance Data ---")
    for key, value in stock_info.items():
        print(f"{key}: {value}")
