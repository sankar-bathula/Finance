import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def analyze_stock(ticker_symbol, days=365):
    """
    Fetches historical stock data from Yahoo Finance,
    calculates Moving Averages (50-day and 200-day) and RSI (14-day),
    and plots the data using Plotly.
    """
    print(f"Fetching data for {ticker_symbol}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch data
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(start=start_date, end=end_date).dropna(subset=['Close'])
    
    if df.empty:
        print(f"No data found for {ticker_symbol}")
        return
        
    # Calculate Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Current Price: INR {df['Close'].iloc[-1]:.2f}")
    print(f"50-Day MA: INR {df['MA50'].iloc[-1]:.2f}")
    print(f"200-Day MA: INR {df['MA200'].iloc[-1]:.2f}")
    print(f"Current RSI: {df['RSI'].iloc[-1]:.2f}")
    print(f"Max Price (Last {days} days): INR {df['High'].max():.2f}")
    print(f"Min Price (Last {days} days): INR {df['Low'].min():.2f}")
    
    # Create interactive plot with 2 rows (Price and RSI)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3])
    
    # Add Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='orange', width=2), name='50-Day MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='blue', width=2), name='200-Day MA'), row=1, col=1)
    
    # Add RSI chart
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=2), name='RSI (14)'), row=2, col=1)
    
    # Add RSI Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker_symbol} Stock Analysis (Last {days} days)',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Stock Price (INR)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    
    # Save the plot as HTML so it can be viewed in browser
    output_file = f"{ticker_symbol.replace('.NS', '')}_analysis.html"
    fig.write_html(output_file)
    print(f"\nAnalysis plot saved to: {output_file}")
    print(f"Open {output_file} in your browser to view the interactive chart.")

if __name__ == "__main__":
    # Using RELIANCE.NS as the default for NSE listed Reliance Industries
    analyze_stock("RELIANCE.NS", days=365)
