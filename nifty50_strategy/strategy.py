import pandas as pd


class Nifty50Strategy:
    def __init__(self, data):
        self.data = data
        self.daily_pivots = pd.DataFrame()
        self.trades = []

    def calculate_pivots(self):
        """
        Calculates Standard Pivot Points based on Daily OHLC.
        Logic: Resample 5m data to Daily to find prev day's High, Low, Close.
        """
        # Resample to Daily to get H, L, C
        # Note: ensuring we have correct time boundaries
        daily_df = self.data.resample('D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        # Calculate Pivots for the *next* day (shift 1)
        # Pivot = (H + L + C) / 3
        daily_df['Pivot'] = (daily_df['High'] + daily_df['Low'] + daily_df['Close']) / 3
        daily_df['R1'] = (2 * daily_df['Pivot']) - daily_df['Low']
        daily_df['S1'] = (2 * daily_df['Pivot']) - daily_df['High']
        daily_df['R2'] = daily_df['Pivot'] + (daily_df['High'] - daily_df['Low'])
        daily_df['S2'] = daily_df['Pivot'] - (daily_df['High'] - daily_df['Low'])
        
        # Shift so that today's row contains *yesterday's* values (which are today's pivots)
        # Actually, if we map by date, we want the pivot for day D to be based on D-1.
        self.daily_pivots = daily_df.shift(1)
        # Drop the first Nan
        self.daily_pivots.dropna(inplace=True)
        # print("Calculated Daily Pivots (Head):")
        # print(self.daily_pivots.head())

    def get_pivot_for_date(self, timestamp):
        # Normalize to date to lookup
        date_key = timestamp.normalize() 
        if date_key in self.daily_pivots.index:
            return self.daily_pivots.loc[date_key]
        return None

    def execute_strategy(self):
        """
        Runs the strat.
        Logic: 
        1. Iterate through 5m candles.
        2. Identify Signal: Breakout of PREVIOUS 5m candle.
           - Long: Close > Prev High
           - Short: Close < Prev Low
           (Or Price Crossing logic - simplified to Close for backtest stability)
        3. Target/SL:
           - Long Target: Next Resistance (R1, R2, etc.)
           - Short Target: Next Support (S1, S2, etc.)
           - SL: Low of Setup Candle (or specific points)
        """
        if self.daily_pivots.empty:
            self.calculate_pivots()

        in_position = False
        position_type = None # 'LONG' or 'SHORT'
        entry_price = 0
        stop_loss = 0
        target_price = 0
        
        # We need previous candle.
        # Iterating with index
        keys = self.data.index
        
        for i in range(1, len(self.data)):
            curr_time = keys[i]
            prev_time = keys[i-1]
            
            curr_candle = self.data.iloc[i]
            prev_candle = self.data.iloc[i-1]
            
            pivots = self.get_pivot_for_date(curr_time)
            if pivots is None:
                continue
                
            # Trading Hours Filter (e.g., 09:15 to 15:30 IST)
            # Assuming data is in correct timezone or just ignoring for simplicity of the core logic
            
            if not in_position:
                # ENTRY LOGIC
                # Buy: Current High breaks Prev High
                # Sell: Current Low breaks Prev Low
                # For robust backtest, we often check if Close > Prev High, but 'Intraday' requires granularity.
                # Simplification: If Close > Prev High -> Buy (Entered at Close)
                
                # Check Long
                if curr_candle['Close'] > prev_candle['High']:
                    # Trigger Long
                    entry_price = curr_candle['Close']
                    stop_loss = prev_candle['Low'] # SL at Low of signal candle
                    
                    # Determine Targ based on Pivots
                    # If below P, Target P. If above P, Target R1. 
                    p = pivots['Pivot']
                    r1 = pivots['R1']
                    r2 = pivots['R2']
                    
                    if entry_price < p:
                        target_price = p
                    elif entry_price < r1:
                        target_price = r1
                    else:
                        target_price = r2
                        
                    in_position = True
                    position_type = 'LONG'
                    self.trades.append({
                        'Entry Time': curr_time,
                        'Type': 'LONG',
                        'Entry Price': entry_price,
                        'SL': stop_loss,
                        'Target': target_price
                    })

                # Check Short
                elif curr_candle['Close'] < prev_candle['Low']:
                    # Trigger Short
                    entry_price = curr_candle['Close']
                    stop_loss = prev_candle['High']
                    
                    p = pivots['Pivot']
                    s1 = pivots['S1']
                    s2 = pivots['S2']
                    
                    if entry_price > p:
                        target_price = p
                    elif entry_price > s1:
                        target_price = s1
                    else:
                        target_price = s2
                        
                    in_position = True
                    position_type = 'SHORT'
                    self.trades.append({
                        'Entry Time': curr_time,
                        'Type': 'SHORT',
                        'Entry Price': entry_price,
                        'SL': stop_loss,
                        'Target': target_price
                    })

            else:
                # EXIT LOGIC
                # Check Hit SL or Hit Target
                # Using High/Low of current candle to see if we got hit
                
                exit_price = None
                exit_reason = None
                
                if position_type == 'LONG':
                    if curr_candle['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'SL Hit'
                    elif curr_candle['High'] >= target_price:
                        exit_price = target_price
                        exit_reason = 'Target Hit'
                elif position_type == 'SHORT':
                    if curr_candle['High'] >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'SL Hit'
                    elif curr_candle['Low'] <= target_price:
                        exit_price = target_price
                        exit_reason = 'Target Hit'
                
                if exit_price is not None:
                    # Update last trade
                    self.trades[-1]['Exit Time'] = curr_time
                    self.trades[-1]['Exit Price'] = exit_price
                    self.trades[-1]['Reason'] = exit_reason
                    self.trades[-1]['PnL'] = (exit_price - entry_price) if position_type == 'LONG' else (entry_price - exit_price)
                    
                    in_position = False
                    position_type = None

        return pd.DataFrame(self.trades)

if __name__ == "__main__":
    from data_loader import fetch_nifty_data
    df = fetch_nifty_data(period="5d", interval="5m")
    if not df.empty:
        strat = Nifty50Strategy(df)
        strat.calculate_pivots()
        trades_df = strat.execute_strategy()
        print("Trades Generated:")
        print(trades_df)
        if not trades_df.empty:
            print("Total PnL:", trades_df['PnL'].sum())
