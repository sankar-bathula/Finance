import yfinance as yf

# Download NIFTY data
nifty = yf.download("^NSEI", period="5d", interval="1d")

# Make sure data exists
if len(nifty) >= 2:
    H = nifty['High'].iloc[-2]
    L = nifty['Low'].iloc[-2]
    C = nifty['Close'].iloc[-2]

    PP = (H + L + C) / 3
    R1 = 2*PP - L
    S1 = 2*PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)

    print("Pivot:", round(PP,2))
    print("Resistance 1:", round(R1,2))
    print("Support 1:", round(S1,2))
    print("Resistance 2:", round(R2,2))
    print("Support 2:", round(S2,2))
else:
    print("Not enough data downloaded")
