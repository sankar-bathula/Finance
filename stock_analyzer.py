"""
╔══════════════════════════════════════════════════════════════════════╗
║           📊  MULTI-STRATEGY STOCK ANALYZER                         ║
║  Covers: Long-Term | Swing Trading | Factor | Sector Rotation        ║
╚══════════════════════════════════════════════════════════════════════╝

INSTALL DEPENDENCIES:
    pip install yfinance pandas numpy tabulate colorama

RUN:
    python stock_analyzer.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
from colorama import Fore, Style, init
import warnings
warnings.filterwarnings("ignore")

init(autoreset=True)  # colorama init

# ─────────────────────────────────────────────
#  CONFIGURATION — Edit tickers as needed
# ─────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ"]
ticker  = pd.read_csv("Data\Stocks.xlsx")["Sector"].tolist()
print(ticker)
exit()

SECTOR_MAP = {
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "GOOGL": "Communication Services",
    "AMZN":  "Consumer Discretionary",
    "NVDA":  "Technology",
    "JPM":   "Financials",
    "XOM":   "Energy",
    "JNJ":   "Healthcare",
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def print_header(title: str, color=Fore.CYAN):
    width = 70
    print()
    print(color + "═" * width)
    print(color + f"  {title}")
    print(color + "═" * width)
    print(Style.RESET_ALL)


def fetch_stock(ticker: str, period: str = "1y") -> tuple[yf.Ticker, pd.DataFrame]:
    """Returns (Ticker object, historical price DataFrame)."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    return t, hist


def color_value(val, good_above=0):
    """Color a number green if above threshold, red otherwise."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return Fore.WHITE + "N/A"
    color = Fore.GREEN if val >= good_above else Fore.RED
    return color + f"{val:.2f}" + Style.RESET_ALL


# ─────────────────────────────────────────────
#  1. 📈  LONG-TERM INVESTING
#     Focus: Fundamentals, growth, valuation
# ─────────────────────────────────────────────

def analyze_longterm(ticker: str) -> dict:
    """Extract fundamentals for long-term value/growth assessment."""
    t, hist = fetch_stock(ticker, period="5y")
    info = t.info

    # Price performance
    price_now = hist["Close"].iloc[-1]
    price_1y_ago = hist["Close"].iloc[-252] if len(hist) >= 252 else hist["Close"].iloc[0]
    price_5y_ago = hist["Close"].iloc[0]
    ret_1y = (price_now / price_1y_ago - 1) * 100
    ret_5y = (price_now / price_5y_ago - 1) * 100

    return {
        "Ticker":         ticker,
        "Price":          round(price_now, 2),
        "P/E Ratio":      info.get("trailingPE"),
        "P/B Ratio":      info.get("priceToBook"),
        "EPS (TTM)":      info.get("trailingEps"),
        "Revenue Growth": info.get("revenueGrowth"),
        "Profit Margin":  info.get("profitMargins"),
        "ROE":            info.get("returnOnEquity"),
        "Debt/Equity":    info.get("debtToEquity"),
        "Dividend Yield": info.get("dividendYield"),
        "1Y Return (%)":  round(ret_1y, 2),
        "5Y Return (%)":  round(ret_5y, 2),
    }


def print_longterm_report(tickers: list[str]):
    print_header("📈  LONG-TERM INVESTING — Fundamental Analysis", Fore.GREEN)
    rows = []
    for t in tickers:
        print(f"  Fetching {t}...", end="\r")
        rows.append(analyze_longterm(t))

    df = pd.DataFrame(rows).set_index("Ticker")

    # Pretty print key columns
    cols = ["Price", "P/E Ratio", "P/B Ratio", "ROE", "Profit Margin",
            "Revenue Growth", "Debt/Equity", "Dividend Yield",
            "1Y Return (%)", "5Y Return (%)"]
    print(tabulate(df[cols].round(3).fillna("N/A"), headers="keys", tablefmt="fancy_grid"))

    print(Fore.YELLOW + "\n  💡 Long-Term Tips:")
    print("     • Low P/E + high ROE = quality at value")
    print("     • Revenue Growth > 15% signals strong compounder")
    print("     • Debt/Equity < 1.0 preferred for stability")


# ─────────────────────────────────────────────
#  2. ⚡  SWING TRADING
#     Focus: Technical indicators, momentum
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return round(macd.iloc[-1], 4), round(signal.iloc[-1], 4), round(histogram.iloc[-1], 4)


def compute_bollinger(series: pd.Series, window: int = 20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = series.iloc[-1]
    pct_b = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
    return round(upper.iloc[-1], 2), round(sma.iloc[-1], 2), round(lower.iloc[-1], 2), round(pct_b, 2)


def swing_signal(rsi, macd_hist, pct_b) -> str:
    """Simple rule-based swing signal."""
    score = 0
    if rsi < 40:   score += 1   # oversold
    if rsi > 60:   score -= 1   # overbought
    if macd_hist > 0: score += 1
    if macd_hist < 0: score -= 1
    if pct_b < 20: score += 1   # near lower band
    if pct_b > 80: score -= 1   # near upper band

    if score >= 2:   return Fore.GREEN  + "🟢 BUY"    + Style.RESET_ALL
    elif score <= -2: return Fore.RED   + "🔴 SELL"   + Style.RESET_ALL
    else:             return Fore.YELLOW + "🟡 HOLD"  + Style.RESET_ALL


def analyze_swing(ticker: str) -> dict:
    _, hist = fetch_stock(ticker, period="6mo")
    close = hist["Close"]

    rsi = compute_rsi(close)
    macd, signal, hist_val = compute_macd(close)
    bb_upper, bb_mid, bb_lower, pct_b = compute_bollinger(close)

    sma20  = round(close.rolling(20).mean().iloc[-1], 2)
    sma50  = round(close.rolling(50).mean().iloc[-1], 2)
    price  = round(close.iloc[-1], 2)

    vol_avg = round(hist["Volume"].rolling(20).mean().iloc[-1] / 1e6, 2)
    vol_now = round(hist["Volume"].iloc[-1] / 1e6, 2)
    vol_ratio = round(vol_now / vol_avg, 2) if vol_avg else None

    signal_str = swing_signal(rsi, hist_val, pct_b)

    return {
        "Ticker":       ticker,
        "Price":        price,
        "RSI(14)":      rsi,
        "MACD":         macd,
        "MACD Signal":  signal,
        "MACD Hist":    hist_val,
        "BB Upper":     bb_upper,
        "BB Mid":       bb_mid,
        "BB Lower":     bb_lower,
        "%B":           pct_b,
        "SMA20":        sma20,
        "SMA50":        sma50,
        "Vol (M)":      vol_now,
        "Vol Ratio":    vol_ratio,
        "Signal":       signal_str,
    }


def print_swing_report(tickers: list[str]):
    print_header("⚡  SWING TRADING — Technical Analysis", Fore.MAGENTA)
    rows = []
    for t in tickers:
        print(f"  Fetching {t}...", end="\r")
        rows.append(analyze_swing(t))

    df = pd.DataFrame(rows)
    display_cols = ["Ticker", "Price", "RSI(14)", "MACD Hist", "%B",
                    "SMA20", "SMA50", "Vol Ratio", "Signal"]
    print(tabulate(df[display_cols].set_index("Ticker"), headers="keys", tablefmt="fancy_grid"))

    print(Fore.YELLOW + "\n  💡 Swing Trading Tips:")
    print("     • RSI < 30 = oversold (potential bounce), RSI > 70 = overbought")
    print("     • MACD Hist turning positive = bullish momentum shift")
    print("     • %B < 20 = price near lower Bollinger Band (buy zone)")
    print("     • Vol Ratio > 1.5 = above-average volume confirms move")


# ─────────────────────────────────────────────
#  3. 🧠  FACTOR INVESTING
#     Factors: Value, Momentum, Quality, Size
# ─────────────────────────────────────────────

def compute_momentum_score(hist: pd.DataFrame) -> float:
    """12-1 month momentum (skip last month to avoid reversal)."""
    if len(hist) < 252:
        return np.nan
    price_now    = hist["Close"].iloc[-21]   # 1 month ago
    price_12m    = hist["Close"].iloc[-252]  # 12 months ago
    return round((price_now / price_12m - 1) * 100, 2)


def analyze_factor(ticker: str) -> dict:
    t, hist = fetch_stock(ticker, period="2y")
    info = t.info

    momentum = compute_momentum_score(hist)
    pe        = info.get("trailingPE")
    pb        = info.get("priceToBook")
    roe       = info.get("returnOnEquity")
    profit_m  = info.get("profitMargins")
    mktcap    = info.get("marketCap")

    # Factor scores (simple z-score proxies via raw values)
    value_score    = round(1 / pe if pe and pe > 0 else 0, 4)          # higher = cheaper
    quality_score  = round((roe or 0) + (profit_m or 0), 4)            # ROE + margin
    size_label     = (
        "Large Cap" if mktcap and mktcap > 10e9 else
        "Mid Cap"   if mktcap and mktcap > 2e9  else
        "Small Cap"
    )

    return {
        "Ticker":        ticker,
        "P/E":           round(pe, 2) if pe else None,
        "P/B":           round(pb, 2) if pb else None,
        "Value Score":   value_score,
        "Momentum(12-1)":momentum,
        "ROE":           round(roe * 100, 2) if roe else None,
        "Profit Margin": round(profit_m * 100, 2) if profit_m else None,
        "Quality Score": round(quality_score * 100, 2),
        "Market Cap":    f"${mktcap/1e9:.1f}B" if mktcap else "N/A",
        "Size":          size_label,
    }


def print_factor_report(tickers: list[str]):
    print_header("🧠  FACTOR INVESTING — Value · Momentum · Quality · Size", Fore.BLUE)
    rows = []
    for t in tickers:
        print(f"  Fetching {t}...", end="\r")
        rows.append(analyze_factor(t))

    df = pd.DataFrame(rows).set_index("Ticker")

    # Rank each factor
    df["Value Rank"]    = df["Value Score"].rank(ascending=False).astype(int)
    df["Momentum Rank"] = df["Momentum(12-1)"].rank(ascending=False).astype(int)
    df["Quality Rank"]  = df["Quality Score"].rank(ascending=False).astype(int)
    df["Composite Rank"]= (df["Value Rank"] + df["Momentum Rank"] + df["Quality Rank"])
    df = df.sort_values("Composite Rank")

    display_cols = ["P/E", "P/B", "Value Score", "Momentum(12-1)",
                    "Quality Score", "Market Cap", "Size",
                    "Value Rank", "Momentum Rank", "Quality Rank", "Composite Rank"]
    print(tabulate(df[display_cols].fillna("N/A"), headers="keys", tablefmt="fancy_grid"))

    top = df.index[0]
    print(Fore.GREEN + f"\n  🏆 Top Composite Factor Pick: {top}")
    print(Fore.YELLOW + "\n  💡 Factor Tips:")
    print("     • Value Score = earnings yield (1/PE); higher = cheaper")
    print("     • Momentum = 12-1 month price return; avoids near-term reversal")
    print("     • Quality = ROE + Profit Margin combined")
    print("     • Lower Composite Rank = better multi-factor stock")


# ─────────────────────────────────────────────
#  4. 🏦  SECTOR ROTATION
#     Analyze relative sector performance & momentum
# ─────────────────────────────────────────────

SECTOR_ETFS = {
    "Technology":              "XLK",
    "Financials":              "XLF",
    "Healthcare":              "XLV",
    "Energy":                  "XLE",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Industrials":             "XLI",
    "Communication Services":  "XLC",
    "Materials":               "XLB",
    "Utilities":               "XLU",
    "Real Estate":             "XLRE",
}


def analyze_sector_rotation() -> pd.DataFrame:
    """Fetch sector ETF performance to identify rotation signals."""
    rows = []
    for sector, etf in SECTOR_ETFS.items():
        try:
            _, hist = fetch_stock(etf, period="1y")
            close = hist["Close"]
            ret_1m  = (close.iloc[-1] / close.iloc[-21] - 1)  * 100
            ret_3m  = (close.iloc[-1] / close.iloc[-63] - 1)  * 100
            ret_6m  = (close.iloc[-1] / close.iloc[-126] - 1) * 100
            ret_1y  = (close.iloc[-1] / close.iloc[0] - 1)    * 100
            rsi     = compute_rsi(close)
            rows.append({
                "Sector": sector,
                "ETF":    etf,
                "1M (%)": round(ret_1m, 2),
                "3M (%)": round(ret_3m, 2),
                "6M (%)": round(ret_6m, 2),
                "1Y (%)": round(ret_1y, 2),
                "RSI":    rsi,
            })
        except Exception as e:
            rows.append({"Sector": sector, "ETF": etf,
                         "1M (%)": None, "3M (%)": None,
                         "6M (%)": None, "1Y (%)": None, "RSI": None})

    df = pd.DataFrame(rows).set_index("Sector")
    df["Momentum Score"] = (
        df["1M (%)"].rank() * 0.4 +
        df["3M (%)"].rank() * 0.35 +
        df["6M (%)"].rank() * 0.25
    ).round(2)
    return df.sort_values("Momentum Score", ascending=False)


def print_sector_rotation_report(tickers: list[str]):
    print_header("🏦  SECTOR ROTATION — Relative Strength & Momentum", Fore.CYAN)
    print("  Fetching sector ETF data...\n")
    df = analyze_sector_rotation()

    # Highlight stocks in top sectors
    top_sectors = df.index[:3].tolist()
    bottom_sectors = df.index[-3:].tolist()

    print(tabulate(df.fillna("N/A"), headers="keys", tablefmt="fancy_grid"))

    print(Fore.GREEN + f"\n  🔥 Overweight Sectors (strong momentum):")
    for s in top_sectors:
        etf = df.loc[s, "ETF"]
        print(f"     ✅ {s} ({etf})")

    print(Fore.RED + f"\n  ❄️  Underweight Sectors (weak momentum):")
    for s in bottom_sectors:
        etf = df.loc[s, "ETF"]
        print(f"     ⚠️  {s} ({etf})")

    # Map tickers to their sectors
    print(Fore.YELLOW + "\n  📌 Your Stock → Sector Alignment:")
    for ticker in tickers:
        sector = SECTOR_MAP.get(ticker, "Unknown")
        if sector in top_sectors:
            label = Fore.GREEN + "✅ Favorable Sector"
        elif sector in bottom_sectors:
            label = Fore.RED + "⚠️  Weak Sector"
        else:
            label = Fore.YELLOW + "🟡 Neutral Sector"
        print(f"     {ticker:6s} → {sector:30s} {label}" + Style.RESET_ALL)

    print(Fore.YELLOW + "\n  💡 Sector Rotation Tips:")
    print("     • Buy stocks in sectors showing strongest 1-3M momentum")
    print("     • Rotate out of sectors with RSI > 75 (overbought sectors)")
    print("     • Early cycle: Financials, Tech  |  Late cycle: Energy, Utilities")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(Fore.CYAN + Style.BRIGHT + """
╔══════════════════════════════════════════════════════════════════════╗
║        📊   MULTI-STRATEGY STOCK ANALYZER   📊                      ║
║   Long-Term · Swing Trading · Factor Investing · Sector Rotation     ║
╚══════════════════════════════════════════════════════════════════════╝
""" + Style.RESET_ALL)

    print(f"  Analyzing: {', '.join(TICKERS)}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    try:
        # ── 1. Long-Term Fundamentals ──────────────────────
        print_longterm_report(TICKERS)

        # ── 2. Swing Trading Technicals ───────────────────
        print_swing_report(TICKERS)

        # ── 3. Factor Investing ───────────────────────────
        print_factor_report(TICKERS)

        # ── 4. Sector Rotation ────────────────────────────
        print_sector_rotation_report(TICKERS)

        print()
        print(Fore.CYAN + "═" * 70)
        print(Fore.WHITE + "  ⚠️  DISCLAIMER: This tool is for educational purposes only.")
        print(Fore.WHITE + "  Not financial advice. Always do your own research.")
        print(Fore.CYAN + "═" * 70 + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"\n  Error: {e}")
        print(Fore.YELLOW + "  Make sure yfinance is installed: pip install yfinance pandas numpy tabulate colorama")


if __name__ == "__main__":
    main()
