import yfinance as yf
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# -----------------------
# Stock Symbols (NSE)
# You can expand this list
# -----------------------
symbols = [
    "TCS.NS", "INFY.NS", "ITC.NS", "SBIN.NS",
    "RELIANCE.NS", "HDFCBANK.NS", "LT.NS",
    "BEL.NS", "COALINDIA.NS", "IOC.NS"
]

# -----------------------
# Screener Function
# -----------------------
def run_screener():
    results = []

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info

            promoter = info.get("heldPercentInsiders", 0) * 100
            market_cap = info.get("marketCap", 0) / 1e7
            profit_margin = info.get("profitMargins", 0) * 100
            dividend = info.get("dividendYield", 0)
            pe = info.get("trailingPE", 0)
            debt_equity = info.get("debtToEquity", 0)
            face_value = info.get("faceValue", "N/A")
            book_value = info.get("bookValue", "N/A")
            high_52 = info.get("fiftyTwoWeekHigh", "N/A")
            low_52 = info.get("fiftyTwoWeekLow", "N/A")

            hist = ticker.history(period="max")
            if hist.empty:
                continue
            years_listed = datetime.now().year - hist.index[0].year

            # Apply filters
            if (
                promoter >= 35 and
                market_cap < 5000 and
                years_listed > 5 and
                profit_margin > 5 and
                dividend and
                pe and pe > 0 and
                debt_equity < 1
            ):
                results.append([
                    sym,
                    round(promoter, 2),
                    round(market_cap, 2),
                    round(profit_margin, 2),
                    round(dividend * 100, 2),
                    pe,
                    debt_equity,
                    face_value,
                    book_value,
                    high_52,
                    low_52
                ])

        except:
            continue

    # Clear old data
    for row in tree.get_children():
        tree.delete(row)

    if not results:
        messagebox.showinfo("Result", "No stocks matched your filters.")
    else:
        for row in results:
            tree.insert("", tk.END, values=row)

# -----------------------
# GUI Layout
# -----------------------
root = tk.Tk()
root.title("Stock Fundamental Screener Dashboard")
root.geometry("1200x500")

title = tk.Label(root, text="Stock Fundamental Screener", font=("Arial", 18))
title.pack(pady=10)

columns = (
    "Symbol", "Promoter %", "MarketCap (Cr)", "Profit %",
    "Dividend %", "PE", "Debt/Equity",
    "Face Value", "Book Value",
    "52W High", "52W Low"
)

tree = ttk.Treeview(root, columns=columns, show="headings")

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=100)

tree.pack(fill=tk.BOTH, expand=True)

btn = tk.Button(root, text="Run Screener", command=run_screener,
                font=("Arial", 12), bg="green", fg="white")
btn.pack(pady=10)

root.mainloop()