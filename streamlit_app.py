import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from urllib.parse import quote

# -----------------------------
# SETTINGS
# -----------------------------
st.set_page_config(page_title="Top-10 S&P 500 Screener", layout="wide")
TAKE_PROFIT_PCT = 0.10  # 10% target
STOP_LOSS_PCT = -0.05   # -5% stop (optional)

# -----------------------------
# HEADER
# -----------------------------
st.title("📈 Top-10 S&P 500 Screener with TradingView Charts")
st.markdown("""
This app screens S&P 500 stocks for potential buy setups, applies a **10% take-profit target**, 
and ranks them by success rate over 5 years. It also gives you a TradingView chart for any stock.
""")

# -----------------------------
# LOAD S&P 500 LIST
# -----------------------------
@st.cache_data
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    df = table[0]
    return df['Symbol'].tolist()

sp500_symbols = get_sp500_symbols()

# -----------------------------
# FETCH DATA & CALCULATE SIGNALS
# -----------------------------
def analyze_stock(symbol):
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if df.empty:
            return None

        df['Return'] = df['Close'].pct_change()
        buy_points = []
        success_count = 0

        # Simple example: buy if yesterday's close > 50-day SMA
        df['SMA50'] = df['Close'].rolling(50).mean()
        for i in range(50, len(df)):
            if df['Close'].iloc[i-1] > df['SMA50'].iloc[i-1]:
                entry_price = df['Close'].iloc[i]
                take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)

                future_prices = df['Close'].iloc[i+1:i+21]  # next ~1 month
                if any(future_prices >= take_profit_price):
                    success_count += 1
                buy_points.append(i)

        success_rate = (success_count / len(buy_points) * 100) if buy_points else 0
        last_close = df['Close'].iloc[-1]
        signal_now = last_close > df['SMA50'].iloc[-1]

        return {
            'Symbol': symbol,
            'Last Close': round(last_close, 2),
            'Success Rate %': round(success_rate, 1),
            'Signal Now': signal_now,
            'Buy Target': round(last_close, 2),
            'Take Profit': round(last_close * (1 + TAKE_PROFIT_PCT), 2),
            'Stop Loss': round(last_close * (1 + STOP_LOSS_PCT), 2)
        }
    except:
        return None

# -----------------------------
# SCREENING
# -----------------------------
max_stocks = st.slider("Number of stocks to scan", 10, len(sp500_symbols), 50, 10)
progress_bar = st.progress(0)
results = []

for idx, symbol in enumerate(sp500_symbols[:max_stocks]):
    stock_data = analyze_stock(symbol)
    if stock_data:
        results.append(stock_data)
    progress_bar.progress((idx+1) / max_stocks)

df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results = df_results.sort_values(
        by=['Signal Now', 'Success Rate %'],
        ascending=[False, False]
    )
    st.subheader("📊 Top 10 Ranked Stocks")
    st.dataframe(df_results.head(10), use_container_width=True)
else:
    st.warning("No stocks found with matching criteria.")

# -----------------------------
# MANUAL CHECK & TRADINGVIEW
# -----------------------------
st.subheader("🔍 Manual Stock Check")
manual_symbol = st.text_input("Enter a stock ticker (e.g., AAPL):").upper()
if manual_symbol:
    stock_info = analyze_stock(manual_symbol)
    if stock_info:
        st.write(stock_info)
        tv_symbol = f"NASDAQ:{manual_symbol}" if manual_symbol not in ["SPY"] else manual_symbol
        tv_embed = f"https://www.tradingview.com/widgetembed/?symbol={quote(tv_symbol)}&interval=D&hidesidetoolbar=1&hidetoptoolbar=1&symboledit=1"
        st.markdown(f'<iframe src="{tv_embed}" width="100%" height="500"></iframe>', unsafe_allow_html=True)
    else:
        st.error("No data found for this ticker.")



