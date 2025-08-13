import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# ---------- CONFIG ----------
st.set_page_config(page_title="S&P 500 AI Screener", layout="wide")

# ---------- FUNCTIONS ----------
@st.cache_data
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df["Symbol"].tolist()

def calculate_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    return df

def score_stock(df):
    latest = df.iloc[-1]
    score = 0
    if latest["EMA20"] > latest["EMA50"]:
        score += 1
    if latest["RSI14"] < 70 and latest["RSI14"] > 30:
        score += 1
    if latest["MACD"] > latest["Signal"]:
        score += 1
    return score

def tradingview_chart(ticker):
    st.markdown(f"""
    <iframe src="https://s.tradingview.com/widgetembed/?symbol={ticker}&interval=1&hidesidetoolbar=0&hidetoptoolbar=0&theme=light&style=1&locale=en&timezone=Etc/UTC" 
    width="100%" height="600" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """, unsafe_allow_html=True)

# ---------- MAIN APP ----------
st.title("📈 S&P 500 AI Stock Screener with TradingView Charts")

tab1, tab2 = st.tabs(["Top 10 Picks", "Manual Search"])

# --- TOP 10 PICKS ---
with tab1:
    st.subheader("🔍 Scanning S&P 500... please wait ~1 min")

    symbols = get_sp500_symbols()
    results = []

    for symbol in symbols:
        try:
            df = yf.download(symbol, period="6mo", interval="1d", progress=False)
            if df.empty:
                continue
            df = calculate_indicators(df)
            score = score_stock(df)
            if score >= 2:
                price = df["Close"].iloc[-1]
                buy_price = price
                sell_price = price * 1.10
                results.append([symbol, price, buy_price, sell_price, score])
        except Exception:
            continue

    results_df = pd.DataFrame(results, columns=["Symbol", "Last Price", "Buy Price", "Sell Price (10%)", "Score"])
    results_df = results_df.sort_values(by="Score", ascending=False).head(10)

    st.dataframe(results_df, hide_index=True)

    selected = st.selectbox("📊 View TradingView Chart for:", results_df["Symbol"])
    tradingview_chart(selected)

# --- MANUAL SEARCH ---
with tab2:
    search_symbol = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, TSLA)")
    if search_symbol:
        tradingview_chart(search_symbol.upper())







