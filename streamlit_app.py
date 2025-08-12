import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta

st.set_page_config(page_title="Market Indicators App", layout="wide")

# --- Sidebar
ticker = st.sidebar.text_input("Ticker", "AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m"], index=0)

# --- Fetch Data
@st.cache_data(ttl=300)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df.dropna()

df = load_data(ticker, period, interval)

if df.empty:
    st.error("No data found.")
    st.stop()

# --- Indicators
df["SMA20"] = ta.sma(df["Close"], length=20)
df["EMA50"] = ta.ema(df["Close"], length=50)
df["RSI14"] = ta.rsi(df["Close"], length=14)
macd = ta.macd(df["Close"])
df["MACD"] = macd["MACD_12_26_9"]
df["MACD_signal"] = macd["MACDs_12_26_9"]

# --- Display
st.title(f"{ticker} — Technical Indicators")
st.line_chart(df[["Close", "SMA20", "EMA50"]])
st.write(df.tail(10))
