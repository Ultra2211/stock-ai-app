import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

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

def sma(series, length):
    return series.rolling(window=length).mean()

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    elif isinstance(series, np.ndarray) and series.ndim > 1:
        series = series.flatten()
    series = pd.Series(series)

    if len(series) < slow:
        raise ValueError(f"Input series length must be at least {slow} for MACD calculation")

    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_signal": signal_line,
        "MACD_hist": histogram
    }, index=series.index)

# Calculate indicators
df["SMA20"] = sma(df["Close"], 20)
df["EMA50"] = ema(df["Close"], 50)
df["RSI14"] = rsi(df["Close"], 14)
macd_df = macd(df["Close"])

# Reset index to columns
df_reset = df.reset_index()
macd_reset = macd_df.reset_index()

# DEBUG: Print column info before merge
st.write("df_reset columns:", df_reset.columns.tolist())
st.write("macd_reset columns:", macd_reset.columns.tolist())

# DEBUG: Check if 'Date' column exists and types
st.write("df_reset 'Date' dtype:", df_reset['Date'].dtype)
st.write("macd_reset 'Date' dtype:", macd_reset['Date'].dtype)

# DEBUG: Check for duplicates in 'Date' columns
st.write("df_reset 'Date' duplicates:", df_reset['Date'].duplicated().sum())
st.write("macd_reset 'Date' duplicates:", macd_reset['Date'].duplicated().sum())

# Attempt merge on 'Date'
df_joined = pd.merge(df_reset, macd_reset, on="Date", how="left")

# Restore Date index
df_joined.set_index("Date", inplace=True)
df = df_joined

# --- Display
st.title(f"{ticker} — Technical Indicators")
st.line_chart(df[["Close", "SMA20", "EMA50"]])
st.line_chart(df[["RSI14"]])
st.line_chart(df[["MACD", "MACD_signal"]])
st.write(df.tail(10))
