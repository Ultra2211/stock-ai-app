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

# --- Indicators -- manual implementations

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
    # Convert input to pandas Series if not already
    series = pd.Series(series)
    
    if len(series) <
