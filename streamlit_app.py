import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="S&P 500 Stock Scanner", layout="wide")

# =========================
# Helper functions
# =========================
@st.cache_data
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

def fetch_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df["RSI14"] = compute_rsi(df["Close"], 14)
        df["SMA20"] = df["Close"].rolling(20).mean()
        return df
    except Exception:
        return None

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_candidates(symbols, period, interval, target_gain):
    results = []
    for sym in symbols:
        df = fetch_data(sym, period, interval)
        if df is None or df.empty:
            continue
        change_pct = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
        if change_pct >= target_gain:
            results.append({
                "Symbol": sym,
                "Change %": round(change_pct, 2),
                "Last Price": round(df["Close"].iloc[-1], 2),
                "RSI14": round(df["RSI14"].iloc[-1], 2),
                "SMA20": round(df["SMA20"].iloc[-1], 2)
            })
    return pd.DataFrame(results).sort_values(by="Change %", ascending=False)

# =========================
# App UI
# =========================
st.title("📈 S&P 500 Live Stock Scanner")

timeframes = {
    "1 min": ("1d", "1m"),
    "5 min": ("5d", "5m"),
    "15 min": ("5d", "15m"),
    "30 min": ("1mo", "30m"),
    "1 hour": ("3mo", "1h"),
    "1 day": ("1y", "1d"),
    "1 week": ("5y", "1wk"),
    "1 month": ("10y", "1mo"),
    "3 months": ("10y", "3mo"),
    "6 months": ("10y", "6mo"),
}

col1, col2, col3 = st.columns(3)
with col1:
    tf_choice = st.selectbox("⏱ Select Timeframe", list(timeframes.keys()), index=5)
with col2:
    target_gain = st.number_input("🎯 Target Gain %", 0.1, 50.0, 10.0)
with col3:
    max_symbols = st.slider("📊 Max Symbols to Scan", 10, 500, 50)

if st.button("Run S&P 500 Scan"):
    st.info("🔍 Scanning S&P 500... please wait.")
    symbols = get_sp500_symbols()[:max_symbols]
    period, interval = timeframes[tf_choice]
    df_results = find_candidates(symbols, period, interval, target_gain)
    
    if df_results.empty:
        st.warning("⚠ No candidates found. Try increasing Max Symbols, lowering Target Gain, or changing timeframe.")
    else:
        st.success(f"✅ Found {len(df_results)} candidates")
        st.dataframe(df_results, use_container_width=True)
        
        selected_symbol = st.selectbox("📌 View chart for:", df_results["Symbol"])
        data = fetch_data(selected_symbol, period, interval)
        if data is not None:
            st.line_chart(data["Close"])
            st.metric("Last Price", f"${data['Close'].iloc[-1]:.2f}")
            st.metric("RSI(14)", f"{data['RSI14'].iloc[-1]:.1f}")
            st.metric("SMA(20)", f"{data['SMA20'].iloc[-1]:.2f}")




