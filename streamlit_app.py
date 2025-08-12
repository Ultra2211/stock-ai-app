import streamlit as st
import pandas as pd
import yfinance as yf
import streamlit.components.v1 as components

st.set_page_config(page_title="Market Indicators + TradingView", layout="wide")

# --- Sidebar inputs
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m"], index=0)

# --- Load data
@st.cache_data(ttl=300)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df.dropna()

df = load_data(ticker, period, interval)

if df.empty:
    st.error("No data found.")
    st.stop()

# --- Flatten MultiIndex columns if exist
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]

# --- Compute indicators
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

# --- Plot technical indicators
st.title(f"{ticker} — Technical Indicators")
st.line_chart(df[["Close", "SMA20", "EMA50"]])

# --- Embed TradingView widget
tradingview_html = f"""
<div class="tradingview-widget-container">
  <div id="tradingview_{ticker}"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "width": "100%",
    "height": 500,
    "symbol": "{ticker}",
    "interval": "D",
    "timezone": "Etc/UTC",
    "theme": "light",
    "style": "1",
    "locale": "en",
    "toolbar_bg": "#f1f3f6",
    "enable_publishing": false,
    "allow_symbol_change": true,
    "container_id": "tradingview_{ticker}"
  }});
  </script>
</div>
"""

components.html(tradingview_html, height=520, scrolling=True)


