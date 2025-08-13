import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback
from streamlit.components.v1 import html as st_html

# ==========================
# App Config
# ==========================
st.set_page_config(page_title="S&P 500 Top-10 Screener", layout="wide")
st.title("📈 S&P 500 — Top-10 Trade Finder (10% target)")
st.caption("Educational use only. Not financial advice. Data via yfinance; may be delayed.")

# --- Fallback list of S&P 500 tickers ---
SP500_FALLBACK_TICKERS = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','GOOG','TSLA','META','BRK-B','UNH','JPM','JNJ','V','XOM','WMT','PG','LLY','MA','HD','CVX','MRK','PEP','AVGO','KO','BAC','ORCL','COST','MCD','TMO','PFE','ACN','CSCO','ABBV','CRM','DHR','ADBE','WFC','NKE','DIS','TXN','NEE','PM','AMD','LIN','UPS','RTX','MS','BMY','HON','INTC'
]

# ==========================
# Utilities
# ==========================
@st.cache_data(ttl=3600)
def get_sp500_tickers() -> list:
    """Try to scrape S&P 500 tickers. Fallback to hardcoded list on failure."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception:
        st.warning("Could not fetch live S&P 500 list. Using fallback list.")
        return SP500_FALLBACK_TICKERS

@st.cache_data
def download_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    return df

@st.cache_data
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators used by the strategy."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    close = df['Close']
    df['SMA200'] = close.rolling(window=200).mean()
    df['EMA20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = (gain / loss).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA_cross_up'] = (df['EMA20'].shift(1) < df['EMA50'].shift(1)) & (df['EMA20'] > df['EMA50'])
    df['EMA_cross_dn'] = (df['EMA20'].shift(1) > df['EMA50'].shift(1)) & (df['EMA20'] < df['EMA50'])
    return df.dropna()

@st.cache_data
def backtest_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Backtest: Buy when price>SMA200 & EMA20 crosses above EMA50 & RSI<70. Sell at +10% or cross-down."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['buy_date','sell_date','buy_price','sell_price'])
    trades = []
    in_pos = False
    buy_px = None
    buy_dt = None
    for i in range(len(df)):
        row = df.iloc[i]
        if in_pos:
            take_profit = buy_px * 1.10
            if row['Close'] >= take_profit or row['EMA_cross_dn']:
                trades.append({
                    'buy_date': buy_dt,
                    'sell_date': df.index[i],
                    'buy_price': float(buy_px),
                    'sell_price': float(row['Close'])
                })
                in_pos = False
        else:
            if row['Close'] > row['SMA200'] and row['EMA_cross_up'] and row['RSI'] < 70:
                in_pos = True
                buy_px = float(row['Close'])
                buy_dt = df.index[i]
    return pd.DataFrame(trades)

# ==========================
# TradingView Embed
# ==========================
def tradingview_widget(symbol: str, height: int = 610):
    """Embed TradingView Advanced Chart widget."""
    widget = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_{symbol}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "60",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "hide_volume": false,
          "container_id": "tradingview_{symbol}"
      }});
      </script>
    </div>
    <style>.tradingview-widget-container {{ height: {height}px; }}</style>
    """
    st_html(widget, height=height)

# ==========================
# Helper to score setup
# ==========================
@st.cache_data
def evaluate_current_setup(df: pd.DataFrame) -> dict:
    """Return dict with entry, buy, TP, stop."""
    if df is None or df.empty:
        return {"entry": False}
    last = df.iloc[-1]
    entry = (last['Close'] > last['SMA200']) and (df['EMA_cross_up'].iloc[-1]) and (last['RSI'] < 70)
    buy = float(last['Close'])
    target = round(buy * 1.10, 2)
    stop = round(buy * 0.95, 2)
    return {
        "entry": bool(entry),
        "buy": round(buy, 2),
        "target": target,
        "stop": stop
    }

# ==========================
# Screener
# ==========================
st.header("🔍 Screener — Find the 10 best candidates now")
colA, colB, colC = st.columns([1,1,1])
with colA:
    min_trades = st.number_input("Minimum trades in backtest (last 5y)", 0, 50, 3)
with colB:
    require_entry_now = st.checkbox("Require entry signal right now", value=True)
with colC:
    universe_size = st.slider("Universe sample size", 50, 500, 150, 25)

if st.button("🚀 Scan S&P 500"):
    tickers = get_sp500_tickers()[:universe_size]
    results = []
    prog = st.progress(0.0)
    status = st.empty()

    for i, t in enumerate(tickers):
        try:
            status.text(f"Fetching {t} ({i+1}/{len(tickers)})")
            hist = download_history(t, period="5y")
            if hist is None or hist.empty or len(hist) < 220:
                continue
            df = calculate_indicators(hist)
            trades = backtest_strategy(df)
            if len(trades) < min_trades:
                continue
            trades['profit'] = (trades['sell_price'] - trades['buy_price']) / trades['buy_price']
            success_rate = float((trades['profit'] > 0).mean() * 100.0)
            setup = evaluate_current_setup(df)
            if require_entry_now and not setup['entry']:
                pass
            else:
                results.append({
                    'Ticker': t,
                    'Last Price': round(float(df['Close'].iloc[-1]), 2),
                    'Success Rate % (5y)': round(success_rate, 2),
                    'Entry Now?': setup['entry'],
                    'Buy Price': setup['buy'],
                    'TP (+10%)': setup['target'],
                    'Stop (-5%)': setup['stop'],
                    'Trades Backtested': int(len(trades))
                })
        except Exception:
            pass
        finally:
            prog.progress((i+1)/len(tickers))

    status.text("Scan complete.")
    if results:
        df_screen = pd.DataFrame(results)
        df_screen['RankKey'] = (~df_screen['Entry Now?']).astype(int)
        df_screen = df_screen.sort_values(['RankKey','Success Rate % (5y)'], ascending=[True, False]).drop(columns='RankKey').head(10)
        st.subheader("🏆 Top 10 candidates")
        st.dataframe(df_screen, use_container_width=True)
        st.info("Buy price = last close; TP = +10%; Stop = -5%.")
    else:
        st.warning("No matches found.")

# ==========================
# Manual Stock Check
# ==========================
st.header("🔎 Manual Stock Check + Chart")
col1, col2 = st.columns([1,1])
with col1:
    symbol = st.text_input("Ticker (e.g., AAPL, MSFT)", "AAPL").upper().strip()
    if st.button("Analyze Ticker"):
        try:
            data = download_history(symbol, period="5y")
            if data is None or data.empty:
                st.error("No data returned.")
            else:
                df = calculate_indicators(data)
                trades = backtest_strategy(df)
                trades['profit'] = (trades['sell_price'] - trades['buy_price']) / trades['buy_price'] if len(trades) else []
                success_rate = (trades['profit'] > 0).mean() * 100 if len(trades) else 0

                st.subheader(f"Backtest — {symbol}")
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Trades", len(trades))
                with c2: st.metric("Success Rate", f"{success_rate:.2f}%")
                setup = evaluate_current_setup(df)
                with c3: st.metric("Entry Now?", "Yes" if setup['entry'] else "No")
                with c4: st.metric("Buy→TP", f"{setup['buy']} → {setup['target']}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', mode='lines'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', mode='lines'))
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA50', mode='lines'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name='SMA200', mode='lines'))
                fig.add_hline(y=setup['buy'], line_dash="dot", annotation_text="Buy")
                fig.add_hline(y=setup['target'], line_dash="dot", annotation_text="TP +10%")
                fig.add_hline(y=setup['stop'], line_dash="dot", annotation_text="Stop -5%")
                fig.update_layout(height=420, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("Unexpected error in analysis:")
            st.code(f"{e}\n\n{traceback.format_exc()}")

with col2:
    st.subheader("TradingView Chart")
    st.caption("Interactive chart (1h interval).")
    tv_symbol = st.text_input("TradingView symbol", "AAPL").upper().strip()
    tradingview_widget(tv_symbol, height=610)

# ==========================
# Footer
# ==========================
st.info("Strategy: Buy when price>SMA200 and EMA20 crosses above EMA50 with RSI<70. Exit at +10% or EMA20 cross below EMA50.")

