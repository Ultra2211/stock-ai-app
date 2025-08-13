# streamlit_app.py
# S&P 500 Screener + TradingView chart with timeframe & range selectors
# Educational use only — not financial advice.

import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="S&P 500 AI Screener", page_icon="📈", layout="wide")
st.title("📈 S&P 500 AI Screener with TradingView Charts")
st.caption("Ranks top candidates using RSI/EMA/MACD and shows fully interactive TradingView charts. Not financial advice.")

# -----------------------------
# Helpers
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def fmt(x, digits=1, default="—"):
    try:
        if pd.isna(x):
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_sp500_symbols():
    # yfinance has this helper; fall back to a core set if needed.
    try:
        syms = yf.tickers_sp500()
        return sorted(list(set(syms)))
    except Exception:
        return sorted([
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
            "V","UNH","XOM","JNJ","PG","LLY","HD","MA","COST","BAC"
        ])

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    macd_line, sig, hist = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = sig
    df["MACD_HIST"] = hist
    return df

def score_row(row):
    # Simple, quick technical score (0–3)
    score = 0
    if safe_float(row.get("EMA20")) > safe_float(row.get("EMA50")): score += 1
    if 30 < safe_float(row.get("RSI14")) < 70: score += 1
    if safe_float(row.get("MACD")) > safe_float(row.get("MACD_SIGNAL")): score += 1
    return score

@st.cache_data(ttl=20*60, show_spinner=False)
def get_history(symbol: str, period="6mo", interval="1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna().copy()

def tradingview_iframe(symbol: str, interval_code: str, range_code: str, theme: str, height: int = 620):
    """
    Embed TradingView chart with chosen timeframe (interval) and visible range.
    interval_code examples: 1,5,15,30,60,240,D,W,M
    range_code examples: 1D,5D,1M,3M,6M,1Y,5Y,ALL
    """
    # TradingView symbols often work fine with raw ticker (AAPL).
    # If you hit mismatches, consider prefixed forms like NASDAQ:AAPL / NYSE:MMM.
    src = (
        "https://s.tradingview.com/widgetembed/?"
        f"symbol={symbol}"
        f"&interval={interval_code}"
        f"&range={range_code}"
        "&hidesidetoolbar=0"
        "&hidetoptoolbar=0"
        "&symboledit=1"
        "&saveimage=1"
        "&toolbarbg=f1f3f6"
        f"&theme={'dark' if theme=='Dark' else 'light'}"
        "&style=1"
        "&timezone=Etc/UTC"
        "&studies=[]"
        "&hide_legend=0"
        "&withdateranges=1"
        "&allow_symbol_change=1"
        "&details=1"
        "&hotlist=0"
        "&calendar=0"
        "&news=0"
        "&hideideas=1"
    )
    components.html(
        f'<iframe src="{src}" width="100%" height="{height}" frameborder="0" allowtransparency="true" scrolling="no"></iframe>',
        height=height, scrolling=False
    )

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Screener Settings")
    target_gain = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5)
    stop_loss = st.number_input("Stop suggestion (%)", 1.0, 30.0, 5.0, 0.5)
    max_scan = st.slider("Max symbols to scan", 50, 505, 300, 25)
    history_period = st.selectbox("History period for indicators", ["6mo", "1y", "2y"])
    run_scan = st.button("🚀 Run S&P 500 Scan")

    st.markdown("---")
    st.header("🕒 Chart Controls")
    timeframe = st.selectbox(
        "Timeframe",
        ["1m","5m","15m","30m","1h","1D","1W","1M"],
        index=5  # default 1D
    )
    # Map to TradingView interval code
    tf_map = {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "1D": "D", "1W": "W", "1M": "M"
    }
    interval_code = tf_map[timeframe]

    range_code = st.selectbox(
        "Visible range",
        ["1D","5D","1M","3M","6M","1Y","5Y","ALL"],
        index=4  # default 6M
    )
    theme = st.radio("Chart theme", ["Light", "Dark"], index=0, horizontal=True)

    st.markdown("---")
    manual_ticker = st.text_input("🔎 Manual ticker lookup", value="AAPL").strip().upper()

# -----------------------------
# Manual chart (always shown)
# -----------------------------
st.subheader("🔎 Manual Chart")
if manual_ticker:
    cols = st.columns([1, 1.2])
    with cols[0]:
        st.write(f"**{manual_ticker}** — timeframe **{timeframe}**, range **{range_code}**")
        tradingview_iframe(manual_ticker, interval_code, range_code, theme, height=560)
    with cols[1]:
        # Show quick stats from Yahoo Finance (daily data)
        df_m = get_history(manual_ticker, period=history_period, interval="1d")
        if df_m.empty:
            st.warning("No Yahoo data for this symbol.")
        else:
            df_m = compute_indicators(df_m)
            last = df_m.iloc[-1]
            price = safe_float(last.get("Close"))
            ema20 = safe_float(last.get("EMA20"))
            ema50 = safe_float(last.get("EMA50"))
            ema200 = safe_float(last.get("EMA200"))
            rsi14 = safe_float(last.get("RSI14"))
            macd_val = safe_float(last.get("MACD"))
            macd_sig = safe_float(last.get("MACD_SIGNAL"))

            st.metric("Price", f"${fmt(price,2)}")
            st.metric("RSI(14)", fmt(rsi14, 1))
            st.metric("EMA20 vs EMA50", "Bullish ✅" if ema20 > ema50 else "Bearish ❌")
            st.metric("EMA50 vs EMA200", "Bullish ✅" if ema50 > ema200 else "Bearish ❌")
            st.metric("MACD - Signal", fmt(macd_val - macd_sig, 3))

            tgt = price * (1 + target_gain/100.0)
            stp = price * (1 - stop_loss/100.0)
            st.write("---")
            st.write("**Simple trade idea (illustrative):**")
            st.write(f"- **Buy (now):** ~ ${fmt(price,2)}")
            st.write(f"- **Target (+{target_gain:.1f}%):** ~ ${fmt(tgt,2)}")
            st.write(f"- **Stop (−{stop_loss:.1f}%):** ~ ${fmt(stp,2)}")

st.markdown("---")

# -----------------------------
# Screener
# -----------------------------
st.subheader("🏆 Top 10 Screener (S&P 500)")
if run_scan:
    syms = get_sp500_symbols()
    if len(syms) > max_scan:
        syms = syms[:max_scan]

    progress = st.progress(0.0, text="Downloading & analyzing…")
    rows = []

    for i, sym in enumerate(syms, start=1):
        try:
            df = get_history(sym, period=history_period, interval="1d")
            if df.empty or len(df) < 60:
                continue
            df = compute_indicators(df)
            last = df.iloc[-1]
            close = safe_float(last.get("Close"))
            if pd.isna(close):
                continue
            s = score_row(last)
            tgt = round(close * 1.10, 2)  # 10% default target for table
            stp = round(close * 0.95, 2)  # 5% default stop for table

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "RSI14": round(safe_float(last.get("RSI14")), 1) if not pd.isna(last.get("RSI14")) else np.nan,
                "EMA20>EMA50": (safe_float(last.get("EMA20")) > safe_float(last.get("EMA50"))),
                "MACD>Signal": (safe_float(last.get("MACD")) > safe_float(last.get("MACD_SIGNAL"))),
                "Score(0-3)": s,
                "Buy": round(close, 2),
                "Target(+10%)": tgt,
                "Stop(-5%)": stp
            })
        except Exception:
            pass
        progress.progress(min(i/len(syms), 1.0))

    progress.empty()

    if not rows:
        st.warning("No candidates found (try increasing Max symbols or changing history period).")
    else:
        dfres = pd.DataFrame(rows).sort_values(["Score(0-3)","RSI14"], ascending=[False, True]).head(10).reset_index(drop=True)
        st.dataframe(dfres, use_container_width=True)

        # Select a symbol to view with the chosen timeframe/range
        pick = st.selectbox("📊 View chart for:", dfres["Symbol"], index=0)
        tradingview_iframe(pick, interval_code, range_code, theme, height=560)

# -----------------------------
# Footer
# -----------------------------
st.caption("Data: Yahoo Finance via yfinance. Charts: TradingView widget © TradingView.")







