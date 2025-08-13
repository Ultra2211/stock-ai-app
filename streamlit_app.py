# streamlit_app.py
# S&P 500 Screener with TradingView-like charts, indicators, and 10% target hit-rate
# Educational use only. Not financial advice.

import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="S&P 500 10% Target Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 S&P 500 — 10% Target Screener")
st.caption("Screens S&P 500 for top 10 setups using RSI, MACD, EMA trend, ATR, and a historical hit-rate for your target. Not financial advice.")

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

@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_sp500_tickers():
    try:
        tickers = yf.tickers_sp500()
        return sorted(list(set(tickers)))
    except Exception:
        return sorted([
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
            "V","UNH","XOM","JNJ","PG","LLY","HD","MA","COST","BAC"
        ])

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 14):
    tr = true_range(df)
    return tr.rolling(period).mean()

def label_for_tradingview(symbol: str) -> str:
    return symbol.replace("-", ".")

@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_history(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().copy()
    df["Symbol"] = symbol
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    macd_line, signal_line, hist = macd(df["Close"], 12, 26, 9)
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal_line
    df["MACD_HIST"] = hist
    df["ATR14"] = atr(df, 14)
    return df

def forward_hit_rate_for_target(df: pd.DataFrame, target_pct=0.10, horizon_days=10):
    if df.empty or "Close" not in df:
        return np.nan, np.nan
    closes = df["Close"].values
    n = len(closes)
    if n < horizon_days + 2:
        return np.nan, np.nan
    forward_max = np.full(n, np.nan)
    for i in range(n - horizon_days):
        forward_max[i] = np.max(closes[i+1:i+1+horizon_days])
    base = closes[:-horizon_days]
    fmax = forward_max[:-horizon_days]
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_ret = (fmax - base) / base
    mask = ~np.isnan(fwd_ret)
    if mask.sum() == 0:
        return np.nan, np.nan
    hit = (fwd_ret[mask] >= target_pct).sum()
    total = mask.sum()
    return (hit / total) * 100.0, total

def composite_score(row, target_hit_rate_weight=0.6):
    hr = row.get("HitRatePct", np.nan)
    hr_norm = np.nan if math.isnan(hr) else max(0.0, min(100.0, hr)) / 100.0
    points = 0.0
    points += 0.4 if row.get("Close", 0) > row.get("EMA200", np.inf) else 0.0
    points += 0.3 if row.get("EMA50", -np.inf) > row.get("EMA200", np.inf) else 0.0
    points += 0.2 if row.get("MACD", -np.inf) > row.get("MACD_SIGNAL", np.inf) else 0.0
    rsi_val = row.get("RSI14", np.nan)
    if not math.isnan(rsi_val) and 45 <= rsi_val <= 65:
        points += 0.1
    ind_norm = min(points, 1.0)
    if hr_norm is np.nan:
        return np.nan
    score = target_hit_rate_weight * hr_norm + (1 - target_hit_rate_weight) * ind_norm
    return round(100 * score, 2)

def tradingview_widget(symbol: str, height: int = 460):
    tv_symbol = label_for_tradingview(symbol)
    widget = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_symbol_overview"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
      {{
        "symbols": [
          ["{tv_symbol}|1D"]
        ],
        "chartType": "candlesticks",
        "width": "100%",
        "height": {height},
        "locale": "en",
        "autosize": true,
        "showVolume": true,
        "lineWidth": 2,
        "dateRanges": ["1M","3M","6M","12M"],
        "showMA": true,
        "maLength": 50,
        "maLength2": 200
      }}
      </script>
    </div>
    """
    components.html(widget, height=height + 20, scrolling=False)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Screener Settings")
    target_pct_input = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5)
    horizon_days = st.number_input("Forward horizon (trading days)", 3, 60, 10, 1)
    stop_pct_input = st.number_input("Stop suggestion (%)", 1.0, 30.0, 5.0, 0.5)
    max_symbols = st.slider("Max symbols to scan", 50, 505, 300, 25)
    data_period = st.selectbox("History period", ["1y","2y","5y"])
    manual_symbol = st.text_input("🔎 Manual ticker lookup", "AAPL").strip().upper()
    run_button = st.button("🚀 Run Screener")

# -----------------------------
# Manual Lookup
# -----------------------------
st.subheader("🔎 Manual Lookup")
if manual_symbol:
    df_manual = get_history(manual_symbol, period=data_period, interval="1d")
    if df_manual.empty:
        st.warning(f"No data for {manual_symbol}")
    else:
        df_manual = compute_indicators(df_manual)
        hr_pct, samples = forward_hit_rate_for_target(df_manual, target_pct_input/100.0, horizon_days)
        latest = df_manual.iloc[-1]
        price = safe_float(latest.get("Close"))
        atr_val = safe_float(latest.get("ATR14"))
        target_price = price * (1 + target_pct_input/100.0)
        stop_price = price * (1 - stop_pct_input/100.0)

        cols = st.columns([2, 1])
        with cols[0]:
            tradingview_widget(manual_symbol, height=440)
        with cols[1]:
            st.metric("Price", f"${price:,.2f}")
            st.metric("RSI(14)", fmt(latest.get("RSI14"), 1))
            macd_diff = safe_float(latest.get("MACD")) - safe_float(latest.get("MACD_SIGNAL"))
            st.metric("MACD - Signal", fmt(macd_diff, 3))
            ema50 = safe_float(latest.get("EMA50"))
            ema200 = safe_float(latest.get("EMA200"))
            st.metric("EMA50 vs EMA200", "Bullish ✅" if ema50 > ema200 else "Bearish ❌")
            st.metric("ATR(14)", fmt(atr_val, 2))
            st.metric(f"Hit-rate {target_pct_input:.1f}% in {horizon_days}d", f"{fmt(hr_pct, 1)}% (n={int(samples)})" if not pd.isna(hr_pct) else "—")
            st.write("---")
            st.write("**Trade Ideas (simple):**")
            st.write(f"- **Buy:** ~ ${price:,.2f}")
            st.write(f"- **Target:** ~ ${target_price:,.2f}")
            st.write(f"- **Stop:** ~ ${stop_price:,.2f}")

# -----------------------------
# Screener Run
# -----------------------------
if run_button:
    st.subheader("🏆 Top 10 Candidates")
    tickers = get_sp500_tickers()
    if len(tickers) > max_symbols:
        tickers = tickers[:max_symbols]
    progress = st.progress(0, text="Downloading & analyzing…")
    rows = []

    for i, sym in enumerate(tickers, start=1):
        try:
            df = get_history(sym, period=data_period, interval="1d")
            if df.empty:
                continue
            df = compute_indicators(df)
            hr_pct, samples = forward_hit_rate_for_target(df, target_pct_input/100.0, horizon_days)
            if math.isnan(hr_pct):
                continue
            latest = df.iloc[-1]
            close = safe_float(latest.get("Close"))
            atr_val = safe_float(latest.get("ATR14"))
            rsi14 = safe_float(latest.get("RSI14"))
            macd_val = safe_float(latest.get("MACD"))
            macd_sig = safe_float(latest.get("MACD_SIGNAL"))
            ema50 = safe_float(latest.get("EMA50"))
            ema200 = safe_float(latest.get("EMA200"))
            if pd.isna(close):
                continue
            target_price = close * (1 + target_pct_input / 100.0)
            stop_price   = close * (1 - stop_pct_input / 100.0)
            rr = (target_price - close) / max(close - stop_price, 1e-6)
            row = {
                "Symbol": sym,
                "Close": close,
                "RSI14": rsi14,
                "MACD": macd_val,
                "MACD_SIGNAL": macd_sig,
                "EMA50": ema50,
                "EMA200": ema200,
                "ATR14": atr_val,
                "HitRatePct": hr_pct,
                "Samples": int(samples),
                "TargetPrice": round(target_price, 2),
                "StopPrice": round(stop_price, 2),
                "RR": round(rr, 2),
            }
            row["Score"] = composite_score(row)
            rows.append(row)
        except Exception:
            pass
        progress.progress(min(i/len(tickers), 1.0))
    progress.empty()

    if not rows:
        st.error("No candidates found.")
    else:
        df_all = pd.DataFrame(rows).dropna(subset=["Score"]).sort_values("Score", ascending=False)
        top10 = df_all.head(10).reset_index(drop=True)
        st.dataframe(top10, use_container_width=True)
        csv = top10.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Top 10 (CSV)", csv, "sp500_top10.csv", "text/csv")
        st.write("---")
        for _, r in top10.iterrows():
            st.markdown(f"### {r['Symbol']} | Score {fmt(r['Score'],1)} | Hit-rate {fmt(r['HitRatePct'],1)}% (n={int(r['Samples'])})")
            cols = st.columns([2, 1])
            with cols[0]:
                tradingview_widget(r["Symbol"], height=420)
            with cols[1]:
                st.metric("Price", f"${r['Close']:,.2f}")
                st.metric("RSI(14)", fmt(r["RSI14"], 1))
                st.metric("EMA50>EMA200", "Yes ✅" if safe_float(r["EMA50"]) > safe_float(r["EMA200"]) else "No ❌")
                st.metric("MACD-Signal", fmt(safe_float(r["MACD"]) - safe_float(r["MACD_SIGNAL"]), 3))
                st.metric("ATR(14)", fmt(r["ATR14"], 2))
                st.write("**Trade Ideas:**")
                st.write(f"- Buy: ~ ${r['Close']:,.2f}")
                st.write(f"- Target: ~ ${r['TargetPrice']:,.2f}")
                st.write(f"- Stop: ~ ${r['StopPrice']:,.2f}")
            st.write("---")

st.caption("Data source: Yahoo Finance. TradingView widget © TradingView.")






