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
# Utilities
# -----------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_sp500_tickers():
    """
    yfinance provides a helper for S&P500 tickers. Fallback to a small core set if needed.
    """
    try:
        tickers = yf.tickers_sp500()
        # Some symbols include dots (e.g. BRK.B). yfinance expects '-' instead of '.' sometimes.
        # We'll keep original and also try replacing '.' with '-'.
        return sorted(list(set(tickers)))
    except Exception:
        # Fallback mini-universe to keep app functional
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
    """
    TradingView widgets can often resolve symbols without the exchange,
    but we try a couple of fallbacks.
    """
    # If it's like BRK-B, also provide BRK.B as a variant
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
    """
    For each day t, check if within the next `horizon_days`, price reaches +target_pct vs close[t].
    Simple approximation: compute max forward return over horizon and test if >= target_pct.
    Returns: % of days (over last ~252 trading days) where this condition held.
    """
    if df.empty or "Close" not in df:
        return np.nan, np.nan

    closes = df["Close"].values
    n = len(closes)
    if n < horizon_days + 2:
        return np.nan, np.nan

    # Compute forward max over a rolling window
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
    """
    Score combines hit-rate and indicator alignment.
    - Hit rate weight dominates (default 60%)
    - Indicator score (40%): EMA trend, MACD>Signal, RSI in 45-65 sweet spot
    """
    hr = row.get("HitRatePct", np.nan)
    # Normalize hit-rate to 0-1 (cap 0..100)
    hr_norm = np.nan if math.isnan(hr) else max(0.0, min(100.0, hr)) / 100.0

    # Indicator points: up to 1.0
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
    return round(100 * score, 2)  # scale to 0..100

def tradingview_widget(symbol: str, height: int = 460):
    """
    Embeds TradingView Symbol Overview widget via an <iframe>-like component.
    """
    tv_symbol = label_for_tradingview(symbol)
    widget = f"""
    <!-- TradingView Widget BEGIN -->
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
    <!-- TradingView Widget END -->
    """
    components.html(widget, height=height + 20, scrolling=False)

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("⚙️ Screener Settings")
    default_universe = "S&P 500 (yfinance)"
    universe_choice = st.selectbox("Universe", [default_universe])

    target_pct_input = st.number_input("Target gain (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
    horizon_days = st.number_input("Forward horizon (trading days)", min_value=3, max_value=60, value=10, step=1)
    stop_pct_input = st.number_input("Stop suggestion (%)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)

    max_symbols = st.slider("Max symbols to scan (for speed)", min_value=50, max_value=505, value=300, step=25)
    data_period = st.selectbox("History period", ["1y","2y","5y"])
    interval = "1d"

    st.markdown("---")
    manual_symbol = st.text_input("🔎 Manual ticker lookup", value="AAPL").strip().upper()
    run_button = st.button("🚀 Run Screener")

# -----------------------------
# Manual lookup section
# -----------------------------
st.subheader("🔎 Manual Lookup")
if manual_symbol:
    df_manual = get_history(manual_symbol, period=data_period, interval=interval)
    if df_manual.empty:
        st.warning(f"No data for {manual_symbol}. Try another symbol.")
    else:
        df_manual = compute_indicators(df_manual)
        hr_pct, samples = forward_hit_rate_for_target(df_manual, target_pct_input/100.0, horizon_days)
        latest = df_manual.iloc[-1]
        price = float(latest["Close"])
        atr_val = float(latest["ATR14"]) if not math.isnan(latest["ATR14"]) else np.nan
        target_price = round(price * (1 + target_pct_input/100.0), 2)
        stop_price = round(price * (1 - stop_pct_input/100.0), 2)

        cols = st.columns([2, 1])
        with cols[0]:
            tradingview_widget(manual_symbol, height=440)
        with cols[1]:
            st.metric("Price", f"${price:,.2f}")
            st.metric("RSI(14)", f"{latest['RSI14']:.1f}")
            st.metric("MACD - Signal", f"{(latest['MACD']-latest['MACD_SIGNAL']):.3f}")
            st.metric("EMA50 vs EMA200", "Bullish ✅" if latest["EMA50"] > latest["EMA200"] else "Bearish ❌")
            st.metric("ATR(14)", f"{atr_val:.2f}" if not math.isnan(atr_val) else "—")
            st.metric(f"Hit-rate {target_pct_input:.1f}% in {horizon_days}d", f"{hr_pct:.1f}% (n={int(samples)})" if not math.isnan(hr_pct) else "—")

            st.write("---")
            st.write("**Trade Ideas (simple):**")
            st.write(f"- **Buy (now):** ~ ${price:,.2f}")
            st.write(f"- **Target (+{target_pct_input:.1f}%):** ~ ${target_price:,.2f}")
            st.write(f"- **Stop (−{stop_pct_input:.1f}%):** ~ ${stop_price:,.2f}")

# -----------------------------
# Screener run
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
            df = get_history(sym, period=data_period, interval=interval)
            if df.empty:
                continue
            df = compute_indicators(df)
            if df.empty or len(df) < 60:
                continue
            hr_pct, samples = forward_hit_rate_for_target(df, target_pct_input/100.0, horizon_days)
            if math.isnan(hr_pct):
                continue
            latest = df.iloc[-1]

            close = float(latest["Close"])
            atr_val = float(latest["ATR14"]) if not math.isnan(latest["ATR14"]) else np.nan
            target_price = close * (1 + target_pct_input / 100.0)
            stop_price = close * (1 - stop_pct_input / 100.0)
            rr = (target_price - close) / max(close - stop_price, 1e-6)

            row = {
                "Symbol": sym,
                "Close": close,
                "RSI14": float(latest["RSI14"]),
                "MACD": float(latest["MACD"]),
                "MACD_SIGNAL": float(latest["MACD_SIGNAL"]),
                "EMA50": float(latest["EMA50"]),
                "EMA200": float(latest["EMA200"]),
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
            # Ignore single-symbol failures
            pass

        # Update progress
        progress.progress(min(i/len(tickers), 1.0))

    progress.empty()

    if not rows:
        st.error("No candidates found (insufficient data or errors). Try expanding the universe or adjusting settings.")
    else:
        df_all = pd.DataFrame(rows)
        df_all = df_all.dropna(subset=["Score"]).sort_values("Score", ascending=False)
        top10 = df_all.head(10).reset_index(drop=True)

        # Display table
        display_cols = [
            "Symbol","Score","HitRatePct","Samples","Close",
            "TargetPrice","StopPrice","RR","RSI14","EMA50","EMA200","ATR14"
        ]
        st.dataframe(top10[display_cols], use_container_width=True)

        # Download
        csv = top10.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Top 10 (CSV)", data=csv, file_name="sp500_top10.csv", mime="text/csv")

        st.write("---")
        st.subheader("📊 Charts & Indicators (Top 10)")
        for _, r in top10.iterrows():
            st.markdown(f"### {r['Symbol']} &nbsp;|&nbsp; Score **{r['Score']:.1f}** &nbsp;|&nbsp; Hit-rate **{r['HitRatePct']:.1f}%** (n={int(r['Samples'])})")
            cols = st.columns([2, 1])
            with cols[0]:
                tradingview_widget(r["Symbol"], height=420)
            with cols[1]:
                st.metric("Price", f"${r['Close']:,.2f}")
                st.metric("RSI(14)", f"{r['RSI14']:.1f}")
                st.metric("EMA50>EMA200", "Yes ✅" if r["EMA50"] > r["EMA200"] else "No ❌")
                st.metric("MACD-Signal", f"{(r['MACD']-r['MACD_SIGNAL']):.3f}")
                st.metric("ATR(14)", f"{r['ATR14']:.2f}" if not math.isnan(r['ATR14']) else "—")

                st.write("**Trade Ideas (simple):**")
                st.write(f"- **Buy (now):** ~ ${r['Close']:,.2f}")
                st.write(f"- **Target (+{target_pct_input:.1f}%):** ~ ${r['TargetPrice']:,.2f}")
                st.write(f"- **Stop (−{stop_pct_input:.1f}%):** ~ ${r['StopPrice']:,.2f}")
            st.write("---")

# -----------------------------
# Footer
# -----------------------------
st.caption("Data source: Yahoo Finance (via yfinance). TradingView widget © TradingView.")





