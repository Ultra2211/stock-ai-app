# streamlit_app.py
# Fast batched S&P 500 scanner with TradingView chart, indicators, investment sizing,
# % success hit rate, earnings, and dividends.
# Educational use only — not financial advice.

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="S&P 500 Scanner Pro", page_icon="📈", layout="wide")
st.title("📈 S&P 500 Scanner — Chart, Top 10, % Success & Earnings/Dividends")
st.caption("Fast batched scanning with interactive TradingView charts. Data via Yahoo Finance; charts by TradingView. Not financial advice.")

# -----------------------------
# Small utils
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def fmt(x, digits=2, default="—"):
    try:
        if pd.isna(x):
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

@st.cache_data(ttl=60*60, show_spinner=False)
def get_sp500_symbols():
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

def bollinger(series: pd.Series, length: int = 20, mult: float = 2.0):
    mid = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = mid + mult * std
    lower = mid - mult * std
    pctb = (series - lower) / (upper - lower)
    return mid, upper, lower, pctb

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    macd_line, sig, hist = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = sig
    df["MACD_HIST"] = hist
    bb_mid, bb_up, bb_low, bb_pctb = bollinger(df["Close"], 20, 2.0)
    df["BB_MID"] = bb_mid
    df["BB_UP"] = bb_up
    df["BB_LOW"] = bb_low
    df["BB_PCTB"] = bb_pctb
    return df

def score_row(row):
    score = 0
    if safe_float(row.get("EMA20")) > safe_float(row.get("EMA50")): score += 1
    if safe_float(row.get("MACD")) > safe_float(row.get("MACD_SIGNAL")): score += 1
    r = safe_float(row.get("RSI14"))
    if 35 < r < 70: score += 1
    if safe_float(row.get("Close")) > safe_float(row.get("BB_MID")): score += 1
    pctb = safe_float(row.get("BB_PCTB"))
    if 0.2 <= pctb <= 0.8: score += 1
    return score

def forward_hit_rate_for_target(close: pd.Series, target_pct: float, horizon_bars: int):
    try:
        arr = close.values
        n = len(arr)
        if n < horizon_bars + 5:
            return np.nan, 0
        fmax = np.full(n, np.nan)
        for i in range(n - horizon_bars):
            fmax[i] = np.max(arr[i+1:i+1+horizon_bars])
        base = arr[:-horizon_bars]
        mx = fmax[:-horizon_bars]
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = (mx - base) / base
        mask = ~np.isnan(fwd)
        if mask.sum() == 0:
            return np.nan, 0
        hits = (fwd[mask] >= target_pct).sum()
        total = mask.sum()
        return 100.0 * hits / total, int(total)
    except Exception:
        return np.nan, 0

@st.cache_data(ttl=15*60, show_spinner=False)
def download_batched(tickers, period, interval, batch_size=60):
    out = {}
    if not tickers:
        return out
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i:i+batch_size]
        try:
            df = yf.download(" ".join(chunk), period=period, interval=interval, auto_adjust=False,
                             group_by="ticker", progress=False, threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for sym in chunk:
                    if sym in df.columns.get_level_values(0):
                        sub = df[sym].dropna().copy()
                        if not sub.empty and "Close" in sub:
                            out[sym] = sub
            else:
                if not df.empty and "Close" in df:
                    out[chunk[0]] = df.dropna().copy()
        except Exception:
            pass
    return out

def tradingview_iframe(symbol: str, interval_code: str, range_code: str, theme: str, height: int = 560):
    src = (
        "https://s.tradingview.com/widgetembed/?"
        f"symbol={symbol}&interval={interval_code}&range={range_code}"
        "&hidesidetoolbar=0&hidetoptoolbar=0&symboledit=1&saveimage=1"
        "&toolbarbg=f1f3f6"
        f"&theme={'dark' if theme=='Dark' else 'light'}"
        "&style=1&timezone=Etc/UTC&withdateranges=1&allow_symbol_change=1"
        "&details=1&hideideas=1"
    )
    components.html(
        f'<iframe src="{src}" width="100%" height="{height}" frameborder="0" allowtransparency="true" scrolling="no"></iframe>',
        height=height, scrolling=False
    )

# -----------------------------
# UI Controls
# -----------------------------
sp500 = get_sp500_symbols()
left, right = st.columns([1.25, 1])

# ----- Left: Chart & Search -----
with left:
    st.subheader("🔎 Always-On Chart & Search")
    timeframe = st.selectbox("Chart timeframe", ["1m","5m","15m","30m","1h","1D","1W","1M","3M","6M"], index=5)
    tf_map = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D","1W":"W","1M":"M","3M":"M","6M":"M"}
    range_map = {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M","1W":"1Y","1M":"5Y","3M":"5Y","6M":"ALL"}
    theme = st.radio("Chart theme", ["Light","Dark"], index=0, horizontal=True)
    symbol = st.selectbox("S&P 500 ticker", sp500, index=0)
    tradingview_iframe(symbol, tf_map[timeframe], range_map[timeframe], theme, height=560)

# ----- Right: Scan -----
with right:
    st.subheader("🏆 Top 10 Scan")
    speed_mode = st.radio("Speed mode", ["Quick (Daily)","Full (Intraday)"], index=0, horizontal=True)
    scan_tf = st.selectbox("Scan timeframe (Full mode only)", ["1m","5m","15m","30m","1h","1D"], index=5)
    scan_tf_to_period = {
        "1m": ("1d", "1m"),
        "5m": ("5d", "5m"),
        "15m": ("5d", "15m"),
        "30m": ("1mo", "30m"),
        "1h": ("3mo", "1h"),
        "1D": ("1y", "1d"),
    }
    investment_amount = st.number_input("Investment amount ($)", 100, 100000, 1000, 100)
    target_gain = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5)
    stop_loss = st.number_input("Stop loss (%)", 1.0, 30.0, 5.0, 0.5)
    max_scan = st.slider("Max symbols to scan", 50, 505, 200, 25)
    run = st.button("🚀 Run Scan")

    if run:
        period, interval = scan_tf_to_period["1D"] if speed_mode.startswith("Quick") else scan_tf_to_period[scan_tf]
        syms = sp500[:max_scan]
        data_dict = download_batched(syms, period, interval)
        rows, todays_earnings, todays_divs = [], [], []
        today = datetime.now(timezone.utc).date()

        for sym, df in data_dict.items():
            if df.empty or len(df) < 30:
                continue
            ind = compute_indicators(df)
            last = ind.iloc[-1]
            close = safe_float(last.get("Close"))
            score = score_row(last)
            shares = investment_amount / close if close > 0 else 0
            tgt_price = close * (1 + target_gain/100)
            stop_price = close * (1 - stop_loss/100)
            potential_profit = (tgt_price - close) * shares
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], target_gain/100, horizon_bars=20)

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "Score": score,
                "Buy": round(close, 2),
                "Target": round(tgt_price, 2),
                "Stop": round(stop_price, 2),
                "Shares": int(shares),
                "Potential $": round(potential_profit, 2),
                "% Success": round(hit_rate, 1) if not np.isnan(hit_rate) else None,
                "Samples": samples
            })

            try:
                tk = yf.Ticker(sym)
                cal = getattr(tk, "calendar", None)
                if cal is not None and not cal.empty:
                    for val in cal.to_numpy().ravel():
                        try:
                            if pd.to_datetime(val).date() == today:
                                todays_earnings.append({"Symbol": sym})
                                break
                        except Exception:
                            pass
                divs = getattr(tk, "dividends", None)
                if divs is not None and not divs.empty:
                    if pd.to_datetime(divs.index[-1]).date() == today:
                        todays_divs.append({"Symbol": sym, "Dividend": float(divs.iloc[-1])})
            except Exception:
                pass

        if not rows:
            st.warning("No candidates found.")
        else:
            dfres = pd.DataFrame(rows).sort_values(["Score","% Success"], ascending=[False, False]).head(10).reset_index(drop=True)
            st.dataframe(dfres, use_container_width=True)

            pick = st.selectbox("📊 View Top-10 chart:", dfres["Symbol"], index=0)
            tradingview_iframe(pick, tf_map[timeframe], range_map[timeframe], theme, height=420)

        st.markdown("### 📅 Earnings Today")
        st.dataframe(pd.DataFrame(todays_earnings) if todays_earnings else pd.DataFrame([{"Symbol":"None"}]))

        st.markdown("### 💸 Dividends Today")
        st.dataframe(pd.DataFrame(todays_divs) if todays_divs else pd.DataFrame([{"Symbol":"None"}]))

