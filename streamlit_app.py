# streamlit_app.py
# S&P 500 always-on chart + search (left) and Top-10 scan (right)
# Indicators: RSI, MACD, EMA20/50/200, Bollinger Bands (20, 2)
# Timeframe controls, speed mode, target/stop, and today's earnings/dividends.
# Educational use only — not financial advice.

import math
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
st.title("📈 S&P 500 Scanner — Chart, Top 10, Earnings & Dividends")
st.caption("Interactive TradingView charts, multi-timeframe scan, and today's corporate actions. Data via Yahoo Finance; charts by TradingView. Not financial advice.")

# -----------------------------
# Helpers (robust formatting)
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
    # Avoid lxml by using yfinance helper
    try:
        syms = yf.tickers_sp500()
        return sorted(list(set(syms)))
    except Exception:
        # fallback subset
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
    # %B: where price sits within bands (0=lower,1=upper)
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
    """
    Simple, robust score (0..5):
    +1: EMA20 > EMA50 (short-term trend up)
    +1: MACD > Signal (bullish momentum)
    +1: 35 < RSI < 70 (not oversold/overbought extremes)
    +1: Close > BB_MID (above mid-band = constructive)
    +1: BB_PCTB between 0.2 and 0.8 (not hugging bands)
    """
    score = 0
    if safe_float(row.get("EMA20")) > safe_float(row.get("EMA50")):
        score += 1
    if safe_float(row.get("MACD")) > safe_float(row.get("MACD_SIGNAL")):
        score += 1
    r = safe_float(row.get("RSI14"))
    if 35 < r < 70:
        score += 1
    if safe_float(row.get("Close")) > safe_float(row.get("BB_MID")):
        score += 1
    pctb = safe_float(row.get("BB_PCTB"))
    if 0.2 <= pctb <= 0.8:
        score += 1
    return score

@st.cache_data(ttl=20*60, show_spinner=False)
def get_history(symbol: str, period="6mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.dropna().copy()
    except Exception:
        return pd.DataFrame()

def tradingview_iframe(symbol: str, interval_code: str, range_code: str, theme: str, height: int = 560):
    """
    Embed TradingView chart with timeframe (interval) + visible range.
    interval_code: "1","5","15","30","60","D","W","M"
    range_code: "1D","5D","1M","3M","6M","1Y","5Y","ALL"
    """
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
        "&withdateranges=1"
        "&allow_symbol_change=1"
        "&details=1"
        "&hideideas=1"
    )
    components.html(
        f'<iframe src="{src}" width="100%" height="{height}" frameborder="0" allowtransparency="true" scrolling="no"></iframe>',
        height=height, scrolling=False
    )

# -----------------------------
# UI Controls (Left = Chart/Search)
# -----------------------------
sp500 = get_sp500_symbols()

left, right = st.columns([1.25, 1])

with left:
    st.subheader("🔎 Always-On Chart & Search")

    # Chart timeframe controls
    timeframe = st.selectbox(
        "Chart timeframe",
        ["1m","5m","15m","30m","1h","1D","1W","1M","3M","6M"],
        index=5
    )
    tf_map = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D","1W":"W","1M":"M","3M":"M","6M":"M"}
    range_map = {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M","1W":"1Y","1M":"5Y","3M":"5Y","6M":"ALL"}
    interval_code = tf_map[timeframe]
    range_code = range_map[timeframe]
    theme = st.radio("Chart theme", ["Light","Dark"], index=0, horizontal=True)

    # Search / pick any ticker
    default_symbol = "AAPL" if "AAPL" in sp500 else (sp500[0] if sp500 else "AAPL")
    symbol = st.selectbox("S&P 500 ticker", sp500, index=sp500.index(default_symbol) if default_symbol in sp500 else 0)

    # Show TradingView chart
    tradingview_iframe(symbol, interval_code, range_code, theme, height=560)

    # Quick metrics for chosen symbol (daily data for stability)
    df_m = get_history(symbol, period="1y", interval="1d")
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
        bb_mid = safe_float(last.get("BB_MID"))
        bb_up = safe_float(last.get("BB_UP"))
        bb_low = safe_float(last.get("BB_LOW"))
        pctb = safe_float(last.get("BB_PCTB"))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price", f"${fmt(price,2)}")
        c2.metric("RSI(14)", fmt(rsi14,1))
        c3.metric("EMA20>EMA50", "Yes ✅" if ema20 > ema50 else "No ❌")
        c4.metric("EMA50>EMA200", "Yes ✅" if ema50 > ema200 else "No ❌")
        c5.metric("MACD-Signal", fmt(macd_val - macd_sig, 3))

        st.write(f"**Bollinger(20,2):** Mid {fmt(bb_mid,2)}, Upper {fmt(bb_up,2)}, Lower {fmt(bb_low,2)}, %B {fmt(pctb,2)}")

# -----------------------------
# UI Controls (Right = Scan/Earnings/Dividends)
# -----------------------------
with right:
    st.subheader("🏆 Top 10 Scan (click to run)")

    # Speed mode: quick daily vs full intraday
    speed_mode = st.radio("Speed mode", ["Quick (Daily)","Full (Intraday)"], index=0, horizontal=True)

    # Scan timeframe separate from chart timeframe
    scan_tf = st.selectbox(
        "Scan timeframe (applies in Full mode)",
        ["1m","5m","15m","30m","1h","1D"],
        index=5,
        help="In Quick mode we force 1D for speed. In Full mode, pick an intraday or 1D interval."
    )
    scan_tf_to_period = {
        "1m": ("1d", "1m"),
        "5m": ("5d", "5m"),
        "15m": ("5d", "15m"),
        "30m": ("1mo", "30m"),
        "1h": ("3mo", "1h"),
        "1D": ("1y", "1d"),
    }

    # Targets & universe size
    target_gain = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5)
    stop_loss = st.number_input("Stop loss (%)", 1.0, 30.0, 5.0, 0.5)
    max_scan = st.slider("Max symbols to scan", 50, 505, 300, 25)

    run = st.button("🚀 Run Scan")

    # Containers to keep layout stable
    top10_container = st.container()
    earnings_container = st.container()
    dividends_container = st.container()

    if run:
        # Choose scan period/interval based on speed mode
        if speed_mode.startswith("Quick"):
            scan_period, scan_interval = scan_tf_to_period["1D"]
        else:
            scan_period, scan_interval = scan_tf_to_period[scan_tf]

        syms = sp500[:max_scan] if len(sp500) > max_scan else sp500

        progress = st.progress(0.0, text=f"Scanning {len(syms)} symbols…")
        rows = []
        todays_earnings = []
        todays_divs = []

        today = datetime.now(timezone.utc).date()  # compare by date (UTC)

        for i, sym in enumerate(syms, start=1):
            # Download scan data
            df = get_history(sym, period=scan_period, interval=scan_interval)
            if df.empty or len(df) < 30:
                progress.progress(min(i/len(syms), 1.0))
                continue

            ind = compute_indicators(df)
            last = ind.iloc[-1]
            close = safe_float(last.get("Close"))
            if pd.isna(close):
                progress.progress(min(i/len(syms), 1.0))
                continue

            s = score_row(last)
            tgt = close * (1 + target_gain/100.0)
            stp = close * (1 - stop_loss/100.0)

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "RSI14": round(safe_float(last.get("RSI14")), 1) if not pd.isna(last.get("RSI14")) else np.nan,
                "EMA20>EMA50": (safe_float(last.get("EMA20")) > safe_float(last.get("EMA50"))),
                "MACD>Signal": (safe_float(last.get("MACD")) > safe_float(last.get("MACD_SIGNAL"))),
                "BB %B": round(safe_float(last.get("BB_PCTB")), 2) if not pd.isna(last.get("BB_PCTB")) else np.nan,
                "Score(0-5)": s,
                "Buy": round(close, 2),
                f"Target(+{int(target_gain)}%)": round(tgt, 2),
                f"Stop(-{int(stop_loss)}%)": round(stp, 2),
            })

            # Corporate actions (best-effort; resilient to errors)
            try:
                tk = yf.Ticker(sym)

                # Earnings calendar (tk.calendar format varies; normalize)
                cal = None
                try:
                    cal = tk.calendar
                except Exception:
                    cal = None
                if cal is not None and not cal.empty:
                    for val in cal.to_numpy().ravel():
                        try:
                            d = pd.to_datetime(val).date()
                            if d == today:
                                todays_earnings.append({"Symbol": sym, "Earnings Date (UTC)": str(d)})
                                break
                        except Exception:
                            pass

                # Dividends
                divs = None
                try:
                    divs = tk.dividends
                except Exception:
                    divs = None
                if divs is not None and not divs.empty:
                    last_div_date = pd.to_datetime(divs.index[-1]).date()
                    if last_div_date == today:
                        todays_divs.append({"Symbol": sym, "Dividend (last)": float(divs.iloc[-1])})
            except Exception:
                pass

            progress.progress(min(i/len(syms), 1.0))

        progress.empty()

        # Build Top 10 (always show something by sorting score desc, then RSI asc)
        if not rows:
            top10_container.warning("No scan rows collected. Try increasing Max symbols or using Quick (Daily) mode for speed and stability.")
        else:
            dfres = pd.DataFrame(rows).sort_values(["Score(0-5)", "RSI14"], ascending=[False, True]).head(10).reset_index(drop=True)
            with top10_container:
                st.markdown("### 🥇 Top 10 Candidates")
                st.dataframe(dfres, use_container_width=True)

                # Quick chart for a selected Top-10 symbol using current chart timeframe
                pick = st.selectbox("📊 View Top-10 chart:", dfres["Symbol"], index=0)
                tradingview_iframe(pick, interval_code, range_code, theme, height=420)

        # Earnings today
        with earnings_container:
            st.markdown("### 📅 Earnings Today (UTC)")
            if todays_earnings:
                st.dataframe(pd.DataFrame(todays_earnings).sort_values("Symbol"), use_container_width=True)
            else:
                st.info("No S&P 500 earnings detected for today (Yahoo Finance calendar).")

        # Dividends today
        with dividends_container:
            st.markdown("### 💸 Dividends Paid Today (UTC)")
            if todays_divs:
                st.dataframe(pd.DataFrame(todays_divs).sort_values("Symbol"), use_container_width=True)
            else:
                st.info("No S&P 500 dividends detected for today (Yahoo Finance data).")

# -----------------------------
# Footer
# -----------------------------
st.caption("Tip: Use Quick (Daily) for fast, broad scans. Switch to Full (Intraday) when you need tighter timing. Intraday scans are slower and may return fewer rows due to API limits.")
