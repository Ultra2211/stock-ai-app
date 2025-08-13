# streamlit_app.py
# Full S&P 500 (505) access + manual tickers (AMD, ENPH, etc.), fast batched scan,
# TradingView chart, indicators, % success, investment sizing, earnings/dividends.
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
st.caption("Full-universe scan with manual tickers (AMD, ENPH, etc.). Fast batched downloads. Data via Yahoo Finance; charts by TradingView. Not financial advice.")

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

def tv_symbol(sym: str) -> str:
    # TradingView uses dots for class tickers (e.g., BRK.B)
    return sym.replace("-", ".")

# -----------------------------
# Universe builders
# -----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def sp500_from_yf():
    try:
        syms = yf.tickers_sp500()
        return sorted(list(set(syms)))
    except Exception:
        return []

@st.cache_data(ttl=60*60, show_spinner=False)
def sp500_from_wiki_if_available():
    # If lxml is installed, we can merge Wikipedia (usually most complete)
    try:
        import lxml  # noqa: F401
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        # Normalize Berkshire etc. to Yahoo style with '-'
        syms = df["Symbol"].astype(str).str.replace(r"\.", "-", regex=True).tolist()
        return sorted(list(set(syms)))
    except Exception:
        return []

def build_universe(extra_csv: str):
    base = set(sp500_from_yf())
    wiki = set(sp500_from_wiki_if_available())
    universe = sorted(list(base.union(wiki)))
    # Fallback subset if both are empty
    if not universe:
        universe = sorted([
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
            "V","UNH","XOM","JNJ","PG","LLY","HD","MA","COST","BAC"
        ])
    # Merge forced extra symbols (typed by user)
    extras = []
    if extra_csv:
        extras = [s.strip().upper() for s in extra_csv.split(",") if s.strip()]
    universe = sorted(list(set(universe + extras)))
    return universe

# -----------------------------
# Indicators & scoring
# -----------------------------
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
    """
    Score (0..5):
    +1 EMA20 > EMA50
    +1 MACD > Signal
    +1 35 < RSI < 70
    +1 Close > BB_MID
    +1 0.2 <= BB_%B <= 0.8
    """
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
    """
    % of times price reached +target_pct within next horizon_bars on this timeframe.
    """
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
        m = ~np.isnan(fwd)
        if m.sum() == 0:
            return np.nan, 0
        hits = (fwd[m] >= target_pct).sum()
        total = m.sum()
        return 100.0 * hits / total, int(total)
    except Exception:
        return np.nan, 0

# -----------------------------
# Fast batched download
# -----------------------------
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

# -----------------------------
# TradingView embed
# -----------------------------
def tradingview_iframe(symbol: str, interval_code: str, range_code: str, theme: str, height: int = 560):
    src = (
        "https://s.tradingview.com/widgetembed/?"
        f"symbol={tv_symbol(symbol)}&interval={interval_code}&range={range_code}"
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
with st.sidebar:
    st.header("⚙️ Global Settings")
    extra_tickers = st.text_input("Force‑include extra tickers (comma)", value="AMD, ENPH")
    speed_mode = st.radio("Speed mode", ["Quick (Daily)","Full (Intraday)"], index=0, horizontal=True)
    chart_timeframe = st.selectbox("Chart timeframe", ["1m","5m","15m","30m","1h","1D","1W","1M","3M","6M"], index=5)
    tf_map = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D","1W":"W","1M":"M","3M":"M","6M":"M"}
    range_map = {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M","1W":"1Y","1M":"5Y","3M":"5Y","6M":"ALL"}
    theme = st.radio("Chart theme", ["Light","Dark"], index=0, horizontal=True)
    st.markdown("---")
    investment_amount = st.number_input("Default investment amount ($)", 100, 100000, 1000, 100)
    target_gain = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5)
    stop_loss = st.number_input("Stop loss (%)", 1.0, 30.0, 5.0, 0.5)

# Build universe with extras merged (so AMD/ENPH are guaranteed present)
UNIVERSE = build_universe(extra_tickers)

left, right = st.columns([1.25, 1])

# ----- Left: Chart & Search -----
with left:
    st.subheader("🔎 Always-On Chart & Search")

    # Free text input (ANY symbol: AMD, ENPH, etc.)
    manual_symbol = st.text_input("Type a ticker and press Enter", value="AMD").strip().upper()
    # Also provide a searchable dropdown over the full universe
    dropdown_symbol = st.selectbox("…or pick from S&P 500", UNIVERSE, index=min(UNIVERSE.index("AAPL") if "AAPL" in UNIVERSE else 0, len(UNIVERSE)-1))
    current_symbol = manual_symbol or dropdown_symbol

    tradingview_iframe(current_symbol, tf_map[chart_timeframe], range_map[chart_timeframe], theme, height=560)

    # Quick daily metrics for the chosen symbol
    try:
        df_m = yf.download(current_symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        if df_m is None or df_m.empty:
            st.warning("No Yahoo data for this symbol.")
        else:
            ind_m = compute_indicators(df_m)
            last = ind_m.iloc[-1]
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
            c5.metric("MACD−Signal", fmt(macd_val - macd_sig, 3))
            st.write(f"**Bollinger(20,2):** Mid {fmt(bb_mid,2)}, Upper {fmt(bb_up,2)}, Lower {fmt(bb_low,2)}, %B {fmt(pctb,2)}")
    except Exception:
        st.warning("Unable to load quick metrics for this symbol.")

# ----- Right: Scan -----
with right:
    st.subheader("🏆 Top 10 Scan")

    scan_tf = st.selectbox("Scan timeframe (Full mode only)", ["1m","5m","15m","30m","1h","1D"], index=5)
    scan_tf_to_period = {
        "1m": ("1d", "1m"),
        "5m": ("5d", "5m"),
        "15m": ("5d", "15m"),
        "30m": ("1mo", "30m"),
        "1h": ("3mo", "1h"),
        "1D": ("1y", "1d"),
    }
    max_scan = st.slider("Max symbols to scan", 50, 505, 505, 5)  # default now full 505
    run = st.button("🚀 Run Scan")

    if run:
        period, interval = scan_tf_to_period["1D"] if speed_mode.startswith("Quick") else scan_tf_to_period[scan_tf]
        symbols = UNIVERSE[:max_scan]

        st.info(f"Scanning {len(symbols)} symbols on {interval}… (batched)")
        data_dict = download_batched(symbols, period, interval)

        rows, todays_earnings, todays_divs = [], [], []
        today = datetime.now(timezone.utc).date()

        for sym, df in data_dict.items():
            if df.empty or len(df) < 30:
                continue
            ind = compute_indicators(df)
            last = ind.iloc[-1]
            close = safe_float(last.get("Close"))
            if np.isnan(close) or close <= 0:
                continue
            score = score_row(last)

            # Investment sizing & P/L
            shares = int(investment_amount // close) if close > 0 else 0
            tgt_price = close * (1 + target_gain/100)
            stop_price = close * (1 - stop_loss/100)
            potential_profit = (tgt_price - close) * shares

            # % Success (hit rate) on this timeframe
            # horizon_bars ~ 20 bars by default (~1 month on daily; ~100 min on 5m)
            horizon_bars = 20 if interval != "1m" else 60
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], target_gain/100, horizon_bars=horizon_bars)

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "Score": score,
                "Buy": round(close, 2),
                "Target": round(tgt_price, 2),
                "Stop": round(stop_price, 2),
                "Shares": shares,
                "Potential $": round(potential_profit, 2),
                "% Success": round(hit_rate, 1) if not np.isnan(hit_rate) else None,
                "Samples": samples
            })

            # Earnings/Dividends today (best-effort)
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
            st.warning("No candidates collected. Try Quick (Daily) mode or reduce timeframe strictness.")
        else:
            dfres = pd.DataFrame(rows).sort_values(["Score","% Success"], ascending=[False, False]).head(10).reset_index(drop=True)
            st.dataframe(dfres, use_container_width=True)

            pick = st.selectbox("📊 View Top‑10 chart:", dfres["Symbol"], index=0)
            tradingview_iframe(pick, tf_map[chart_timeframe], range_map[chart_timeframe], theme, height=420)

        st.markdown("### 📅 Earnings Today")
        st.dataframe(pd.DataFrame(todays_earnings) if todays_earnings else pd.DataFrame([{"Symbol":"—"}]), use_container_width=True)

        st.markdown("### 💸 Dividends Today")
        st.dataframe(pd.DataFrame(todays_divs) if todays_divs else pd.DataFrame([{"Symbol":"—"}]), use_container_width=True)

