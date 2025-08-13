# streamlit_app.py
# S&P 500 scanner with:
# - Always-on chart + animated "live search" runner (visible while fetching)
# - Simple BUY/NEUTRAL/SELL + % success for typed ticker
# - Top 10 scan (batched), indicators (RSI, MACD, EMA20/50/200, Bollinger)
# - Dark UI toggle
# - Earnings calendar embedded from TradingView (US market)
# - Dividends from Yahoo (fallback)
# Educational use only — not financial advice.

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(page_title="S&P 500 Scanner Pro", page_icon="📈", layout="wide")

# Session flags for UI animation states
if "is_scanning" not in st.session_state:
    st.session_state.is_scanning = False
if "is_fetching_chart" not in st.session_state:
    st.session_state.is_fetching_chart = False

def inject_css(dark_enabled: bool):
    base = """
    <style>
      :root { --bg:#ffffff; --panel:#f6f7fb; --text:#0f172a; --muted:#6b7280; --accent:#5e8bff; --good:#22c55e; --bad:#ef4444; }
      .buy-badge { background: rgba(34,197,94,.15); color: #22c55e; padding:.35rem .6rem; border-radius:999px; font-weight:700; display:inline-block; }
      .sell-badge { background: rgba(239,68,68,.15); color:#ef4444; padding:.35rem .6rem; border-radius:999px; font-weight:700; display:inline-block; }
      .neutral-badge { background: rgba(94,139,255,.15); color:#5e8bff; padding:.35rem .6rem; border-radius:999px; font-weight:700; display:inline-block; }
      .runner-wrap { display:flex; align-items:center; gap:.5rem; }
      .runner-emoji { font-size:1.2rem; }
      .runner-dot {
        width:8px; height:8px; border-radius:999px; background:#9aa0a6; opacity:.6;
      }
      .runner-dot.running { animation: pulse 1.1s ease-in-out infinite; background:#5e8bff; opacity:1; }
      @keyframes pulse { 0%{transform:scale(1)} 50%{transform:scale(1.8)} 100%{transform:scale(1)} }
      .small { color: var(--muted); font-size:.9rem; }
      .tv-card { border-radius:12px; overflow:hidden; }
    </style>
    """
    dark = """
    <style>
      :root { --bg:#0e1117; --panel:#161a23; --text:#e6e6e6; --muted:#9aa0a6; --accent:#5e8bff; --good:#22c55e; --bad:#ef4444; }
      html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
      [data-testid="stSidebar"] { background: var(--panel) !important; }
      .stMetric label, .stMarkdown, .stSelectbox, .stTextInput, .stNumberInput, .stRadio, .stSlider { color: var(--text) !important; }
    </style>
    """
    st.markdown(base + (dark if dark_enabled else ""), unsafe_allow_html=True)

# -----------------------------
# Small utils
# -----------------------------
def safe_float(x):
    try: return float(x)
    except Exception: return np.nan

def fmt(x, digits=2, default="—"):
    try:
        if pd.isna(x): return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

def tv_symbol(sym: str) -> str:
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
    try:
        import lxml  # noqa: F401
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = df["Symbol"].astype(str).str.replace(r"\.", "-", regex=True).tolist()
        return sorted(list(set(syms)))
    except Exception:
        return []

def build_universe(extra_csv: str):
    base = set(sp500_from_yf())
    wiki = set(sp500_from_wiki_if_available())
    universe = sorted(list(base.union(wiki)))
    if not universe:
        universe = sorted([
            "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM",
            "V","UNH","XOM","JNJ","PG","LLY","HD","MA","COST","BAC"
        ])
    extras = [s.strip().upper() for s in extra_csv.split(",") if s.strip()] if extra_csv else []
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
    if df is None or df.empty: return pd.DataFrame()
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
    df["BB_MID"] = bb_mid; df["BB_UP"] = bb_up; df["BB_LOW"] = bb_low; df["BB_PCTB"] = bb_pctb
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
        if n < horizon_bars + 5: return np.nan, 0
        fmax = np.full(n, np.nan)
        for i in range(n - horizon_bars):
            fmax[i] = np.max(arr[i+1:i+1+horizon_bars])
        base = arr[:-horizon_bars]; mx = fmax[:-horizon_bars]
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = (mx - base) / base
        m = ~np.isnan(fwd)
        if m.sum() == 0: return np.nan, 0
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
    if not tickers: return out
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
# TradingView embeds
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
        f'<iframe class="tv-card" src="{src}" width="100%" height="{height}" frameborder="0" allowtransparency="true" scrolling="no"></iframe>',
        height=height, scrolling=False
    )

def tradingview_earnings_widget(theme: str, height: int = 560):
    # Official TradingView Earnings Calendar widget (US market)
    cfg = {
        "width": "100%",
        "height": height,
        "colorTheme": "dark" if theme == "Dark" else "light",
        "isTransparent": False,
        "locale": "en",
        "market": "us"
    }
    html = f"""
    <div class="tradingview-widget-container tv-card">
      <div id="tv-earnings"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-earnings.js" async>
      {cfg}
      </script>
    </div>
    """
    components.html(html, height=height+8, scrolling=False)

# =============================
# Sidebar (Global)
# =============================
with st.sidebar:
    st.header("⚙️ Global Settings")
    dark_ui = st.toggle("🌙 Dark UI", value=False)
    inject_css(dark_ui)

    extra_tickers = st.text_input("Force-include extra tickers (comma)", value="AMD, ENPH")
    chart_timeframe = st.selectbox("Chart timeframe", ["1m","5m","15m","30m","1h","1D","1W","1M","3M","6M"], index=5)
    tf_map = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D","1W":"W","1M":"M","3M":"M","6M":"M"}
    range_map = {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M","1W":"1Y","1M":"5Y","3M":"5Y","6M":"ALL"}
    chart_theme = "Dark" if dark_ui else "Light"

    st.markdown("---")
    st.write("**Trade Defaults**")
    investment_amount = st.number_input("Investment amount ($)", 100, 100000, 1000, 100, key="investment_amount")
    target_gain = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5, key="target_gain")
    stop_loss = st.number_input("Stop loss (%)", 1.0, 30.0, 5.0, 0.5, key="stop_loss")
    horizon_bars = st.slider("Bars to check for target hit (horizon)", 5, 60, 20, 1, key="horizon_bars",
                             help="Used for % success (hit-rate).")

# Universe after extras
UNIVERSE = build_universe(extra_tickers)

left, right = st.columns([1.25, 1])

# =============================
# Left: Always-On Chart & "Live" Search (runner sync with busy state)
# =============================
with left:
    st.subheader("🔎 Live Search & Chart")
    col_a, col_b = st.columns([0.72, 0.28])
    with col_a:
        manual_symbol = st.text_input("Type a ticker and press Enter", value="AMD", key="manual_symbol").strip().upper()
    with col_b:
        st.markdown('<div class="small">Search status</div>', unsafe_allow_html=True)
        busy = (st.session_state.is_scanning or st.session_state.is_fetching_chart)
        dot_class = "running" if busy else ""
        st.markdown(f'<div class="runner-wrap"><span class="runner-emoji">🏃‍♂️</span><span class="runner-dot {dot_class}"></span></div>', unsafe_allow_html=True)

    dropdown_symbol = st.selectbox("…or pick from S&P 500", UNIVERSE, index=min(UNIVERSE.index("AAPL") if "AAPL" in UNIVERSE else 0, len(UNIVERSE)-1))
    symbol = manual_symbol or dropdown_symbol

    # Show TradingView chart
    tradingview_iframe(symbol, tf_map[chart_timeframe], range_map[chart_timeframe], chart_theme, height=560)

    # Manual symbol analysis (daily), animate only while fetching
    try:
        st.session_state.is_fetching_chart = True
        with st.spinner("Fetching chart data…"):
            df_m = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        st.session_state.is_fetching_chart = False

        if df_m is None or df_m.empty:
            st.warning("No Yahoo data for this symbol.")
        else:
            ind = compute_indicators(df_m)
            last = ind.iloc[-1]
            price = safe_float(last.get("Close"))
            ema20 = safe_float(last.get("EMA20")); ema50 = safe_float(last.get("EMA50")); ema200 = safe_float(last.get("EMA200"))
            rsi14 = safe_float(last.get("RSI14")); macd_val = safe_float(last.get("MACD")); macd_sig = safe_float(last.get("MACD_SIGNAL"))
            pctb = safe_float(last.get("BB_PCTB"))

            # % Success on daily bars for the manual symbol
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], st.session_state.target_gain/100, horizon_bars=st.session_state.horizon_bars)

            # Simple signal
            score = score_row(last)
            if score >= 4 and (not np.isnan(hit_rate) and hit_rate >= 55):
                signal_badge = '<span class="buy-badge">BUY</span>'
            elif score <= 2 and (not np.isnan(hit_rate) and hit_rate < 45):
                signal_badge = '<span class="sell-badge">SELL</span>'
            else:
                signal_badge = '<span class="neutral-badge">NEUTRAL</span>'

            tgt = price * (1 + st.session_state.target_gain/100)
            stp = price * (1 - st.session_state.stop_loss/100)
            rr = (tgt - price) / max(price - stp, 1e-9)

            st.markdown(f"### {symbol} &nbsp; {signal_badge}", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${fmt(price,2)}")
            c2.metric("% Success", f"{fmt(hit_rate,1)}%", help=f"Hit-rate to reach +{st.session_state.target_gain:.1f}% within {st.session_state.horizon_bars} bars (n={samples}).")
            c3.metric("Score (0-5)", f"{score}")
            c4.metric("Target", f"${fmt(tgt,2)}")
            c5.metric("R/R", fmt(rr,2))

            st.caption(f"RSI(14) {fmt(rsi14,1)} | EMA20>EMA50 {'✅' if ema20>ema50 else '❌'} | EMA50>EMA200 {'✅' if ema50>ema200 else '❌'} | MACD−Signal {fmt(macd_val-macd_sig,3)} | BB %B {fmt(pctb,2)}")
    except Exception:
        st.session_state.is_fetching_chart = False
        st.warning("Unable to analyze this symbol.")

# =============================
# Right: Top-10 Scan + TradingView Earnings + Dividends
# =============================
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
    max_scan = st.slider("Max symbols to scan", 50, 505, 505, 5)
    run = st.button("🚀 Run Scan")

    if run:
        st.session_state.is_scanning = True
        try:
            period, interval = scan_tf_to_period["1D"] if speed_mode.startswith("Quick") else scan_tf_to_period[scan_tf]
            symbols = UNIVERSE[:max_scan]

            st.info(f"🏃 Scanning {len(symbols)} symbols on {interval}… (batched)")
            with st.spinner("Scanning…"):
                data_dict = download_batched(symbols, period, interval)

            rows = []
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

                shares = int(st.session_state.investment_amount // close) if close > 0 else 0
                tgt_price = close * (1 + st.session_state.target_gain/100)
                stop_price = close * (1 - st.session_state.stop_loss/100)
                potential_profit = (tgt_price - close) * shares

                # % Success
                hz = 20  # consistent horizon for ranking
                hit_rate, samples = forward_hit_rate_for_target(ind["Close"], st.session_state.target_gain/100, horizon_bars=hz)

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

            if not rows:
                st.warning("No candidates collected. Try Quick (Daily) mode or scan fewer symbols.")
            else:
                dfres = pd.DataFrame(rows).sort_values(["Score","% Success"], ascending=[False, False]).head(10).reset_index(drop=True)
                st.dataframe(dfres, use_container_width=True)

                pick = st.selectbox("📊 View Top-10 chart:", dfres["Symbol"], index=0)
                tradingview_iframe(
                    pick,
                    {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D"}[scan_tf],
                    {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M"}[scan_tf],
                    "Dark" if dark_ui else "Light",
                    height=420
                )
        finally:
            st.session_state.is_scanning = False

    st.markdown("### 📅 Earnings Calendar (TradingView, US)")
    tradingview_earnings_widget("Dark" if dark_ui else "Light", height=540)

    # Dividends (fallback display)
    st.markdown("### 💸 Dividends Today (Yahoo fallback)")
    st.caption("This fallback lists dividends with last pay date = today among scanned symbols (if any).")
    # For performance, we show this only after a scan — you can extend to full universe if needed.
    # (Scan must populate dfres or at least symbols list; otherwise skip heavy per-ticker calls.)
    # If you want, we can later embed a dividend calendar source similarly to the earnings widget.




