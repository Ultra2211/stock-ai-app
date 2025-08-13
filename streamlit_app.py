# streamlit_app.py
# Dark-mode-only S&P 500 scanner with robust dark styling for ALL dropdowns/menus.
# - TradingView chart + ticker search (timeframe controlled on the chart)
# - Expanded indicators: EMA20/50/200, RSI, MACD, Bollinger, Stochastic, ADX(+DI/−DI), MFI, ATR, Supertrend(10,3), OBV
# - 10-point score + blended % Success
# - Top-10 scan (batched) + TradingView earnings calendar (US)
# Educational use only — not financial advice.

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# -----------------------------
# Page / Theme (Dark mode only)
# -----------------------------
st.set_page_config(page_title="S&P 500 Scanner Pro (Dark)", page_icon="📈", layout="wide")

def inject_dark_css():
    st.markdown("""
    <style>
      :root {
        --bg:#0e1117; --panel:#161a23; --text:#ffffff; --muted:#c8c8c8;
        --blue:#2563eb; --blue-soft:#1d4ed8; --blue-contrast:#ffffff;
        --border:#2a2f3a; --input:#11141a; --menu:#0f1420; --menu-hover:#1d2330;
        --good:#22c55e; --bad:#ef4444;
      }

      html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
      [data-testid="stSidebar"] { background: var(--panel) !important; }
      /* Force WHITE text globally in dark mode */
      label, .stMarkdown, .stRadio, .stSlider, .stSelectbox, .stNumberInput, .stTextInput,
      div, span, p { color: var(--text) !important; }

      /* Inputs */
      .stTextInput input, .stNumberInput input {
        color: var(--text) !important; background: var(--input) !important; border-color: var(--border) !important;
      }

      /* ===== BaseWeb Select (Streamlit selectbox) – value container, parts, caret ===== */
      div[data-baseweb="select"] > div {
        background: var(--input) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
      }
      /* ensure all descendants (value text, placeholder, singleValue, input) are white */
      div[data-baseweb="select"] * { color: var(--text) !important; }
      div[data-baseweb="select"] input { color: var(--text) !important; }
      div[data-baseweb="select"] svg { fill: var(--text) !important; }

      /* ===== Select MENU overlay ===== */
      div[data-baseweb="menu"] {
        background: var(--menu) !important;
        border: 1px solid var(--border) !important;
      }
      /* text within menu (options, groups, etc.) */
      div[data-baseweb="menu"] * { color: var(--text) !important; }
      div[data-baseweb="option"] {
        background: transparent !important;
        color: var(--text) !important;
      }
      div[data-baseweb="option"]:hover { background: var(--menu-hover) !important; }
      div[data-baseweb="option"][aria-selected="true"] { background: var(--menu-hover) !important; }

      /* Spinner visibility */
      div[data-testid="stSpinner"] *, div[role="alert"] * { color: var(--text) !important; }

      /* Compact metrics */
      div[data-testid="stMetric"] { padding:.25rem .5rem; }
      div[data-testid="stMetricValue"] { font-size:1.2rem; line-height:1.2rem; }
      div[data-testid="stMetricLabel"] { font-size:.85rem; color: var(--muted) !important; }

      /* Badges */
      .buy-badge { background: rgba(34,197,94,.15); color:#22c55e; padding:.35rem .6rem; border-radius:999px; font-weight:700; display:inline-block; }
      .sell-badge { background: rgba(239,68,68,.15); color:#ef4444; padding:.35rem .6rem; border-radius:999px; font-weight:700; display:inline-block; }
      .neutral-badge { background: rgba(37,99,235,.15); color:#2563eb; padding:.35rem .6rem; border-radius:999px; font-weight:700; display:inline-block; }

      .tv-card { border-radius:12px; overflow:hidden; }

      /* Blue primary button (Run Scan) */
      div.stButton > button {
        background: var(--blue) !important;
        color: var(--blue-contrast) !important;
        border: 1px solid var(--blue) !important;
      }
      div.stButton > button:hover { filter: brightness(0.95); }

      /* Blue accent for ticker heading */
      .ticker-accent { color: var(--blue); font-weight:700; }

      /* Blue label utility */
      .blue-label { color: var(--blue) !important; font-weight:700; margin: 0 0 4px 0; }

      /* Tighter control spacing */
      .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stSlider { margin-bottom:.5rem; }
    </style>
    """, unsafe_allow_html=True)

inject_dark_css()

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

# Default initial chart settings (users change inside TradingView toolbar)
DEFAULT_INTERVAL = "D"   # daily
DEFAULT_RANGE = "6M"     # 6 months initial

# For Top‑10 mini chart (initial view)
TF_MAP = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D"}
RANGE_MAP = {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M"}

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
        import lxml  # noqa
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = df["Symbol"].astype(str).str.replace(r"\.", "-", regex=True).tolist()
        return sorted(list(set(syms)))
    except Exception:
        return []

@st.cache_data(ttl=60*60, show_spinner=False)
def build_universe(extra_csv: str):
    base = set(sp500_from_yf())
    wiki = set(sp500_from_wiki_if_available())
    universe = sorted(list(base.union(wiki)))
    if not universe:
        universe = sorted(["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM"])
    extras = [s.strip().upper() for s in extra_csv.split(",") if s.strip()] if extra_csv else []
    return sorted(list(set(universe + extras)))

# -----------------------------
# Indicator calculations
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

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

def stochastic(high, low, close, k=14, d=3):
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_percent = 100 * (close - lowest) / (highest - lowest + 1e-12)
    d_percent = k_percent.rolling(d).mean()
    return k_percent, d_percent

def true_range(df):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df, period=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df, period=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    tr_sm = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (tr_sm + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (tr_sm + 1e-12)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return plus_di.rename("PLUS_DI"), minus_di.rename("MINUS_DI"), adx_val.rename("ADX")

def mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    mf = tp * df["Volume"]
    pos_flow = np.where(tp > tp.shift(1), mf, 0.0)
    neg_flow = np.where(tp < tp.shift(1), mf, 0.0)
    pos_sm = pd.Series(pos_flow).rolling(period).sum()
    neg_sm = pd.Series(neg_flow).rolling(period).sum()
    mfr = pos_sm / (neg_sm + 1e-12)
    return 100 - (100 / (1 + mfr))

def supertrend(df, period=10, mult=3.0):
    atr_val = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2.0
    upperband = hl2 + mult * atr_val
    lowerband = hl2 - mult * atr_val
    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    in_uptrend = pd.Series(index=df.index, dtype=bool)

    for i in range(len(df)):
        if i == 0:
            in_uptrend.iloc[i] = True
            continue
        if df["Close"].iloc[i-1] > final_upper.iloc[i-1]:
            in_uptrend.iloc[i] = True
        elif df["Close"].iloc[i-1] < final_lower.iloc[i-1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i-1]
            if in_uptrend.iloc[i] and lowerband.iloc[i] < final_lower.iloc[i-1]:
                lowerband.iloc[i] = final_lower.iloc[i-1]
            if not in_uptrend.iloc[i] and upperband.iloc[i] > final_upper.iloc[i-1]:
                upperband.iloc[i] = final_upper.iloc[i-1]
        final_upper.iloc[i] = upperband.iloc[i]
        final_lower.iloc[i] = lowerband.iloc[i]
    return final_upper.rename("ST_UPPER"), final_lower.rename("ST_LOWER"), in_uptrend.rename("ST_UPTREND"), atr_val.rename("ATR")

def on_balance_volume(df):
    dirv = np.sign(df["Close"].diff()).fillna(0.0)
    return (dirv * df["Volume"]).cumsum().rename("OBV")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    macd_line, sig, hist = macd(df["Close"])
    df["MACD"] = macd_line; df["MACD_SIGNAL"] = sig; df["MACD_HIST"] = hist

    bb_mid, bb_up, bb_low, bb_pctb = bollinger(df["Close"], 20, 2.0)
    df["BB_MID"] = bb_mid; df["BB_UP"] = bb_up; df["BB_LOW"] = bb_low; df["BB_PCTB"] = bb_pctb

    k, d = stochastic(df["High"], df["Low"], df["Close"], 14, 3)
    df["STO_K"] = k; df["STO_D"] = d

    plus_di, minus_di, adx_v = adx(df, 14)
    df["PLUS_DI"] = plus_di; df["MINUS_DI"] = minus_di; df["ADX"] = adx_v

    df["MFI14"] = mfi(df, 14)

    st_u, st_l, st_up, atr_v = supertrend(df, 10, 3.0)
    df["ST_UPPER"] = st_u; df["ST_LOWER"] = st_l; df["ST_UPTREND"] = st_up; df["ATR14"] = atr_v

    df["OBV"] = on_balance_volume(df)
    return df

# -----------------------------
# Scoring & success metrics
# -----------------------------
def score_row(row):
    s = 0
    c = safe_float(row.get("Close"))
    if safe_float(row.get("EMA20")) > safe_float(row.get("EMA50")): s += 1
    if safe_float(row.get("EMA50")) > safe_float(row.get("EMA200")): s += 1
    if safe_float(row.get("MACD")) > safe_float(row.get("MACD_SIGNAL")): s += 1
    r = safe_float(row.get("RSI14"))
    if 40 < r < 65: s += 1
    if c > safe_float(row.get("BB_MID")): s += 1
    pctb = safe_float(row.get("BB_PCTB"))
    if 0.2 <= pctb <= 0.8: s += 1
    if safe_float(row.get("STO_K")) > safe_float(row.get("STO_D")) and safe_float(row.get("STO_K")) < 80: s += 1
    if safe_float(row.get("ADX")) > 20 and safe_float(row.get("PLUS_DI")) > safe_float(row.get("MINUS_DI")): s += 1
    mfi14 = safe_float(row.get("MFI14"))
    if 35 < mfi14 < 75: s += 1
    if bool(row.get("ST_UPTREND")): s += 1
    return s

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

def blended_success_pct(hitrate_pct: float, score10: int) -> float:
    a = 0 if hitrate_pct is None or np.isnan(hitrate_pct) else float(hitrate_pct)
    b = max(0, min(10, int(score10))) * 10.0
    return (a + b) / 2.0

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
def tradingview_iframe(symbol: str, interval_code: str, range_code: str, theme: str, height: int = 520):
    src = (
        "https://s.tradingview.com/widgetembed/?"
        f"symbol={tv_symbol(symbol)}&interval={interval_code}&range={range_code}"
        "&hidesidetoolbar=0&hidetoptoolbar=0&symboledit=1&saveimage=1"
        "&toolbarbg=f1f3f6"
        f"&theme=dark"
        "&style=1&timezone=Etc/UTC&withdateranges=1&allow_symbol_change=1"
        "&details=1&hideideas=1"
    )
    components.html(
        f'<iframe class="tv-card" src="{src}" width="100%" height="{height}" frameborder="0" allowtransparency="true" scrolling="no"></iframe>',
        height=height, scrolling=False
    )

def tradingview_earnings_widget(height: int = 500):
    cfg = {
        "width": "100%",
        "height": height,
        "colorTheme": "dark",
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
# Sidebar (Dark-only, globals)
# =============================
with st.sidebar:
    st.header("⚙️ Global Settings (Dark)")
    extra_tickers = st.text_input("Force-include extra tickers (comma)", value="AMD, ENPH")

    st.markdown("---")
    st.write("**Trade Defaults**")
    investment_amount = st.number_input("Investment amount ($)", 100, 100000, 1000, 100, key="investment_amount")
    target_gain = st.number_input("Target gain (%)", 1.0, 50.0, 10.0, 0.5, key="target_gain")
    stop_loss_pct = st.number_input("Stop loss (%)", 1.0, 30.0, 5.0, 0.5, key="stop_loss")
    horizon_bars = st.slider("Bars to check for target hit (horizon)", 5, 60, 20, 1, key="horizon_bars",
                             help="Used for % success (hit-rate).")

UNIVERSE = build_universe(extra_tickers)

left, right = st.columns([1.25, 1])

# =============================
# Left: Search & Chart (timeframe controlled on the chart)
# =============================
with left:
    st.subheader("🔎 Search & Chart")
    manual_symbol = st.text_input("Type a ticker and press Enter", value="AMD", key="manual_symbol").strip().upper()
    dropdown_symbol = st.selectbox("…or pick from S&P 500", UNIVERSE, index=min(UNIVERSE.index("AAPL") if "AAPL" in UNIVERSE else 0, len(UNIVERSE)-1))
    symbol = manual_symbol or dropdown_symbol

    tradingview_iframe(symbol, "D", "6M", "Dark", height=520)

    # Manual symbol analysis (daily)
    try:
        with st.spinner("Fetching chart data…"):
            df_m = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        if df_m is None or df_m.empty:
            st.warning("No Yahoo data for this symbol.")
        else:
            ind = compute_indicators(df_m)
            last = ind.iloc[-1]
            price = safe_float(last.get("Close"))

            # Scores & success
            score10 = score_row(last)
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], st.session_state.target_gain/100, horizon_bars=st.session_state.horizon_bars)
            success_blend = blended_success_pct(hit_rate, score10)

            # Signal badge
            if score10 >= 7 and (not np.isnan(success_blend) and success_blend >= 55):
                signal_badge = '<span class="buy-badge">BUY</span>'
            elif score10 <= 3 and (not np.isnan(success_blend) and success_blend < 45):
                signal_badge = '<span class="sell-badge">SELL</span>'
            else:
                signal_badge = '<span class="neutral-badge">NEUTRAL</span>'

            # Targets & stops
            tgt = price * (1 + st.session_state.target_gain/100)
            stp_pct_val = price * (1 - st.session_state.stop_loss/100)
            atr_val = safe_float(last.get("ATR14"))
            stp_atr = price - 1.5 * atr_val if not np.isnan(atr_val) else np.nan
            stp_use = stp_pct_val
            rr = (tgt - price) / max(price - stp_use, 1e-9)

            ema20 = safe_float(last.get("EMA20")); ema50 = safe_float(last.get("EMA50")); ema200 = safe_float(last.get("EMA200"))
            rsi14 = safe_float(last.get("RSI14")); macd_spread = safe_float(last.get("MACD")) - safe_float(last.get("MACD_SIGNAL"))
            pctb = safe_float(last.get("BB_PCTB"))
            adx_v = safe_float(last.get("ADX")); di_plus = safe_float(last.get("PLUS_DI")); di_minus = safe_float(last.get("MINUS_DI"))
            sto_k = safe_float(last.get("STO_K")); sto_d = safe_float(last.get("STO_D")); mfi14 = safe_float(last.get("MFI14"))
            st_up = bool(last.get("ST_UPTREND"))

            st.markdown(f"### <span class='ticker-accent'>{symbol}</span> &nbsp; {signal_badge}", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${fmt(price,2)}")
            c2.metric("% Success (blend)", f"{fmt(success_blend,1)}%", help=f"Blend of hit-rate and indicator score (n={samples}).")
            c3.metric("Score (0–10)", f"{score10}")
            c4.metric("Target", f"${fmt(tgt,2)}")
            c5.metric("R/R", fmt(rr,2))

            st.caption(
                f"RSI(14) {fmt(rsi14,1)} | EMA20>50 {'✅' if ema20>ema50 else '❌'} | EMA50>200 {'✅' if ema50>ema200 else '❌'} | "
                f"MACD−Sig {fmt(macd_spread,3)} | BB %B {fmt(pctb,2)} | "
                f"Stoch K/D {fmt(sto_k,1)}/{fmt(sto_d,1)} | ADX {fmt(adx_v,1)} (+DI {fmt(di_plus,1)} / −DI {fmt(di_minus,1)}) | "
                f"MFI {fmt(mfi14,1)} | Supertrend {'UP ✅' if st_up else 'DOWN ❌'} | ATR {fmt(atr_val,2)}"
            )
            st.caption(f"Stop (you set): ${fmt(stp_pct_val,2)} • ATR stop (1.5×): {('$'+fmt(stp_atr,2)) if not np.isnan(stp_atr) else '—'}")
    except Exception:
        st.warning("Unable to analyze this symbol.")

# =============================
# Right: Top-10 Scan + Earnings
# =============================
with right:
    st.subheader("🏆 Top 10 Scan")

    # Force a BLUE label above the timeframe select; collapse the default label to avoid duplicate
    st.markdown("<div class='blue-label'>Scan timeframe (Full mode only)</div>", unsafe_allow_html=True)
    speed_mode = st.radio("Speed mode", ["Quick (Daily)","Full (Intraday)"], index=0, horizontal=True, label_visibility="visible")
    scan_tf = st.selectbox("", ["1m","5m","15m","30m","1h","1D"], index=5, label_visibility="collapsed")

    scan_tf_to_period = {
        "1m": ("1d", "1m"), "5m": ("5d", "5m"), "15m": ("5d", "15m"),
        "30m": ("1mo", "30m"), "1h": ("3mo", "1h"), "1D": ("1y", "1d"),
    }
    max_scan = st.slider("Max symbols to scan", 50, 505, 505, 5)
    run = st.button("🚀 Run Scan", use_container_width=True)

    if run:
        period, interval = (scan_tf_to_period["1D"] if speed_mode.startswith("Quick") else scan_tf_to_period[scan_tf])
        symbols = UNIVERSE[:max_scan]

        st.info(f"Scanning {len(symbols)} symbols on {interval}… (batched)")
        with st.spinner("Scanning…"):
            data_dict = download_batched(symbols, period, interval)

        rows = []
        for sym, df in data_dict.items():
            if df.empty or len(df) < 30:
                continue
            ind = compute_indicators(df)
            last = ind.iloc[-1]
            close = safe_float(last.get("Close"))
            if np.isnan(close) or close <= 0:
                continue

            score10 = score_row(last)
            hz = 20
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], st.session_state.target_gain/100, horizon_bars=hz)
            success_blend = blended_success_pct(hit_rate, score10)

            shares = int(st.session_state.investment_amount // close) if close > 0 else 0
            tgt_price = close * (1 + st.session_state.target_gain/100)
            stop_price = close * (1 - st.session_state.stop_loss/100)
            potential_profit = (tgt_price - close) * shares

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "Score": score10,
                "% Success": round(success_blend, 1) if not np.isnan(success_blend) else None,
                "Buy": round(close, 2),
                "Target": round(tgt_price, 2),
                "Stop": round(stop_price, 2),
                "Shares": shares,
                "Potential $": round(potential_profit, 2),
            })

        if not rows:
            st.warning("No candidates collected. Try Quick (Daily) or scan fewer symbols.")
        else:
            dfres = pd.DataFrame(rows).sort_values(["Score","% Success"], ascending=[False, False]).head(10).reset_index(drop=True)
            st.dataframe(dfres, use_container_width=True)

            pick = st.selectbox("📊 View Top-10 chart:", dfres["Symbol"], index=0)
            tradingview_iframe(pick, TF_MAP[scan_tf], RANGE_MAP[scan_tf], "Dark", height=420)

    st.markdown("### 📅 Earnings Calendar (TradingView, US)")
    tradingview_earnings_widget(height=500)

