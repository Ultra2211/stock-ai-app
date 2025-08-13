# streamlit_app.py
# Dark-mode-only S&P 500 scanner with bulletproof BLACK dropdowns (value + menu via CSS + MutationObserver JS).
# - TradingView chart + ticker search (timeframe controlled on the chart)
# - Expanded indicators: EMA20/50/200, RSI, MACD, Bollinger, Stochastic, ADX(+DI/−DI), MFI, ATR, Supertrend(10,3), OBV
# - 10-point score + blended % Success
# - Top-10 scan (batched) + TradingView earnings calendar (US)
# Educational use only — not financial advice.

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# -----------------------------
# Page / Theme (Dark mode only)
# -----------------------------
st.set_page_config(page_title="S&P 500 Scanner Pro (Dark)", page_icon="📈", layout="wide")

def force_black_selects():
    """Inject CSS + JS to ensure ALL selectboxes (value area + menu) are pure black background with white text."""
    components.html("""
    <style>
      :root {
        --bg:#0e1117; --panel:#161a23; --text:#ffffff; --muted:#c8c8c8;
        --blue:#2563eb; --blue-contrast:#ffffff;
        --border:#2a2f3a; --input:#000000; --menu:#000000; --menu-hover:#11141a;
        --good:#22c55e; --bad:#ef4444;
      }

      html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; color: var(--text) !important; }
      [data-testid="stSidebar"] { background: var(--panel) !important; }

      /* Force WHITE text globally */
      label, .stMarkdown, .stRadio, .stSlider, .stSelectbox, .stNumberInput, .stTextInput,
      div, span, p { color: var(--text) !important; }

      /* Text/number inputs */
      .stTextInput input, .stNumberInput input {
        background: var(--input) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
      }

      /* ====== SELECT VALUE AREA (BaseWeb + Streamlit wrappers) — BLACK background ====== */
      /* Base container */
      div[data-baseweb="select"] > div {
        background: var(--input) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
      }
      /* Any descendant text/icons/placeholder */
      div[data-baseweb="select"] * { color: var(--text) !important; }
      div[data-baseweb="select"] input { color: var(--text) !important; caret-color: var(--text) !important; }
      div[data-baseweb="select"] svg { fill: var(--text) !important; }

      /* Streamlit wrapper roles (covers collapsed labels and different DOM paths) */
      .stSelectbox [role="combobox"], .stSelectbox [role="button"] {
        background: var(--input) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
      }
      .stSelectbox div { color: var(--text) !important; }

      /* ====== DROPDOWN MENU (portal) — BLACK background ====== */
      /* BaseWeb menu portal */
      div[data-baseweb="menu"] {
        background: var(--menu) !important; border: 1px solid var(--border) !important;
      }
      div[data-baseweb="menu"] * { color: var(--text) !important; }
      div[data-baseweb="option"] { background: transparent !important; color: var(--text) !important; }
      div[data-baseweb="option"]:hover,
      div[data-baseweb="option"][aria-selected="true"] { background: var(--menu-hover) !important; }

      /* ARIA listbox fallback (some Streamlit versions) */
      [role="listbox"] { background: var(--menu) !important; border: 1px solid var(--border) !important; }
      [role="listbox"] * { color: var(--text) !important; }
      [role="option"] { background: transparent !important; color: var(--text) !important; }
      [role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--menu-hover) !important; }

      /* Buttons, metrics, badges */
      div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] { color: var(--text) !important; }
      div.stButton > button {
        background: var(--blue) !important; color: var(--blue-contrast) !important; border-color: var(--blue) !important;
      }
      div.stButton > button:hover { filter: brightness(0.95); }
      .ticker-accent { color: var(--blue); font-weight:700; }
      .buy-badge{background:rgba(34,197,94,.15);color:#22c55e;padding:.35rem .6rem;border-radius:999px;font-weight:700;display:inline-block}
      .sell-badge{background:rgba(239,68,68,.15);color:#ef4444;padding:.35rem .6rem;border-radius:999px;font-weight:700;display:inline-block}
      .neutral-badge{background:rgba(37,99,235,.15);color:#2563eb;padding:.35rem .6rem;border-radius:999px;font-weight:700;display:inline-block}
      .blue-label { color: var(--blue) !important; font-weight:700; margin: 0 0 4px 0; }
      .tv-card { border-radius:12px; overflow:hidden; }
    </style>

    <script>
      // MutationObserver: force-style any portaled menus/value containers that appear later
      (function(){
        const darkify = (root) => {
          // Value containers
          root.querySelectorAll('div[data-baseweb="select"] > div, .stSelectbox [role="combobox"], .stSelectbox [role="button"]').forEach(n => {
            n.style.background = '#000';
            n.style.color = '#fff';
            n.style.border = '1px solid #2a2f3a';
          });
          // Menus (BaseWeb + ARIA fallback)
          root.querySelectorAll('div[data-baseweb="menu"], [role="listbox"]').forEach(m => {
            m.style.background = '#000';
            m.style.border = '1px solid #2a2f3a';
            m.querySelectorAll('*').forEach(x => x.style.color = '#fff');
            m.querySelectorAll('div[data-baseweb="option"], [role="option"]').forEach(o => {
              o.addEventListener('mouseenter', () => { o.style.background = '#11141a'; });
              o.addEventListener('mouseleave', () => { o.style.background = 'transparent'; });
            });
          });
        };
        // Initial pass
        darkify(document);
        // Observe DOM for portals
        const obs = new MutationObserver(muts => {
          for (const m of muts) {
            if (!m.addedNodes) continue;
            m.addedNodes.forEach(n => {
              if (n.nodeType === 1) { darkify(n); }
            });
          }
        });
        obs.observe(document.body, { childList: true, subtree: true });
      })();
    </script>
    """, height=0)

force_black_selects()

# -----------------------------
# Helpers
# -----------------------------
def fmt(x, digits=2, default="—"):
    try:
        if pd.isna(x): return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

def tv_symbol(sym: str) -> str:
    return sym.replace("-", ".")

# Chart defaults (users change on TradingView toolbar)
DEFAULT_INTERVAL = "D"
DEFAULT_RANGE = "6M"
TF_MAP = {"1m":"1","5m":"5","15m":"15","30m":"30","1h":"60","1D":"D"}
RANGE_MAP = {"1m":"1D","5m":"5D","15m":"5D","30m":"1M","1h":"3M","1D":"6M"}

# -----------------------------
# Universe
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def sp500_from_yf():
    try:
        return sorted(list(set(yf.tickers_sp500())))
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def sp500_from_wiki_if_available():
    try:
        import lxml  # noqa: F401
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        return sorted(df["Symbol"].astype(str).str.replace(r"\.", "-", regex=True).unique().tolist())
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def build_universe(extra_csv: str):
    base = set(sp500_from_yf())
    wiki = set(sp500_from_wiki_if_available())
    universe = sorted(list(base.union(wiki)))
    if not universe:
        universe = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","BRK-B","JPM"]
    extras = [s.strip().upper() for s in (extra_csv or "").split(",") if s.strip()]
    return sorted(list(set(universe + extras)))

# -----------------------------
# Indicators
# -----------------------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(s, period=14):
    delta = s.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(s, fast=12, slow=26, signal=9):
    f = ema(s, fast); sl = ema(s, slow)
    line = f - sl; sig = ema(line, signal)
    return line, sig, line - sig

def bollinger(s, length=20, mult=2.0):
    mid = s.rolling(length).mean(); std = s.rolling(length).std()
    up = mid + mult*std; lo = mid - mult*std
    pctb = (s - lo) / (up - lo)
    return mid, up, lo, pctb

def stochastic(h, l, c, k=14, d=3):
    lo = l.rolling(k).min(); hi = h.rolling(k).max()
    kpct = 100*(c - lo) / (hi - lo + 1e-12)
    return kpct, kpct.rolling(d).mean()

def true_range(df):
    pc = df["Close"].shift(1)
    return pd.concat([(df["High"]-df["Low"]).abs(), (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)

def atr(df, period=14): return true_range(df).ewm(alpha=1/period, adjust=False).mean()

def adx(df, period=14):
    up = df["High"].diff(); dn = -df["Low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0); minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_sm = true_range(df).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100*pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()/(tr_sm+1e-12)
    minus_di = 100*pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()/(tr_sm+1e-12)
    dx = 100*(abs(plus_di - minus_di)/(plus_di + minus_di + 1e-12))
    return plus_di.rename("PLUS_DI"), minus_di.rename("MINUS_DI"), dx.ewm(alpha=1/period, adjust=False).mean().rename("ADX")

def mfi(df, period=14):
    tp = (df["High"]+df["Low"]+df["Close"])/3.0
    mf = tp*df["Volume"]
    pos = np.where(tp > tp.shift(1), mf, 0.0); neg = np.where(tp < tp.shift(1), mf, 0.0)
    mfr = pd.Series(pos).rolling(period).sum() / (pd.Series(neg).rolling(period).sum() + 1e-12)
    return 100 - (100/(1+mfr))

def supertrend(df, period=10, mult=3.0):
    a = atr(df, period); hl2 = (df["High"]+df["Low"])/2.0
    up = hl2 + mult*a; lo = hl2 - mult*a
    fu, fl = up.copy(), lo.copy()
    uptrend = pd.Series(index=df.index, dtype=bool)
    for i in range(len(df)):
        if i == 0:
            uptrend.iloc[i] = True
            continue
        if df["Close"].iloc[i-1] > fu.iloc[i-1]: uptrend.iloc[i] = True
        elif df["Close"].iloc[i-1] < fl.iloc[i-1]: uptrend.iloc[i] = False
        else:
            uptrend.iloc[i] = uptrend.iloc[i-1]
            if uptrend.iloc[i] and lo.iloc[i] < fl.iloc[i-1]: lo.iloc[i] = fl.iloc[i-1]
            if (not uptrend.iloc[i]) and up.iloc[i] > fu.iloc[i-1]: up.iloc[i] = fu.iloc[i-1]
        fu.iloc[i] = up.iloc[i]; fl.iloc[i] = lo.iloc[i]
    return fu.rename("ST_UPPER"), fl.rename("ST_LOWER"), uptrend.rename("ST_UPTREND"), a.rename("ATR")

def on_balance_volume(df):
    return (np.sign(df["Close"].diff()).fillna(0.0) * df["Volume"]).cumsum().rename("OBV")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20); df["EMA50"] = ema(df["Close"], 50); df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    macd_line, sig, hist = macd(df["Close"]); df["MACD"] = macd_line; df["MACD_SIGNAL"] = sig; df["MACD_HIST"] = hist
    bb_mid, bb_up, bb_lo, bb_pctb = bollinger(df["Close"], 20, 2.0)
    df["BB_MID"] = bb_mid; df["BB_UP"] = bb_up; df["BB_LOW"] = bb_lo; df["BB_PCTB"] = bb_pctb
    k, d = stochastic(df["High"], df["Low"], df["Close"], 14, 3); df["STO_K"] = k; df["STO_D"] = d
    plus_di, minus_di, adx_v = adx(df, 14); df["PLUS_DI"] = plus_di; df["MINUS_DI"] = minus_di; df["ADX"] = adx_v
    df["MFI14"] = mfi(df, 14)
    st_u, st_l, st_up, atr_v = supertrend(df, 10, 3.0); df["ST_UPPER"] = st_u; df["ST_LOWER"] = st_l; df["ST_UPTREND"] = st_up; df["ATR14"] = atr_v
    df["OBV"] = on_balance_volume(df)
    return df

# -----------------------------
# Score + success
# -----------------------------
def score_row(row):
    s = 0
    c = float(row.get("Close", np.nan))
    if float(row.get("EMA20", np.nan)) > float(row.get("EMA50", np.nan)): s += 1
    if float(row.get("EMA50", np.nan)) > float(row.get("EMA200", np.nan)): s += 1
    if float(row.get("MACD", np.nan)) > float(row.get("MACD_SIGNAL", np.nan)): s += 1
    r = float(row.get("RSI14", np.nan));  s += 1 if 40 < r < 65 else 0
    if c > float(row.get("BB_MID", np.nan)): s += 1
    pctb = float(row.get("BB_PCTB", np.nan)); s += 1 if 0.2 <= pctb <= 0.8 else 0
    if float(row.get("STO_K", np.nan)) > float(row.get("STO_D", np.nan)) and float(row.get("STO_K", np.nan)) < 80: s += 1
    if float(row.get("ADX", np.nan)) > 20 and float(row.get("PLUS_DI", np.nan)) > float(row.get("MINUS_DI", np.nan)): s += 1
    mfi14 = float(row.get("MFI14", np.nan)); s += 1 if 35 < mfi14 < 75 else 0
    if bool(row.get("ST_UPTREND", False)): s += 1
    return s

def forward_hit_rate_for_target(close: pd.Series, target_pct: float, horizon_bars: int):
    try:
        arr = close.values; n = len(arr)
        if n < horizon_bars + 5: return np.nan, 0
        fmax = np.full(n, np.nan)
        for i in range(n - horizon_bars):
            fmax[i] = np.max(arr[i+1:i+1+horizon_bars])
        base = arr[:-horizon_bars]; mx = fmax[:-horizon_bars]
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = (mx - base) / (base + 1e-12)
        m = ~np.isnan(fwd)
        if m.sum() == 0: return np.nan, 0
        hits = (fwd[m] >= target_pct).sum(); total = m.sum()
        return 100.0 * hits / total, int(total)
    except Exception:
        return np.nan, 0

def blended_success_pct(hitrate_pct: float, score10: int) -> float:
    a = 0 if hitrate_pct is None or np.isnan(hitrate_pct) else float(hitrate_pct)
    b = max(0, min(10, int(score10))) * 10.0
    return (a + b) / 2.0

# -----------------------------
# Downloads
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
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
                        if not sub.empty and "Close" in sub: out[sym] = sub
            else:
                if not df.empty and "Close" in df: out[chunk[0]] = df.dropna().copy()
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
        "&toolbarbg=f1f3f6&theme=dark&style=1&timezone=Etc/UTC&withdateranges=1"
        "&allow_symbol_change=1&details=1&hideideas=1"
    )
    components.html(
        f'<iframe class="tv-card" src="{src}" width="100%" height="{height}" frameborder="0" allowtransparency="true" scrolling="no"></iframe>',
        height=height, scrolling=False
    )

def tradingview_earnings_widget(height: int = 500):
    cfg = {"width":"100%","height":height,"colorTheme":"dark","isTransparent":False,"locale":"en","market":"us"}
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
# Sidebar (globals)
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
# Left: Search & Chart
# =============================
with left:
    st.subheader("🔎 Search & Chart")
    manual_symbol = st.text_input("Type a ticker and press Enter", value="AMD", key="manual_symbol").strip().upper()
    dropdown_symbol = st.selectbox("…or pick from S&P 500", UNIVERSE, index=min(UNIVERSE.index("AAPL") if "AAPL" in UNIVERSE else 0, len(UNIVERSE)-1))
    symbol = manual_symbol or dropdown_symbol

    tradingview_iframe(symbol, "D", "6M", "Dark", height=520)

    try:
        with st.spinner("Fetching chart data…"):
            df_m = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        if df_m is None or df_m.empty:
            st.warning("No Yahoo data for this symbol.")
        else:
            ind = compute_indicators(df_m)
            last = ind.iloc[-1]
            price = float(last["Close"])

            score10 = score_row(last)
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], st.session_state.target_gain/100, horizon_bars=st.session_state.horizon_bars)
            success_blend = blended_success_pct(hit_rate, score10)

            if score10 >= 7 and (not np.isnan(success_blend) and success_blend >= 55):
                signal_badge = '<span class="buy-badge">BUY</span>'
            elif score10 <= 3 and (not np.isnan(success_blend) and success_blend < 45):
                signal_badge = '<span class="sell-badge">SELL</span>'
            else:
                signal_badge = '<span class="neutral-badge">NEUTRAL</span>'

            tgt = price * (1 + st.session_state.target_gain/100)
            stp_pct_val = price * (1 - st.session_state.stop_loss/100)
            atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
            stp_atr = price - 1.5 * atr_val if not np.isnan(atr_val) else np.nan
            stp_use = stp_pct_val
            rr = (tgt - price) / max(price - stp_use, 1e-9)

            ema20 = float(last["EMA20"]); ema50 = float(last["EMA50"]); ema200 = float(last["EMA200"])
            rsi14 = float(last["RSI14"]); macd_spread = float(last["MACD"]) - float(last["MACD_SIGNAL"])
            pctb = float(last["BB_PCTB"])
            adx_v = float(last["ADX"]); di_plus = float(last["PLUS_DI"]); di_minus = float(last["MINUS_DI"])
            sto_k = float(last["STO_K"]); sto_d = float(last["STO_D"]); mfi14 = float(last["MFI14"])
            st_up = bool(last["ST_UPTREND"])

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

    # BLUE label above the timeframe select; collapse the default label to avoid duplicate
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
            close = float(last["Close"])
            if close <= 0:
                continue

            score10 = score_row(last)
            hit_rate, samples = forward_hit_rate_for_target(ind["Close"], st.session_state.target_gain/100, horizon_bars=20)
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
            # Ensure MutationObserver has already initialized before a new portal opens
            components.html("", height=0)
            tradingview_iframe(pick, TF_MAP[scan_tf], RANGE_MAP[scan_tf], "Dark", height=420)

    st.markdown("### 📅 Earnings Calendar (TradingView, US)")
    tradingview_earnings_widget(height=500)
