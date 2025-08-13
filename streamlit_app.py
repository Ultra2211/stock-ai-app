import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="S&P 500 AI Scanner", layout="wide")

TIMEFRAMES = {
    "1 minute": ("1m", "1d"),
    "5 minutes": ("5m", "5d"),
    "15 minutes": ("15m", "5d"),
    "30 minutes": ("30m", "1mo"),
    "1 hour": ("1h", "3mo"),
    "1 day": ("1d", "6mo"),
    "1 week": ("1wk", "2y"),
    "1 month": ("1mo", "5y"),
    "3 months": ("3mo", "10y"),
    "6 months": ("6mo", "20y")
}

# ----------------------
# FUNCTIONS
# ----------------------
@st.cache_data
def get_sp500_symbols():
    """Get S&P 500 stock symbols from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    return df['Symbol'].tolist()

def get_indicators(df):
    """Calculate SMA, EMA, MACD, RSI."""
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    return df

def score_stock(df):
    """Score stock based on indicators."""
    latest = df.iloc[-1]
    score = 0
    if latest['RSI14'] < 40: score += 1
    if latest['MACD_Hist'] > 0: score += 1
    if latest['Close'] > latest['SMA50']: score += 1
    return score, latest

# ----------------------
# APP UI
# ----------------------
st.title("📈 S&P 500 AI Stock Scanner with Targets")

tf_label = st.selectbox("Select Timeframe", list(TIMEFRAMES.keys()), index=1)
interval, period = TIMEFRAMES[tf_label]

max_symbols = st.slider("Max Symbols to Scan", 50, 500, 200, step=50)

if st.button("🚀 Run S&P 500 SCAN"):
    symbols = get_sp500_symbols()[:max_symbols]
    candidates = []

    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        try:
            data = yf.download(symbol, interval=interval, period=period, progress=False)
            if data.empty:
                continue
            data = get_indicators(data)
            score, latest = score_stock(data)
            buy_price = latest["Close"]
            target_price = buy_price * 1.10  # +10% target
            stop_loss = buy_price * 0.95     # -5% stop loss

            candidates.append({
                "Symbol": symbol,
                "Score": score,
                "RSI14": latest["RSI14"],
                "MACD_Hist": latest["MACD_Hist"],
                "SMA50": latest["SMA50"],
                "Close": buy_price,
                "Buy/Sell": "BUY" if score >= 2 else "SELL",
                "Target Price": target_price,
                "Stop Loss": stop_loss
            })
        except Exception:
            continue
        progress.progress((i+1)/len(symbols))

    df_candidates = pd.DataFrame(candidates)
    df_candidates = df_candidates.sort_values(by="Score", ascending=False).head(10)

    if df_candidates.empty:
        st.error("No stocks found. Try increasing Max symbols or changing timeframe.")
    else:
        st.success(f"Top {len(df_candidates)} Candidates Found!")
        st.dataframe(df_candidates, use_container_width=True)

        for _, row in df_candidates.iterrows():
            st.subheader(f"{row['Symbol']} — {row['Buy/Sell']}")
            st.metric("Price", f"${row['Close']:.2f}")
            st.metric("RSI(14)", f"{row['RSI14']:.1f}")
            st.metric("MACD Hist", f"{row['MACD_Hist']:.3f}")
            st.metric("Target Price", f"${row['Target Price']:.2f}")
            st.metric("Stop Loss", f"${row['Stop Loss']:.2f}")
            st.markdown(
                f'<iframe src="https://s.tradingview.com/widgetembed/?symbol={row["Symbol"]}&interval={interval}&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=light&style=1&timezone=Etc/UTC&studies_overrides={{}}" '
                'width="100%" height="500" frameborder="0"></iframe>',
                unsafe_allow_html=True
            )




