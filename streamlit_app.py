import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import traceback

# --- Fallback list of S&P 500 tickers for robustness ---
SP500_FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'TSLA', 'META', 'BRK-B', 'UNH',
    'JPM', 'JNJ', 'V', 'XOM', 'WMT', 'PG', 'LLY', 'MA', 'HD', 'CVX', 'MRK', 'PEP',
    'AVGO', 'KO', 'BAC', 'ORCL', 'COST', 'MCD', 'TMO', 'PFE', 'ACN', 'CSCO', 'ABBV',
    'CRM', 'DHR', 'ADBE', 'WFC', 'NKE', 'DIS', 'TXN', 'NEE', 'PM', 'AMD', 'LIN',
    'UPS', 'RTX', 'MS', 'BMY', 'HON', 'INTC'
]

# --- Core Functions ---
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """Tries to scrape S&P 500 tickers, falls back to a hardcoded list on failure."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception:
        st.warning("Could not fetch live S&P 500 list from Wikipedia. Using a built-in fallback list of 50 stocks.")
        return SP500_FALLBACK_TICKERS

@st.cache_data
def calculate_indicators(df):
    df_copy = df.copy()
    df_copy['SMA200'] = df_copy['Close'].rolling(window=200).mean()
    df_copy['EMA20'] = df_copy['Close'].ewm(span=20, adjust=False).mean()
    df_copy['EMA50'] = df_copy['Close'].ewm(span=50, adjust=False).mean()
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = (gain / loss).replace([np.inf, -np.inf], 999).fillna(0)
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    df_copy.dropna(inplace=True)
    return df_copy

@st.cache_data
def run_backtest(_df):
    df = _df.copy()
    trades = []
    in_position = False
    buy_price = 0
    buy_date = None
    df['EMA_cross_above'] = (df['EMA20'].shift(1) < df['EMA50'].shift(1)) & (df['EMA20'] > df['EMA50'])
    df['EMA_cross_below'] = (df['EMA20'].shift(1) > df['EMA50'].shift(1)) & (df['EMA20'] < df['EMA50'])

    for i in range(len(df)):
        if pd.isna(df['SMA200'].iloc[i]) or pd.isna(df['RSI'].iloc[i]):
            continue
        if in_position:
            if df['Close'].iloc[i] >= buy_price * 1.10 or df['EMA_cross_below'].iloc[i]:
                trades.append({'buy_date': buy_date, 'sell_date': df.index[i], 'buy_price': buy_price, 'sell_price': df['Close'].iloc[i]})
                in_position = False
        elif not in_position:
            if df['EMA_cross_above'].iloc[i] and df['Close'].iloc[i] > df['SMA200'].iloc[i] and df['RSI'].iloc[i] < 70:
                in_position = True
                buy_price = df['Close'].iloc[i]
                buy_date = df.index[i]
    return pd.DataFrame(trades)

def plot_signals(df, trades_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='20-day EMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', name='50-day EMA'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='200-day SMA', line_dash='dash'))
    if not trades_df.empty:
        fig.add_trace(go.Scatter(x=trades_df['buy_date'], y=trades_df['buy_price'], mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=12, name='Buy Signal'))
        fig.add_trace(go.Scatter(x=trades_df['sell_date'], y=trades_df['sell_price'], mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=12, name='Sell Signal'))
    fig.update_layout(title='Stock Price with Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price', legend_title_text='Legend')
    return fig

def display_analysis_results(ticker):
    try:
        data = yf.download(ticker, period="5y", progress=False)
        if data.empty:
            st.error(f"No data downloaded for '{ticker}'. The ticker may be invalid or there could be a temporary network issue.")
            return

        df_with_indicators = calculate_indicators(data)
        trades_df = run_backtest(df_with_indicators)

        st.subheader(f"Backtest Results for {ticker}")
        if not trades_df.empty:
            trades_df['profit'] = (trades_df['sell_price'] - trades_df['buy_price']) / trades_df['buy_price']
            success_rate = (trades_df['profit'] > 0).mean() * 100
            st.metric("Success Rate", f"{success_rate:.2f}%")
            st.metric("Number of Trades Found", f"{len(trades_df)}")
            st.plotly_chart(plot_signals(df_with_indicators, trades_df), use_container_width=True)
        else:
            st.warning("No trades were found for this stock with the current strategy.")
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis for {ticker}:")
        st.code(f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}")

def run_screener(progress_bar, status_text):
    """Scans stocks and returns results and logs."""
    tickers = get_sp500_tickers()
    results, error_log, skip_log = [], [], []

    for i, ticker in enumerate(tickers):
        status_text.text(f"Scanning: {ticker} ({i+1}/{len(tickers)})")
        try:
            data = yf.download(ticker, period="5y", progress=False, timeout=10)
            if len(data) > 252:
                df_indicators = calculate_indicators(data)
                trades_df = run_backtest(df_indicators)
                if len(trades_df) >= 3:
                    trades_df['profit'] = (trades_df['sell_price'] - trades_df['buy_price']) / trades_df['buy_price']
                    success_rate = (trades_df['profit'] > 0).mean() * 100
                    results.append({'Ticker': ticker, 'Success Rate (%)': success_rate, 'Trades': len(trades_df)})
                else:
                    skip_log.append(f"{ticker}: Skipped (Found {len(trades_df)} trades, need 3+)")
            else:
                skip_log.append(f"{ticker}: Skipped (Downloaded {len(data)} rows, need 252+)")
        except Exception as e:
            error_log.append(f"Could not process {ticker}: {e}")
        progress_bar.progress((i + 1) / len(tickers))

    status_text.text("Scan complete!")
    return pd.DataFrame(results), skip_log, error_log

# --- Streamlit App UI ---
st.set_page_config(page_title="Stock Signal App", layout="wide")
st.title("📈 Stock Investment Signal Analyzer")
st.write("This app backtests a trading strategy. This is for educational purposes only and is not financial advice.")

with st.expander("ℹ️ About the Strategy"):
    st.write("Buy: Price > 200-SMA & 20-EMA crosses above 50-EMA & RSI < 70. Sell: 10% profit or 20-EMA crosses below 50-EMA.")

# --- Screener Section ---
st.header("🔍 Stock Screener")
if st.button("Scan For Top Performing Stocks"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    screener_df, skip_log, error_log = run_screener(progress_bar, status_text)

    if not screener_df.empty:
        st.subheader("Top 10 Screener Results")
        st.dataframe(screener_df.sort_values(by='Success Rate (%)', ascending=False).head(10).style.format({'Success Rate (%)': '{:.2f}'}))
    else:
        st.warning("No stocks that met the screening criteria (min. 3 trades) were found.")

    if skip_log:
        with st.expander("Show Skipped Stock Log"):
            st.write(skip_log)
    if error_log:
        with st.expander("Show Errors from Scan"):
            st.write(error_log)

# --- Detailed Analysis Section ---
st.header("📊 Detailed Stock Analysis")
ticker_input = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):", "AAPL").upper()

if st.button("Analyze Ticker"):
    if ticker_input:
        display_analysis_results(ticker_input)
    else:
        st.warning("Please enter a ticker symbol.")
