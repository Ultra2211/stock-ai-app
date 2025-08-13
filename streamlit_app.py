import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# --- Core Functions ---

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        tables = pd.read_html(url)
        return tables[0]['Symbol'].tolist()
    except Exception as e:
        st.error(f"Failed to load S&P 500 tickers: {e}")
        return []

@st.cache_data
def calculate_indicators(df):
    df_copy = df.copy()
    df_copy['SMA200'] = df_copy['Close'].rolling(window=200).mean()
    df_copy['EMA20'] = df_copy['Close'].ewm(span=20, adjust=False).mean()
    df_copy['EMA50'] = df_copy['Close'].ewm(span=50, adjust=False).mean()
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
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
        # Ensure we don't have NaN values for our conditions
        is_ema_cross_below = df['EMA_cross_below'].iloc[i]
        is_ema_cross_above = df['EMA_cross_above'].iloc[i]
        current_close = df['Close'].iloc[i]
        current_sma200 = df['SMA200'].iloc[i]
        current_rsi = df['RSI'].iloc[i]

        if np.isnan(current_sma200) or np.isnan(current_rsi):
            continue

        if in_position:
            if current_close >= buy_price * 1.10 or is_ema_cross_below:
                sell_price = current_close
                trades.append({'buy_date': buy_date, 'sell_date': df.index[i], 'buy_price': buy_price, 'sell_price': sell_price, 'profit': (sell_price - buy_price) / buy_price})
                in_position = False
        # Use elif to prevent selling and buying on the same day
        elif not in_position:
            if is_ema_cross_above and current_close > current_sma200 and current_rsi < 70:
                in_position = True
                buy_price = current_close
                buy_date = df.index[i]

    if not trades:
        return pd.DataFrame(), df

    return pd.DataFrame(trades), df

def plot_signals(df, trades_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], mode='lines', name='20-day EMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], mode='lines', name='50-day EMA', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], mode='lines', name='200-day SMA', line=dict(color='gray', dash='dash')))

    # Use the trades_df directly for plotting, it only contains closed trades
    if not trades_df.empty:
        fig.add_trace(go.Scatter(x=trades_df['buy_date'], y=trades_df['buy_price'], mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=12, name='Buy Signal'))
        fig.add_trace(go.Scatter(x=trades_df['sell_date'], y=trades_df['sell_price'], mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=12, name='Sell Signal'))

    fig.update_layout(title='Stock Price with Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price', legend_title='Legend', template='plotly_dark')
    return fig

def display_analysis_results(ticker):
    try:
        data = yf.download(ticker, period="5y", progress=False)
        if data.empty:
            st.error(f"No data found for ticker {ticker}.")
            return

        with st.spinner(f"Analyzing {ticker}..."):
            df_indicators = calculate_indicators(data)
            trades_df, df_with_signals = run_backtest(df_indicators)

            st.subheader(f"Backtest Results for {ticker}")
            if not trades_df.empty:
                success_rate = (trades_df['profit'] > 0).mean() * 100
                col1, col2 = st.columns(2)
                col1.metric("Success Rate", f"{success_rate:.2f}%")
                col2.metric("Number of Trades Found", len(trades_df))
                st.plotly_chart(plot_signals(df_with_signals, trades_df), use_container_width=True)
            else:
                st.warning("No trades were found for this stock with the current strategy.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Streamlit App UI ---
st.set_page_config(page_title="Stock Signal App", layout="wide")
st.title("📈 Stock Investment Signal Analyzer")
st.write("This app backtests a trading strategy and is for educational purposes only. It does not provide financial advice.")

if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = ''

with st.expander("ℹ️ About the Strategy"):
    st.write("""
    - **Buy Signal:** Price > 200-day SMA AND 20-day EMA crosses above 50-day EMA AND RSI < 70.
    - **Sell Signal:** 10% profit OR 20-day EMA crosses below 50-day EMA.
    """)

# --- Screener Section ---
st.header("🔍 S&P 500 Stock Screener")
if st.button("Scan S&P 500 for Top 10 Stocks"):
    tickers = get_sp500_tickers()
    if tickers:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(tickers):
            status_text.text(f"Scanning: {ticker} ({i+1}/{len(tickers)})")
            try:
                data = yf.download(ticker, period="5y", progress=False, timeout=5)
                if len(data) > 252:
                    df_indicators = calculate_indicators(data)
                    trades_df, _ = run_backtest(df_indicators)
                    if not trades_df.empty and len(trades_df) >= 3:
                        success_rate = (trades_df['profit'] > 0).mean() * 100
                        results.append({'Ticker': ticker, 'Success Rate (%)': success_rate, 'Trades': len(trades_df)})
            except Exception:
                pass
            progress_bar.progress((i + 1) / len(tickers))

        status_text.text("Scan complete!")
        progress_bar.empty()

        if results:
            screener_df = pd.DataFrame(results).sort_values(by='Success Rate (%)', ascending=False).head(10)
            st.session_state.screener_results = screener_df
        else:
            st.warning("No stocks met the screening criteria.")

if 'screener_results' in st.session_state and st.session_state.screener_results is not None:
    st.subheader("Top 10 Screener Results")
    for index, row in st.session_state.screener_results.iterrows():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
        col1.text(row['Ticker'])
        col2.text(f"Success: {row['Success Rate (%)']:.2f}%")
        col3.text(f"Trades: {row['Trades']}")
        if col4.button("View Details", key=f"details_{row['Ticker']}"):
            st.session_state.selected_ticker = row['Ticker']

# --- Detailed Analysis Section ---
st.header("📊 Detailed Stock Analysis")
ticker_input = st.text_input("Enter a stock ticker:", value=st.session_state.selected_ticker, key="ticker_input").upper()

if st.button("Analyze Ticker", key="analyze_button"):
    if ticker_input:
        display_analysis_results(ticker_input)
    else:
        st.warning("Please enter a ticker symbol.")

# If a ticker was selected from the screener, run the analysis automatically once
if st.session_state.selected_ticker and st.session_state.selected_ticker != st.session_state.get('last_analyzed', ''):
    st.session_state.last_analyzed = st.session_state.selected_ticker
    display_analysis_results(st.session_state.selected_ticker)
