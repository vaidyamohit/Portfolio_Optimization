import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from scipy.optimize import minimize
import time

# Function to download stock data with retries and caching using Streamlit's cache
@st.cache_data
def download_data_with_retry(ticker, start_date, end_date, retries=5, delay=5):
    for i in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                return data
        except Exception as e:
            time.sleep(delay)
    return pd.DataFrame()  # Return empty DataFrame if all retries fail

# List of NIFTY 50 companies and their sectors
nifty50_stocks = {
    'RELIANCE.NS': 'Energy', 'TCS.NS': 'IT', 'HDFCBANK.NS': 'Financials', 'INFY.NS': 'IT',
    'ICICIBANK.NS': 'Financials', 'HINDUNILVR.NS': 'Consumer Goods', 'ITC.NS': 'Consumer Goods',
    'KOTAKBANK.NS': 'Financials', 'SBIN.NS': 'Financials', 'LT.NS': 'Infrastructure',
    'BHARTIARTL.NS': 'Telecom', 'AXISBANK.NS': 'Financials', 'HDFC.NS': 'Financials',
    'BAJFINANCE.NS': 'Financials', 'ASIANPAINT.NS': 'Consumer Goods', 'MARUTI.NS': 'Automobile',
    'HCLTECH.NS': 'IT', 'WIPRO.NS': 'IT', 'ULTRACEMCO.NS': 'Cement', 'ONGC.NS': 'Energy',
    'TITAN.NS': 'Consumer Goods', 'BAJAJFINSV.NS': 'Financials', 'SUNPHARMA.NS': 'Pharmaceuticals',
    'NTPC.NS': 'Energy', 'NESTLEIND.NS': 'Consumer Goods', 'POWERGRID.NS': 'Energy',
    'TATAMOTORS.NS': 'Automobile', 'M&M.NS': 'Automobile', 'ADANIENT.NS': 'Conglomerate',
    'INDUSINDBK.NS': 'Financials', 'HINDALCO.NS': 'Metals', 'JSWSTEEL.NS': 'Metals',
    'DIVISLAB.NS': 'Pharmaceuticals', 'DRREDDY.NS': 'Pharmaceuticals', 'GRASIM.NS': 'Cement',
    'BPCL.NS': 'Energy', 'BRITANNIA.NS': 'Consumer Goods', 'CIPLA.NS': 'Pharmaceuticals',
    'HEROMOTOCO.NS': 'Automobile', 'COALINDIA.NS': 'Metals', 'EICHERMOT.NS': 'Automobile',
    'APOLLOHOSP.NS': 'Healthcare', 'TATACONSUM.NS': 'Consumer Goods', 'SBILIFE.NS': 'Insurance',
    'ICICIGI.NS': 'Insurance', 'HDFCLIFE.NS': 'Insurance', 'TECHM.NS': 'IT', 'BAJAJ-AUTO.NS': 'Automobile',
    'DABUR.NS': 'Consumer Goods'
}

# Parameters
market_ticker = '^NSEI'
risk_free_rate = 0.0677  # Risk-free rate (e.g., 6.77%)
start_date_default = '2020-01-01'
end_date_default = '2024-01-01'

# Streamlit app layout
st.title("NIFTY 50 CAPM & Industry-wise Analysis Dashboard")

# Sidebar inputs
st.sidebar.header("User Input")
selected_sector = st.sidebar.selectbox('Select Sector:', list(set(nifty50_stocks.values())), index=0)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start_date_default))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end_date_default))

# Validate dates
if start_date > end_date:
    st.error("End date must be after the start date.")
    st.stop()

# Investment and Risk input
investment_amount = st.sidebar.number_input("Investment Amount (₹):", min_value=1000, value=100000, step=1000)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0-1):", 0.0, 1.0, 0.5)

# Download market data
market_data = download_data_with_retry(market_ticker, start_date, end_date)

# Filter stocks by selected sector
stocks = [stock for stock, sector in nifty50_stocks.items() if sector == selected_sector]

stock_data_dict = {}
for stock in stocks:
    stock_data = download_data_with_retry(stock, start_date, end_date)
    if not stock_data.empty:
        stock_data_dict[stock] = stock_data

# Continue if data is available
if stock_data_dict:
    stock_data = pd.DataFrame({ticker: stock['Adj Close'] for ticker, stock in stock_data_dict.items()})
    stock_returns = stock_data.pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()

    # Align stock and market data
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner', axis=0)

    if not stock_returns.empty and not market_returns.empty:
        # Calculate mean returns and covariance matrix
        mean_returns = stock_returns.mean() * 252
        cov_matrix = stock_returns.cov() * 252

        # Optimization setup for portfolio
        def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return portfolio_return, portfolio_volatility, sharpe_ratio

        def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            return -portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]

        num_stocks = len(mean_returns)
        initial_weights = num_stocks * [1.0 / num_stocks]
        bounds = tuple((0, 1) for _ in range(num_stocks))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        # Optimize portfolio
        optimized_result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                                    method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized_result.x

        # Display portfolio performance
        st.subheader("Optimal Portfolio Weights")
        portfolio_df = pd.DataFrame({'Stock': mean_returns.index, 'Weight': optimal_weights})
        st.write(portfolio_df)

        # Efficient frontier
        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
        efficient_frontier = []
        for target_return in target_returns:
            try:
                ef_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                  {'type': 'eq', 'fun': lambda x: portfolio_statistics(x, mean_returns, cov_matrix, risk_free_rate)[0] - target_return}]
                result = minimize(lambda x: portfolio_statistics(x, mean_returns, cov_matrix, risk_free_rate)[1], initial_weights,
                                  method='SLSQP', bounds=bounds, constraints=ef_constraints)
                efficient_frontier.append(result['fun'])
            except Exception as e:
                st.error(f"Error in optimization for return {target_return}: {str(e)}")
                continue

        # Plot Efficient Frontier
        st.subheader("Efficient Frontier")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=efficient_frontier, y=target_returns, mode='lines', name='Efficient Frontier'))
        st.plotly_chart(fig)

        # Suggestion based on risk tolerance and Sharpe ratio
        sharpe_ratio = portfolio_statistics(optimal_weights, mean_returns, cov_matrix, risk_free_rate)[2]
        if sharpe_ratio > risk_tolerance:
            st.write("The portfolio has a good risk-adjusted return. Consider investing.")
        else:
            st.write("The portfolio may not meet your risk-adjusted return expectations. Consider adjusting your allocation.")

else:
    st.write("No stock data available for the selected sector.")

# Add Awesome Oscillator for trends
def awesome_oscillator(stock_data):
    # Calculate Awesome Oscillator using the high and low prices
    median_price = (stock_data['High'] + stock_data['Low']) / 2
    ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
    return ao

for stock, data in stock_data_dict.items():
    ao = awesome_oscillator(data)
    st.subheader(f"Awesome Oscillator for {stock}")
    st.line_chart(ao)
