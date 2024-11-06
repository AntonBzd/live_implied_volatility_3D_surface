import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.interpolate import griddata
import plotly.graph_objects as go
import scipy.optimize as optimize



st.title('Implied Volatility Surface')


### FUNCTIONS ###

def black_scholes_price(S, K, T, r, sigma, q=0.0, option_type='call'):
    d1 = (np.log(S/K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def implied_volatility(S, K, T, r, price, q=0, option_type = 'call'):

    def objective_function(sigma):
        return black_scholes_price(S, K, T, r, sigma, q, option_type) - price

    implied_vol = optimize.root(objective_function, 0.2)
    return implied_vol.x[0] if implied_vol.success else np.nan

def calculate_greeks(S, K, T, r, sigma, q=0.0, option_type='call'):
    d1 = (np.log(S/K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = np.exp(-q * T) * norm.cdf(d1) if option_type == 'call' else -np.exp(-q * T) * norm.cdf(-d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else
             r * K * np.exp(-r * T) * norm.cdf(-d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    return delta, gamma, vega, theta, rho


### STREAMLIT INTERFACE

tickers = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "IWM": "iShares Russell 2000 ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF Trust",
    "XLF": "Financial Select Sector SPDR Fund",
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "NVDA": "NVIDIA Corporation",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "EWJ": "iShares MSCI Japan ETF",
    "EWQ": "iShares MSCI France ETF",
    "EFA": "iShares MSCI EAFE ETF",
    "FXI": "iShares China Large-Cap ETF",
    "EWZ": "iShares MSCI Brazil ETF",
}
st.sidebar.header('Ticker Symbol')
ticker_symbol = st.sidebar.selectbox('Choose a ticker', options=list(tickers.keys()), format_func=lambda ticker: f"{ticker} - {tickers[ticker]}",index=0 )


st.sidebar.header('Parameters for the Black-Scholes model.')
option_type = st.sidebar.radio("Option Type", ("call", "put"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (0.01 for 1.0%)",value=0.005, step=0.01, format="%.3f")
dividend_yield = st.sidebar.number_input("Dividend Yield (0.01 for 1.0%)",value=0.013, step=0.01, format="%.3f")

st.sidebar.header('Strike Price Filter')
min_strike_pct = st.sidebar.slider("Minimum Strike Price (% of Spot Price)", min_value=0, max_value=200, value=80)
max_strike_pct = st.sidebar.slider("Maximum Strike Price (% of Spot Price)", min_value=0, max_value=200, value=120)
if min_strike_pct >= max_strike_pct:
    st.sidebar.error('Minimum percentage must be less than maximum percentage.')
    st.stop()


### DATA ###

ticker = yf.Ticker(ticker_symbol)
today = pd.Timestamp('today').normalize()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f'Error fetching options for {ticker_symbol}: {e}')
    st.stop()

exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]

if not exp_dates:
    st.error(f'No available option expiration dates for {ticker_symbol}.')
else:
    option_data = []

    for exp_date in exp_dates:
        try:
            opt_chain = ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
            calls = opt_chain.calls
            puts = opt_chain.puts
        except Exception as e:
            st.warning(f'Failed to fetch option chain for {exp_date.date()}: {e}')
            continue

        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        puts = puts[(puts['bid'] > 0) & (puts['ask'] > 0)]

        iter_option = calls.iterrows() if option_type == "call" else puts.iterrows()

        for index, row in iter_option :
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            option_data.append({
                'expirationDate': exp_date,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })

    if not option_data:
        st.error('No option data available after filtering.')
    else:
        options_df = pd.DataFrame(option_data)

        try:
            spot_history = ticker.history(period='5d')
            if spot_history.empty:
                st.error(f'Failed to retrieve spot price data for {ticker_symbol}.')
                st.stop()
            else:
                spot_price = spot_history['Close'].iloc[-1]
        except Exception as e:
            st.error(f'An error occurred while fetching spot price data: {e}')
            st.stop()

        options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

        options_df = options_df[
            (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
            (options_df['strike'] <= spot_price * (max_strike_pct / 100))
        ]

        options_df.reset_index(drop=True, inplace=True)

        with st.spinner('Calculating implied volatility...'):
            options_df['impliedVolatility'] = options_df.apply(
                lambda row: implied_volatility(
                    price=row['mid'],
                    S=spot_price,
                    K=row['strike'],
                    T=row['timeToExpiration'],
                    r=risk_free_rate,
                    q=dividend_yield
                ), axis=1
            )


        with st.spinner('Calculating the Greeks...'):
          options_df["delta"], options_df["gamma"], options_df["vega"], options_df["theta"], options_df["rho"] =  calculate_greeks(
                            S=spot_price,
                            K=options_df['strike'],
                            T=options_df['timeToExpiration'],
                            r=risk_free_rate,
                            sigma = options_df["impliedVolatility"],
                            q=dividend_yield,
                            option_type= option_type)


        options_df.dropna(subset=['impliedVolatility'], inplace=True)
        options_df['impliedVolatility'] *= 100
        options_df.sort_values('strike', inplace=True)


### FIGURE ###

        X = options_df['timeToExpiration'].values
        Y = options_df['strike'].values
        Z = options_df['impliedVolatility'].values
        ti = np.linspace(X.min(), X.max(), 50)
        ki = np.linspace(Y.min(), Y.max(), 50)
        T, K = np.meshgrid(ti, ki)
        Zi = griddata((X, Y), Z, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))


        fig = go.Figure(data=[go.Surface(
            x=T, y=K, z=Zi,colorscale='Viridis',
            colorbar_title='Implied Volatility (%)'
        )])

        fig.update_layout(
            title=f'Implied Volatility Surface for {ticker_symbol} Options (Last Spot Price : {round(spot_price, 2)})',
            scene=dict(xaxis_title='Time to Expiration (years)', yaxis_title='Strike Price ($)', zaxis_title='Implied Volatility (%)'),
            autosize=False,
            width=700,
            height=700,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        st.plotly_chart(fig)

        # Greeks
        Zi = griddata((X, Y), options_df["delta"].values, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))
        fig_delta = go.Figure(data=[go.Surface(x=T, y=K, z=Zi,colorscale='Viridis')])
        fig_delta.update_layout(title="Delta",
            scene=dict(xaxis_title='Time to Expiration (years)', yaxis_title='Strike Price ($)', zaxis_title='Delta'),
            autosize=False)

        Zi = griddata((X, Y), options_df["gamma"].values, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))
        fig_gamma = go.Figure(data=[go.Surface(x=T, y=K, z=Zi,colorscale='Viridis')])
        fig_gamma.update_layout(title="Gamma",
            scene=dict(xaxis_title='Time to Expiration (years)', yaxis_title='Strike Price ($)', zaxis_title='Gamma'),
            autosize=False)

        Zi = griddata((X, Y), options_df["vega"].values, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))
        fig_vega = go.Figure(data=[go.Surface(x=T, y=K, z=Zi,colorscale='Viridis')])
        fig_vega.update_layout(title="Vega",
            scene=dict(xaxis_title='Time to Expiration (years)', yaxis_title='Strike Price ($)', zaxis_title='Vega'), autosize=False)

        Zi = griddata((X, Y), options_df["theta"].values, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))
        fig_theta = go.Figure(data=[go.Surface(x=T, y=K, z=Zi,colorscale='Viridis')])
        fig_theta.update_layout(title="Theta", scene=dict(xaxis_title='Time to Expiration (years)', yaxis_title='Strike Price ($)', zaxis_title='Theta'), autosize=False)

        Zi = griddata((X, Y), options_df["rho"].values, (T, K), method='linear')
        Zi = np.ma.array(Zi, mask=np.isnan(Zi))
        fig_rho = go.Figure(data=[go.Surface(x=T, y=K, z=Zi,colorscale='Viridis')])
        fig_rho.update_layout(title="Rho", scene=dict(xaxis_title='Time to Expiration (years)', yaxis_title='Strike Price ($)', zaxis_title='Rho'), autosize=False)


        st.write(f'Greek surfaces for {ticker_symbol} Options')
        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_delta, use_container_width=True)
        col2.plotly_chart(fig_gamma, use_container_width=True)
        col1.plotly_chart(fig_vega, use_container_width=True)
        col2.plotly_chart(fig_theta, use_container_width=True)
        col1.plotly_chart(fig_rho, use_container_width=True)

        
        st.write("---")
        st.markdown(
            "Created by Antonin Bezard  |   [LinkedIn](https://www.linkedin.com/in/antonin-bezard-a11511177/)"
        )


# Original version : https://github.com/MateuszJastrzebski21/Implied-Volatility-Surface/blob/master/volatility_surface.py
