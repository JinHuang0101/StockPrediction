import streamlit as st
from datetime import date

import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go


START = "2023-01-01"
# format today's date according to the above date format
TODAY = date.today().strftime("%Y-%m-%d")

# app layout
st.write(""" # Stock Prediction Web App """)
st.title("Top 5 Most Actives of the Day")

stocks = ("TSLA", "AMZN", "SIRI", "AAPL", "NIO")
selected_stock = st.selectbox("Select dataset for predictions", stocks)

n_years = st.slider("Years of prediction:", 1, 3)
period = n_years * 365

# load stock data


@st.cache_data  # cache the data
def load_data(ticker):
    # Return data in a Pandas dataframe
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)  # put date in the first column
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forcasting with prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet(
    interval_width=0.95,
    growth="flat",
    seasonality_mode="multiplicative",
    yearly_seasonality=False,
    daily_seasonality=False,
    weekly_seasonality=False,


)

model.fit(df_train)

# future dataframe
future = model.make_future_dataframe(periods=period)

# return predictions in this forecast data
forecast = model.predict(future)

# plot the forecast data
st.subheader('Forecast data')
st.write(forecast.tail())
