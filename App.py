import streamlit as st
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import json

today = date.today()

interval = {
    "1d": 1,
    "3d": 3,
    "1w": 7,
    "1m": 30,
    "3m": 90
}

with open("popular_stocks.json", "r") as file:
    popular_stocks_json = json.load(file)

popular_stocks = pd.DataFrame(popular_stocks_json.keys())

st.title("🚀 Stock Prediction App ARIMA 📈")
st.markdown("Powered by ARIMA time series modeling, this app provides predictive analytics for major NASDAQ and NYSE-listed stocks, helping users gain insight into potential future price trends.")

selected_ticker = st.selectbox("Select Stock Ticker", popular_stocks, index=0)
selected_time = st.selectbox("Select Forecasting Interval", interval, index=0)


if st.button("Analyze"):

    def stock_info(ticker_name, interval):

        ticker_name = ticker_name
        ticker = yf.download(ticker_name, start="2020-01-01", end=today)
        df_close = ticker["Close"]

        order_aic_bic = []
        for p in range(3):
            for q in range(3):
                model = ARIMA(df_close, order=(p,0,q))
                results = model.fit()
                order_aic_bic.append((p, q, results.aic, results.bic))

        order_df = pd.DataFrame(order_aic_bic, columns=["p", "q", "AIC", "BIC"])
        order_rate = order_df.sort_values("AIC").iloc[0]
        p = order_rate[0]
        q = order_rate[1]

        arima = ARIMA(df_close, order=(p,0,q))
        results = arima.fit()

        steps_2 = int(df_close.shape[0]*0.2)
        steps_9 = int(df_close.shape[0]*0.9)

        date = df_close.index[-1] + timedelta(days=1)
        date = date.strftime("%Y-%m-%d")

        prediction = results.get_prediction(start=-steps_2)
        prediction_mean = prediction.predicted_mean
        actual_mean = df_close.tail(steps_2)
        actual_mean = pd.DataFrame(actual_mean)
        actual_list = actual_mean[f"{ticker_name}"].tolist()

        confidence_interval = prediction.conf_int()
        conf_lower = confidence_interval.iloc[:,0]
        conf_upper = confidence_interval.iloc[:,1]

        conf_lower_list = conf_lower.tolist()
        conf_upper_list = conf_upper.tolist()
        prediction_list = prediction_mean.tolist()

        ###########################PERFORMANCE METRICS###########################
        mae = mean_absolute_error(actual_mean, prediction_mean)
        rmse = mean_squared_error(actual_mean, prediction_mean, squared=False)
        mape = mean_absolute_percentage_error(actual_mean, prediction_mean)
        mape = mape * 100

        table_metric = {
            "Metric": ["MAE", "RMSE", "MAPE (%)"],
            "Value": [f"{mae:.2f}", f"{rmse:.2f}", f"{mape:.2f}"],
        }

        df_metric = pd.DataFrame(table_metric, index=["MAE", "RMSE", "MAPE (%)"], columns=["Value"])

        forecast = results.get_forecast(steps=interval)
        forecast_mean = forecast.predicted_mean
        forecast_date = pd.date_range(start=date, periods=len(forecast_mean), freq="D")
        forecast_series = pd.Series(forecast_mean.values, index=forecast_date, name="Forecasted Value")

        fig, ax = plt.subplots()
        ax.plot(df_close.index[steps_9:], df_close[steps_9:], label="Closed Price")
        ax.plot(forecast_series.index, forecast_series, label="Forecasted Price")
        ax.set_title(f"{selected_ticker} Forecasted Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.plotly_chart(fig)

        st.table(pd.Series(forecast_mean.values, index=forecast_date.strftime("%Y-%m-%d"), name="Forecasted Value"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prediction_mean.index, y=conf_lower_list, mode='lines', line_color="pink", name="Lower Bound"))
        fig.add_trace(go.Scatter(x=prediction_mean.index, y=conf_upper_list, fill="tonexty", fillcolor="pink", mode='lines', line_color="pink", name="Upper Bound"))
        fig.add_trace(go.Scatter(x=prediction_mean.index, y=prediction_list, mode="lines", line_color="red", name="Prediction Price"))
        fig.add_trace(go.Scatter(x=prediction_mean.index, y=actual_list, mode="lines", line_color="blue", name="Test Price"))
        fig.update_layout(title='Predicted Price and Confidence Interval',xaxis_title='Date',yaxis_title='Price',template='plotly_white')
        st.plotly_chart(fig)

        st.table(df_metric)




    with st.spinner("Model is running..."):
        stock_info(popular_stocks_json[selected_ticker], interval[selected_time])







