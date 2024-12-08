import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

# Define the time period
start_date = "2010-01-01"
end_date = "2023-12-31"

# Fetch stock data using yfinance
st.title(':red[Stock] :green[Trend] :blue[Prediction]')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
stock_data = yf.download(user_input, start=start_date, end=end_date)


# Describing Data
st.subheader('Data from 2010 to 2023')
st.write(stock_data.describe())

# Data Visualization
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (15, 6))
plt.plot(stock_data.Close)
st.pyplot(fig)


st.subheader('In 100 Moving Average')
ma100 = stock_data.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'orange')
plt.plot(stock_data.Close)
st.pyplot(fig)

st.subheader('Combination of 100 and 200 Moving Average')
ma100 = stock_data.Close.rolling(100).mean()
ma200 = stock_data.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'orange')
plt.plot(ma200, 'red')
plt.plot(stock_data.Close)
st.pyplot(fig)

# Training and Testing  Data
data_training = pd.DataFrame(stock_data['Close'][0 : int(len(stock_data) * 0.70)])
data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data) * 0.70): int(len(stock_data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))




# Load the models
model = load_model('keras_model.h5')


# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predict = model.predict(x_test)
scaler.scale_

scale_factor = 1 / 0.00646057
y_predict = y_predict * scale_factor
y_test = y_test * scale_factor


# Final Graph
st.subheader('Predictive Stock Vs Original Stock')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y_test, 'b', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

