from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array

st.write("""
# Stock Market Web Application
**Stock Market** data from June 6, 2016 - June 01, 2021
""")
img = Image.open("C:/DATA/Programming/Python3.7/FirstProgram/HelloWorld/imag1.png")
st.image(img, use_column_width=True)
st.sidebar.header('User Input')


def get_input():
    start_date = st.sidebar.text_input("Start Date", "06-02-2016")
    end_date = st.sidebar.text_input("End Date", "06-01-2021")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "APPL")
    return start_date, end_date, stock_symbol


def get_company_name(symbol):
    if symbol == 'AMZN':
        return 'Amazon'
    elif symbol == 'APPL':
        return 'Apple'
    elif symbol == 'GOOG':
        return 'Google'
    else:
        'None'


def get_data(symbol, start, end):
    if symbol.upper() == 'AMZN':
        df = pd.read_csv("C:/DATA/Programming/Python3.7/FirstProgram/HelloWorld/AMZN.csv")
    elif symbol.upper() == 'APPL':
        df = pd.read_csv("C:/DATA\Programming/Python3.7/FirstProgram/HelloWorld/AAPL.csv")
    elif symbol.upper() == 'GOOG':
        df = pd.read_csv("C:/DATA/Programming/Python3.7/FirstProgram/HelloWorld/GOOG.csv")
    else:
        df = pd.read_csv(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj', 'Close', 'Volume'])
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    start_row = 0
    end_row = 0
    for i in range(0, len(df)):
        if start <= pd.to_datetime(df['Date'][i]):
            start_row = i
            break
    for j in range(0, len(df)):
        if end >= pd.to_datetime(df['Date'][len(df) - 1 - j]):
            end_row = len(df) - 1 - j
            break
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    return df.iloc[start_row:end_row + 1, :]


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


start, end, symbol = get_input()
df = get_data(symbol, start, end)
company_name = get_company_name(symbol.upper())
df1 = df.reset_index()['Close']
st.header(company_name + " Close Price\n")
st.line_chart(df1)
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
training_size = int(len(df1) * 0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
### Calculate RMSE performance metrics
srm=math.sqrt(mean_squared_error(y_train,train_predict))
### Test Data RMSE
srm1=math.sqrt(mean_squared_error(ytest,test_predict))
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
x_input=test_data[340:].reshape(1,-1)
print(x_input.shape)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
lst_output = []
n_steps = 100
i = 0
while (i < 30):

    if (len(temp_input) > 100):
        print(len(temp_input))
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1
df3=df1.tolist()
df3.extend(lst_output)
st.header(company_name + " 08-03-2021 to 01-07-2021 previous data with 30 days prediction \n")
st.line_chart(df3[1200:])
st.header(company_name + " 02-06-2016 to 01-07-2021 \n")
df3=scaler.inverse_transform(df3).tolist()
st.line_chart(df3)
st.header('Data Statistics')
st.write(df.describe())
