#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 14:35:16 2021

@author: macbookpro
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from PIL import Image
import datetime
from datetime import datetime, date, time, timedelta
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import base64
#

st.title('Financial Analysis')
st.write('Financial Data Analysis and Visualiton')
logo = Image.open('/Users/macbookpro/Documents/Noya Sunumlar/3P_logoSREZ.png')
st.sidebar.image(logo, use_column_width=True)
st.sidebar.title('Filter')

transactiontype = st.sidebar.radio('Transaction Type', ['Crypto', 'BIST(Borsa Istanbul)'])


if transactiontype == 'Crypto':
    kriptosec = st.sidebar.selectbox('Crypto Currency', ['BTC', 'ETH', 'XRP', 'DOT1', 'DOGE', 'AVAX', 'BNB'])
    kriptosec = kriptosec + '-USD'
    sembol = kriptosec
else:
    borsasec = st.sidebar.selectbox('BIST', ['ASELSAN', 'THY', 'GARANTI', 'AKBANK', 'GALATASARAY'])
    senetler = {'ASELSAN':'ASELS.IS',
                'THY':'THYAO.IS',
                'GARANTI':'GARAN.IS',
                'AKBANK':'AKBNK.IS',
                'GALATASARAY':'GSRAY.IS'
                }
    hissesec = senetler[borsasec]
    sembol = hissesec

zaralik = range(1,366)
slider = st.sidebar.select_slider('Time Interval', options=zaralik, value=30)

st.sidebar.write('### Date Range')

bugun = datetime.today()
aralik = timedelta(days=slider)

start_d = st.sidebar.date_input('Start Date', value=bugun-aralik)
end_d = st.sidebar.date_input('End Date', value=bugun)

st.sidebar.write('### Machine Learning Forecast')

prophet = st.sidebar.checkbox('FB Prohephet')


if prophet:
    fbaralik = range(1,721)
    fbperiyot = st.sidebar.select_slider('Forecast Period', options=fbaralik, value=30)
    components = st.sidebar.checkbox('Components')
    
if prophet:
    cvsec = st.sidebar.checkbox('XValidation')
    if cvsec:
        st.sidebar.write('### Choose Metric')
        metric = st.sidebar.radio('Metrics', ['rmse', 'mse', 'mape', 'mdape'])
        st.sidebar.write('### Choose Parameter')
        inaralik = range(1,1441)
        cvin = st.sidebar.select_slider('Initial', options=inaralik, value=120)
        peraralik = range(1,1441)
        cvper = st.sidebar.select_slider('XVal Period', options=peraralik, value=30)
        horaralik = range(1,1441)
        cvhor = st.sidebar.select_slider('Horizon', options=horaralik, value=60)
        

def grafikgetir(sembol,start_d, end_d):
    data = yf.Ticker(sembol)
    global df
    df = data.history(period='1d', start=start_d, end=end_d)
    st.line_chart(df['Close'])
    if prophet:
        fb = df.reset_index()
        fb = fb[['Date', 'Close']]
        fb.columns = ['ds', 'y']
        global model
        model = Prophet()
        model.fit(fb)
        future = model.make_future_dataframe(periods=fbperiyot)
        predict = model.predict(future)
        graph = model.plot(predict)
        # predict = predict[['ds', 'trend']]
        # predict = predict.set_index('ds')
        # st.line_chart(predict['trend'])
        st.write(graph)
        
        if components:
            graph2 = model.plot_components(predict)
            st.write(graph2)
    else:
        pass

def cvgrafik(model, initial, period, horizon, metric):
    initial = str(initial) + ' days'
    period = str(period) + ' days'
    horizon = str(horizon) + ' days'
    cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    graph3 = plot_cross_validation_metric(cv, metric=metric)
    st.write(graph3)

grafikgetir(sembol,start_d, end_d)


if prophet:
    if cvsec:
        cvgrafik(model, cvin, cvper, cvhor, metric)


def indir(df):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">CSV download</a>'
    return href


st.markdown(indir(df), unsafe_allow_html=True)


def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()


def EMA(data, period=21, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    ShortEMA = EMA(data, period_short, column=column)
    LongEMA = EMA(data, period_long, column=column)
    data['MACD'] = ShortEMA - LongEMA
    data['Signal Line'] = EMA(data, period_signal, column='MACD')
    return data


def RSI(data, period=14, column='Close'):
    delta = data[column].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up<0] = 0
    down[down>0] = 0
    data['up'] = up
    data['down'] = down
    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))
    RS = AVG_Gain/AVG_Loss
    RSI = 100.0 - (100.0/(1 + RS))
    data['RSI'] = RSI
    return data

st.sidebar.write('### Financial Indicators')
fi = st.sidebar.checkbox('Financial Indicators')

def filer():
    if fi:
        fimacd = st.sidebar.checkbox('MACD')
        firsi = st.sidebar.checkbox('RSI')
        fisl = st.sidebar.checkbox('Signal Line')
        if fimacd:
            macd = MACD(df)
            st.line_chart(macd['MACD'])
        if firsi:
            rsi = RSI(df)
            st.line_chart(rsi['RSI'])
        if fisl:
            macd = MACD(df)
            st.line_chart(macd['Signal Line'])

filer()







