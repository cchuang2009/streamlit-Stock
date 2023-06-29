import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from math import floor
from helper import *

#from streamlit_echarts import st_pyecharts

from datetime import datetime, timedelta
# Define the start date
start_date = datetime(2020, 1, 1)

# Get the current date
current_date = datetime.now()

# Calculate the difference in days
days_diff = (current_date - start_date).days


# Define the stock symbols and target prices
stocks = {
    '2308.TW': '台達電',
    '2330.TW': '台積電',
    '2382.TW': '廣達',
    '2498.TW': '宏達電',
    '2603.TW': '長榮',
    '2618.TW': '長榮航空',
    '2634.TW': '漢翔',
    '3583.TW': '辛耘',
    '5222.TW': '全訊',
    '5347.TWO': '世界',
    '6415.TW': '矽力',
    '6510.TWO': '精測',
    '6533.TW': '晶心科',
    '6515.TW': '穎崴',
    '6691.TW': '洋基工程',
    '3008.TW': '大立光',
    # Add more stocks and their corresponding symbols here
}

 
# Streamlit configuration
st.title("Stock MACD Analysis")

#symbol = st.text_input("Enter a company", "")
selected_stock = st.selectbox("Select a stock", list(stocks.keys()), format_func=lambda x: f"{x} - {stocks[x]}")
time_period = st.selectbox("Select a time period", ['3 months', '6 months', '1 year', 'All'])

update_button = st.button("Submit")



end_date = datetime.now().strftime('%Y-%m-%d')
if time_period == '1 month':
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    ndays=30
elif time_period == '3 months':
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    ndays=90
elif time_period == '6 months':
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    ndays=180
elif time_period == '1 year':
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    ndays=365
else:
    start_date = "2020-01-01"
    ndays=days_diff
    
    
    
if update_button:
   tick = yf.download(selected_stock, start=start_date, end=current_date)

   # Calculate MACD
   tick_macd = get_macd(tick['Close'], slow=26, fast=12, smooth=9)

   # Implement MACD strategy
   buy_price, sell_price, macd_signal = implement_macd_strategy(tick['Close'], tick_macd)

   # Visualize MACD signals
   trading_vis_matplotlib(tick, tick_macd, buy_price, sell_price)

   # Calculate position and perform backtest
   tick_stradegy=calculate_strategy(macd_signal, tick_macd, tick)

   investment_value = 100000
   result = backtest_macd_strategy(investment_value, tick, tick_stradegy)

   # Print the backtest result
   st.write("Total investment return:", result['total_investment_return'])
   st.write("Profit percentage:", result['profit_percentage'])

