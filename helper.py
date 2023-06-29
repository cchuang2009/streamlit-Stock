# tools

import pandas as pd
import streamlit as st

def get_macd(price, slow, fast, smooth):
    """
    macd_df=get_macd(df['Close'], 26, 12, 9)
    """
    exp1 = price.ewm(span=fast, adjust=False).mean()
    exp2 = price.ewm(span=slow, adjust=False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns={'Close': 'macd'})
    signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={'macd': 'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns={0: 'hist'})
    frames = [macd, signal, hist]
    df = pd.concat(frames, join='inner', axis=1)
    return df
    
import numpy as np

def implement_macd_strategy(prices, data):
    """
    buy_price, sell_price, macd_signal = implement_macd_strategy(googl['Close'], googl_macd)
    
    """
    buy_price = np.empty(len(data))
    sell_price = np.empty(len(data))
    macd_signal = np.zeros(len(data))
    signal = 0

    buy_price[:] = np.nan
    sell_price[:] = np.nan

    buy_condition = (data['macd'] > data['signal'])
    sell_condition = (data['macd'] < data['signal'])

    buy_signal = np.where(np.logical_and(buy_condition, signal != 1))
    sell_signal = np.where(np.logical_and(sell_condition, signal != -1))

    buy_price[buy_signal] = prices[buy_signal]
    sell_price[sell_signal] = prices[sell_signal]

    macd_signal[buy_signal] = 1
    macd_signal[sell_signal] = -1

    return buy_price, sell_price, macd_signal    
    
    
def implement_macd_strategy(prices, data):    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(data)):
        if data['macd'][i] > data['signal'][i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['macd'][i] < data['signal'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal
    
import matplotlib.pyplot as plt

def trading_vis_matplotlib(googl, googl_macd, buy_price, sell_price):
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

    ax1.plot(googl['Close'], color='skyblue', linewidth=2, label='GOOGL')
    ax1.plot(googl.index, buy_price, marker='^', color='green', markersize=10, label='BUY SIGNAL', linewidth=0)
    ax1.plot(googl.index, sell_price, marker='v', color='r', markersize=10, label='SELL SIGNAL', linewidth=0)
    ax1.legend()
    ax1.set_title('MACD SIGNALS')
    ax2.plot(googl_macd.index, googl_macd['macd'], color='grey', linewidth=1.5, label='MACD')
    ax2.plot(googl_macd.index, googl_macd['signal'], color='skyblue', linewidth=1.5, label='SIGNAL')

    for i in range(len(googl_macd)):
        if str(googl_macd['hist'][i])[0] == '-':
            ax2.bar(googl_macd.index[i], googl_macd['hist'][i], color='#ef5350')
        else:
            ax2.bar(googl_macd.index[i], googl_macd['hist'][i], color='#26a69a')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    plt.legend(loc='lower right')
    #plt.show()
    st.pyplot(fig)
    #return fig



from pyecharts import options as opts
from pyecharts.charts import Bar, Line
from pyecharts.commons.utils import JsCode
from pyecharts.faker import Faker

def trading_vis_pyecharts(googl, buy_price, sell_price, googl_macd):
    # Create the Line chart for GOOGL price
    line = (
        Line()
        .add_xaxis(xaxis_data=googl.index.tolist())
        .add_yaxis(
            series_name="Tick",
            y_axis=googl["Close"].tolist(),
            color="skyblue",
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="MACD SIGNALS"),
            legend_opts=opts.LegendOpts(),
        )
    )

    # Create the Line chart for MACD and SIGNAL lines
    line_macd = (
        Line()
        .add_xaxis(xaxis_data=googl_macd.index.tolist())
        .add_yaxis(
            series_name="MACD",
            y_axis=googl_macd["macd"].tolist(),
            color="grey",
            linestyle_opts=opts.LineStyleOpts(width=1.5),
        )
        .add_yaxis(
            series_name="SIGNAL",
            y_axis=googl_macd["signal"].tolist(),
            color="skyblue",
            linestyle_opts=opts.LineStyleOpts(width=1.5),
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_right="5%", pos_top="10%"),
        )
    )

    # Create the Bar chart for MACD Histogram
    bar = (
        Bar()
        .add_xaxis(xaxis_data=googl_macd.index.tolist())
        .add_yaxis(
            "MACD Histogram",googl_macd["hist"].tolist(),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                    function(params) {
                        if (params.data >= 0) {
                            return '#26a69a';
                        } else {
                            return '#ef5350';
                        }
                    }
                    """
                )
            ),
        )
    )

    # Combine the charts
    grid_chart = (
        line.overlap(bar)
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        )
        .extend_axis(yaxis=opts.AxisOpts(type_="value", name="MACD Histogram", position="right"))
    )

    # Render the charts
    grid_chart.render()



import pandas as pd

def calculate_macd_position(macd_signal):
    position = []
    for i in range(len(macd_signal)):
        if macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
    return position

def calculate_strategy(macd_signal, googl_macd, googl):
    position = calculate_macd_position(macd_signal)

    for i in range(len(googl['Close'])):
        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]

    macd = googl_macd['macd']
    signal = googl_macd['signal']
    close_price = googl['Close']

    macd_signal_df = pd.DataFrame(macd_signal).rename(columns={0: 'macd_signal'}).set_index(googl.index)
    position_df = pd.DataFrame(position).rename(columns={0: 'macd_position'}).set_index(googl.index)

    strategy = pd.concat([close_price, macd, signal, macd_signal_df, position_df], join='inner', axis=1)
    return strategy


from math import floor
from termcolor import colored as cl

def backtest_macd_strategy(investment_value, googl, strategy):
    googl_ret = pd.DataFrame(np.diff(googl['Close'])).rename(columns={0: 'returns'})
    macd_strategy_ret = []

    for i in range(len(googl_ret)):
        try:
            returns = googl_ret['returns'][i] * strategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass

    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns={0: 'macd_returns'})

    number_of_stocks = floor(investment_value / googl['Close'][0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks * macd_strategy_ret_df['macd_returns'][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns={0: 'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret / investment_value) * 100)
    
    result = {
        'total_investment_return': total_investment_ret,
        'profit_percentage': profit_percentage
    }
    
    return result
