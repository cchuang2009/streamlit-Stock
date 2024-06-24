import yfinance as yf
import talib
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Kline, Grid
from pyecharts.commons.utils import JsCode
from prophet import Prophet

from streamlit_echarts import st_pyecharts

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
    'MSFT': '微軟',
    'NVDA': '輝達',
    'ARM': '安謀',
    'AAPL':'蘋果',
    # Add more stocks and their corresponding symbols here
}

#
#  Pyechart's Kline supports "list" format but not native pandas DataFrame
#

def data_(df_):
    ydata=df_[['Open', 'Close', 'Low', 'High']].values.tolist()
    xdata=(df_.index.strftime('%Y-%m-%d')).tolist()
    return xdata,ydata
 
# Streamlit configuration
st.title("Stock Analysis")

stock_list_text = "Stocks:\n\n"
#for key, value in stocks.items():
#    stock_list_text += f"{key}: {value}, \n"

st.markdown(stock_list_text)

#selected_stock = st.selectbox("Select a stock", list(stocks.keys()))
selected_stock = st.selectbox("Select a stock", list(stocks.keys()), format_func=lambda x: f"{x} - {stocks[x]}")

time_period = st.selectbox("Select a time period", ['3 months', '6 months', '1 year', 'All'])

# Calculate the start date based on the selected time period
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
    
start_date_all= "2020-01-01"   
# Input column for company
symbol = st.text_input("Enter a company", "")
update_button = st.button("Update Chart")

# Download historical data for the selected stock or entered company
if update_button and symbol:
    #st.write(rf"{company_input}")
    #stock_data_ = yf.download(f"{company_input}", start=start_date_all, end=end_date)
    stock_data_ = yf.download(symbol, start=start_date_all, end=end_date)
else:
    stock_data_ = yf.download(selected_stock, start=start_date_all, end=end_date)
    

# Download historical data for the selected stock
#stock_data_ = yf.download(selected_stock, start=start_date_all, end=end_date)



# Calculate the technical indicators
sma5_ = talib.SMA(stock_data_['Close'], timeperiod=5)  # Simple Moving Average
sma10_ = talib.SMA(stock_data_['Close'], timeperiod=10)  # Simple Moving Average
sma20_ = talib.SMA(stock_data_['Close'], timeperiod=20)  # Simple Moving Average

rsi = talib.RSI(stock_data_['Close'], timeperiod=14)  # Relative Strength Index
upper, middle, lower = talib.BBANDS(stock_data_['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)  # Bollinger Bands

difs_, deas_,macd_ = talib.MACD(stock_data_['Close'])

# Determine the buy and sell signals based on the indicators
#buy_signals = (stock_data['Close'] > sma) & (sma > stock_data['Close'].shift()) & (rsi < 30)
#sell_signals = (stock_data['Close'] < sma) & (rsi > 70)
buy_signals = (sma10_ > stock_data_['Close']) & (rsi < 30)
sell_signals = (sma10_ < stock_data_['Close']) & (rsi > 70)
# Implement a trailing stop loss
stop_loss_percent = 5  # Adjust the stop loss percentage as desired
highest_price = stock_data_['Close'].rolling(window=20, min_periods=1).max()
stop_loss = highest_price * (1 - stop_loss_percent / 100)

# Calculate the target price
target_percent = 10  # Default target percentage
target_price = stock_data_['Close'] * (1 + target_percent / 100)

#data_kd=stock_data[['Open', 'Close', 'Low', 'High']]
#data_kd.columns=['open', 'close', 'low', 'high']

stock_data=stock_data_.tail(ndays)
sma5=sma5_[-ndays:]
sma10=sma10_[-ndays:]
sma20=sma20_[-ndays:]
difs,deas,macd = difs_[-ndays:], deas_[-ndays:],macd_[-ndays:]

data_kd=stock_data[['Open', 'Close', 'Low', 'High']]
#data_kd.columns=['open', 'close', 'low', 'high']


# Create a Line chart using Pyecharts
line_chart = (
    Line()
    .add_xaxis(stock_data.index.strftime('%Y-%m-%d'))
    .add_yaxis('Close', stock_data['Close'], yaxis_index=0)
    .add_yaxis('RSI', rsi, yaxis_index=1)
    .add_yaxis('SMA5', sma5)
    .add_yaxis('SMA10', sma10)
    .add_yaxis('SMA20', sma20)   
    .add_yaxis('Buy', stock_data['Close'][buy_signals], symbol='triangle', symbol_size=10)
    .add_yaxis('Sell', stock_data['Close'][sell_signals], symbol='triangle-down', symbol_size=10)
    .add_yaxis('Stop Loss', stop_loss)
    .add_yaxis('Target Price', target_price, linestyle_opts=opts.LineStyleOpts(type_='dashed'))
    .add_yaxis('Upper Band', upper)
    .add_yaxis('Middle Band', middle)
    .add_yaxis('Lower Band', lower)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(),#title=f"{stocks[selected_stock]}"),#,pos_top='top', pos_left='center'),
                     xaxis_opts=opts.AxisOpts(type_="category"))
    .extend_axis(yaxis=opts.AxisOpts(name="RSI", position="right",min_=0,max_=100),)        
)

xdata,ydata=data_(stock_data)

kline= (
    Kline()
        .add_xaxis(xdata)
        .add_yaxis(
            "KD",
            ydata,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",
                color0="#14b143",
                border_color="#ef232a",
                border_color0="#14b143",
            ),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值"),
                    opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            markline_opts=opts.MarkLineOpts(
                label_opts=opts.LabelOpts(
                    position="middle", color="blue", font_size=15
                ),
                #data=split_data_part(),
                data=data_kd,
                symbol=["circle", "none"],
            ),
        )
        .set_series_opts(
            #markarea_opts=opts.MarkAreaOpts(is_silent=True, data=split_data_part())
            #markarea_opts=opts.MarkAreaOpts(is_silent=True, data=ydata)
            label_opts=opts.LabelOpts(is_show=False)
        )   
        .set_global_opts(
            title_opts=opts.TitleOpts(title="K-D", pos_left="0"),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False, type_="inside", xaxis_index=[0, 0], range_end=100
                ),
                opts.DataZoomOpts(
                    is_show=True, xaxis_index=[0, 1], pos_top="97%", range_end=100
                ),
                opts.DataZoomOpts(is_show=False, xaxis_index=[0, 2], range_end=100),
            ],
        )
)
line_ma = (
    Line()
    .add_xaxis(stock_data.index.strftime('%Y-%m-%d'))
    #.add_yaxis('Close', stock_data['Close'], yaxis_index=0)
    #.add_yaxis('RSI', rsi, yaxis_index=1)
    .add_yaxis('SMA5', sma5,is_smooth=True,linestyle_opts=opts.LineStyleOpts(opacity=0.5),)
    .add_yaxis('SMA10', sma10)
    .add_yaxis('SMA20', sma20)   
    .add_yaxis('Buy', stock_data['Close'][buy_signals], symbol='triangle', symbol_size=10)
    .add_yaxis('Sell', stock_data['Close'][sell_signals], symbol='triangle-down', symbol_size=10)
    #.add_yaxis('Stop Loss', stop_loss)
    #.add_yaxis('Target Price', target_price, linestyle_opts=opts.LineStyleOpts(type_='dashed'))
    #.add_yaxis('Upper Band', upper)
    #.add_yaxis('Middle Band', middle)
    #.add_yaxis('Lower Band', lower)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(),#title=f"{stocks[selected_stock]}"),#,pos_top='top', pos_left='center'),
                     xaxis_opts=opts.AxisOpts(type_="category"),
                     yaxis_opts=opts.AxisOpts(
                     grid_index=1,split_number=3,
                     axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                     axistick_opts=opts.AxisTickOpts(is_show=False),
                     splitline_opts=opts.SplitLineOpts(is_show=False),
                     axislabel_opts=opts.LabelOpts(is_show=True),
            ),
                    )
    #.extend_axis(yaxis=opts.AxisOpts(name="RSI", position="right",min_=0,max_=100),)        
)
# Overlap Kline + Line
overlap_kline = kline.overlap(line_ma)

vols= stock_data["Volume"].values.tolist()
bar_1 = (
        Bar()
        .add_xaxis(xdata)
        .add_yaxis(
            "Volumn",vols,
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                        function(params) {
                            var colorList;
                            if (params.data >= 0) {
                              colorList = '#ef232a';
                            } else {
                              colorList = '#14b143';
                            }
                            return colorList;
                        }
                        """
                )
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts("Volumns"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    ) 

#difs, deas,macd = talib.MACD(stock_data['Close'])
macd_=macd.values.tolist()

# Bar-2 (Overlap Bar + Line)
bar_2 = (
        Bar()
        .add_xaxis(xaxis_data=xdata)
        .add_yaxis(
            series_name="MACD",
            y_axis=macd_,
            xaxis_index=2,
            yaxis_index=2,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                        function(params) {
                            var colorList;
                            if (params.data >= 0) {
                              colorList = '#ef232a';
                            } else {
                              colorList = '#14b143';
                            }
                            return colorList;
                        }
                        """
                )
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=2,
                split_number=4,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

line_2 = (
        Line()
        .add_xaxis(xaxis_data=xdata)
        .add_yaxis(
            series_name="DIF",
            y_axis=difs,
            xaxis_index=2,
            yaxis_index=2,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add_yaxis(
            series_name="DEA",
            y_axis=deas,
            xaxis_index=2,
            yaxis_index=2,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
    )
# 最下面的柱状图和折线图
overlap_bar_line = bar_2.overlap(line_2)


#st_pyecharts(line_chart,height="500px")
#st_pyecharts(overlap_kline,height="500px")
#st_pyecharts(bar_1,height="200px")
#st_pyecharts(overlap_bar_line,height="200px")


grid_chart = Grid()
#grid_chart.add_js_funcs("var barData = {}".format(vols))
grid_chart.add(
        overlap_kline,
        grid_opts=opts.GridOpts(pos_left="3%", pos_right="1%", height="60%"),
)
grid_chart.add(
        bar_1,
        grid_opts=opts.GridOpts(
            pos_left="3%", pos_right="1%", pos_top="71%", height="10%"
        ),
)
grid_chart.add(
        overlap_bar_line,
        grid_opts=opts.GridOpts(
            pos_left="3%", pos_right="1%", pos_top="82%", height="14%"
        ),
)

st_pyecharts(grid_chart,height="800px")

#st.write(macd)


if symbol:
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if stock_data.empty:
        st.write("No data available for the target stock symbol")
    else:
        #st.write("Historical Data for", symbol)
        #st.write(stock_data)

        # Prepare the data for Prophet forecasting
        df = stock_data.reset_index()
        df = df[['Date', 'Close']]
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

        # Train the Prophet model
        model = Prophet()
        model.fit(df)

        # Generate future dates for forecasting
        future_dates = model.make_future_dataframe(periods=7)

        # Perform the forecasting
        forecast = model.predict(future_dates)
        
        # Display the forecasted data
        st.write("Forecasted Data for the Next Week")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))
        f_=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
        min_y = min(f_['yhat'])*0.9
        max_y = max(f_['yhat'])*1.1
        line_prediction = (
    Line()
    .add_xaxis(f_.ds)
    #.add_yaxis('Close', stock_data['Close'], yaxis_index=0)
    #.add_yaxis('RSI', rsi, yaxis_index=1)
    .add_yaxis(r'ŷ', f_['yhat'])
    .add_yaxis('y-2σ', f_['yhat_lower'])
    .add_yaxis('y+2σ', f_['yhat_upper'])        
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(),#title=f"{stocks[selected_stock]}"),#,pos_top='top', pos_left='center'),
                     xaxis_opts=opts.AxisOpts(type_="category"),
                     yaxis_opts=opts.AxisOpts(min_=min_y, max_=max_y,                         
                     axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                     axistick_opts=opts.AxisTickOpts(is_show=False),
                     splitline_opts=opts.SplitLineOpts(is_show=False),
                     axislabel_opts=opts.LabelOpts(is_show=True),
            ),
                    )    
)

        
#st_pyecharts(line_prediction,height="400px")
        
