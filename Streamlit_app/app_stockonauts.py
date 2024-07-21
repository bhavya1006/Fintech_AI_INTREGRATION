from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
#from yahoo_fin import stock_info

import streamlit as st  
from streamlit_option_menu import option_menu 

import pandas as pd 
import numpy as np

from prophet import Prophet 
import yfinance as yf  
import datetime  
import time
from datetime import date,timedelta

import plotly.express as px
from plotly import graph_objs as go  
from plotly.subplots import make_subplots
from prophet.plot import plot_plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot

import requests
import os
import replicate  
import random
import cufflinks

import transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Stockonauts",page_icon="chart_with_upwards_trend",layout="wide", initial_sidebar_state="auto")



def add_meta_tag():
    meta_tag = """
        <head>
            <meta name="google-site-verification" content="QBiAoAo1GAkCBe1QoWq-dQ1RjtPHeFPyzkqJqsrqW-s" />
            <meta name="Real time stock analysis" content=" Algorthimic trading strategies" />
            <meta name="author" content="Your name" />
        </head>
    """
    st.markdown(meta_tag, unsafe_allow_html=True)
    
add_meta_tag()




today = date.today() 
st.write('''# Real Time Market Insights ''')
# Sidebar
st.sidebar.image("./static/Stockonauts.png", width=400,
                 use_column_width="auto")



with st.sidebar: 
     selected = option_menu("Dashboard", ["Home","Live Stock Prices", "Comparative Analysis of Stock Performances", "Stock Forecast","Personalized Investment Insights", "Algorithmic Trading","Multifactor Market Dynamics",'Strategy Overview'])
     st.write("Pick Date Interval")

start = st.sidebar.date_input('Start', datetime.date(2019, 1, 1))  # start date input
end = st.sidebar.date_input('End', datetime.date.today())


stock_df = pd.read_csv("./datasets/TickersData.csv")
#print(stock_df)
if(selected == 'Comparative Analysis of Stock Performances'):
    st.subheader("Comparative Analysis of Stock Performances")

    tickers = stock_df["Company Name"]
    # print(tickers)
    dropdown = st.multiselect('Pick your assets', tickers)
    # print(','.join([str(i) for i in dropdown]))
    with st.spinner('Loading...'):
        time.sleep(2)
       # st.success('Loaded')
    dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
    # print(dict_csv)
    symb_list = [] 
    for i in dropdown:  
        val = dict_csv.get(i)  
        symb_list.append(val)
        # Daily return is calculated by subtracting the opening price from the closing price.
    def relativeret(df):  
        rel = df.pct_change()  
        cumret = (1+rel).cumprod() - 1  
        cumret = cumret.fillna(0) 
        return cumret  
    # The cumprod() method goes through the values in the DataFrame, from the top, row by row, multiplying the values with the value from the previous row, ending up with a DataFrame where the last row contains the product of all values for each column
    if len(dropdown) > 0:
        #df2=yf.download(symb_list,start,end)
        df = relativeret(yf.download(symb_list, start, end))[
            'Adj Close']
        #print(df)
        #print(df2)
        raw_df = relativeret(yf.download(symb_list, start, end))
        raw_df.reset_index(inplace=True) 

        closingPrice = yf.download(symb_list, start, end)[
            'Adj Close'] 
        volume = yf.download(symb_list, start, end)['Volume']

        
        st.subheader('Raw Data :  {}'.format(' - '.join([str(i)for i in dropdown])))

        st.write(raw_df)
   

        chart = ('Line Chart', 'Area Chart', 'Bar Chart')
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)
        st.subheader('Relative Returns {}'.format(' - '.join([str(i)for i in dropdown])))
        if (dropdown1) == 'Line Chart':  
            st.line_chart(df)  
            st.write("### Closing Price of {}".format(' - '.join([str(i)for i in dropdown])))
            st.line_chart(closingPrice) 
            st.write("### Volume of {}".format(' - '.join([str(i)for i in dropdown])))
            st.line_chart(volume)  
        elif (dropdown1) == 'Area Chart':  
            st.area_chart(df)  
            st.write("### Closing Price of {}".format(dropdown))
            st.area_chart(closingPrice)  
            st.write("### Volume of {}".format(dropdown))
            st.area_chart(volume)
        elif (dropdown1) == 'Bar Chart': 
            st.bar_chart(df) 
            st.write("### Closing Price of {}".format(dropdown))
            st.bar_chart(closingPrice)  
            st.write("### Volume of {}".format(dropdown))
            st.bar_chart(volume)  

        else:
            st.line_chart(df, width=1000, height=800,
                          use_container_width=False) 
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume) 

    else: 
        st.write('Please select atleast one asset') 

    


elif(selected == 'Live Stock Prices'):
    st.subheader("Real-Time Stock Price")
    tickers = stock_df["Company Name"]  # get company names from csv file
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)

    with st.spinner('Loading...'):  # spinner while loading
        time.sleep(2)

        dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
        symb_list = []  # list for storing symbols

        val = dict_csv.get(a)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list

        if "button_clicked" not in st.session_state:  # if button is not clicked
            st.session_state.button_clicked = False  # set button clicked to false

        def callback():  # function for updating data
            # if button is clicked
            st.session_state.button_clicked = True  # set button clicked to true
        if (
            st.button("Search", on_click=callback)  # button for searching data
            or st.session_state.button_clicked  # if button is clicked
        ):
            if(a == ""):  # if user doesn't select any company
                st.write("Click Search to Search for a Company")
                with st.spinner('Loading...'):  # spinner while loading
                    time.sleep(2)
            else:  # if user selects a company
                # download data from yfinance
                data = yf.download(symb_list, start=start, end=end)
                data.reset_index(inplace=True)  # reset index
                st.subheader('Raw Data of {}'.format(a))  # display raw data
                st.write(data)  # display data

                def plot_raw_data():  # function for plotting raw data
                    fig = go.Figure()  # create figure
                    fig.add_trace(go.Scatter(  # add scatter plot
                        x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
                    fig.add_trace(go.Scatter(  # add scatter plot
                        x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
                    fig.layout.update(  # update layout
                        title_text='Line Chart of {}'.format(a) , xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
                    st.plotly_chart(fig)  # display plotly chart

                def plot_candle_data():  # function for plotting candle data
                    fig = go.Figure()  # create figure
                    fig.add_trace(go.Candlestick(x=data['Date'],  # add candlestick plot
                                                # x-axis: date, open
                                                open=data['Open'],
                                                high=data['High'],  # y-axis: high
                                                low=data['Low'],  # y-axis: low
                                                close=data['Close'], name='market data'))  # y-axis: close
                    fig.update_layout(  # update layout
                        title='Candlestick Chart of {}'.format(a),  # title
                        yaxis_title='Stock Price',  # y-axis: title
                        xaxis_title='Date')  # x-axis: title
                    st.plotly_chart(fig)  # display plotly chart

                chart = ('Candle Stick', 'Line Chart')  # chart types
                # dropdown for selecting chart type
                dropdown1 = st.selectbox('Pick your chart', chart)
                with st.spinner('Loading...'):  # spinner while loading
                    time.sleep(2)
                if (dropdown1) == 'Candle Stick':  # if user selects 'Candle Stick'
                    plot_candle_data()  # plot candle data
                elif (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
                    plot_raw_data()  # plot raw data
                else:  # if user doesn't select any chart
                    plot_candle_data()  # plot candle data
        


elif(selected == 'Stock Forecast'):  # if user selects 'Stock Prediction'
    st.subheader("Stock Prediction")
    tickers = stock_df["Company Name"]  # get company names from csv file
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)
    with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
    dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols
    val = dict_csv.get(a)  # get symbol from csv file
    symb_list.append(val)  # append symbol to list
    if(a == ""):  # if user doesn't select any company
        st.write("Enter a Stock Name")  # display message
    else:  # if user selects a company
        # download data from yfinance
        data = yf.download(symb_list, start=start, end=end)
        data.reset_index(inplace=True)  # reset index
        # st.subheader('Raw Data of {}'.format(a))  # display raw data
        # st.write(data)  # display data

        def plot_raw_data():  # function for plotting raw data
            fig = go.Figure()  # create figure
            fig.add_trace(go.Scatter(  # add scatter plot
                x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
            fig.add_trace(go.Scatter(  # add scatter plot
                x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
            fig.layout.update(  # update layout
                title_text='Time Series Data of {}'.format(a), xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
            st.plotly_chart(fig)  # display plotly chart

        plot_raw_data()  # plot raw data
        # slider for selecting number of years
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365  # calculate number of days

        # Predict forecast with Prophet
        # create dataframe for training data
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(
            columns={"Date": "ds", "Close": "y"})  # rename columns

        m = Prophet()  # create object for prophet
        m.fit(df_train)  # fit data to prophet
        future = m.make_future_dataframe(
            periods=period)  # create future dataframe
        forecast = m.predict(future)  # predict future dataframe

        # Show and plot forecast
        st.subheader('Forecast Data of {}'.format(a))  # display forecast data
        st.write(forecast)  # display forecast data

        st.subheader(f'Forecast plot for {n_years} years')  # display message
        fig1 = plot_plotly(m, forecast)  # plot forecast
        st.plotly_chart(fig1)  # display plotly chart

        st.subheader("Forecast components of {}".format(a))  # display message
        fig2 = m.plot_components(forecast)  # plot forecast components
        st.write(fig2)  # display plotly chart

elif(selected == 'Strategy Overview'):
    st.subheader("Strategy 1 - Algorithmic Trading")
    st.markdown("""
        <style>
    .big-font {
        font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    


    # Banking Sector Analysis
    
    st.title("Banking sector - Considerations for analysis")
    # Higher Net Interest Margin (NIM)
    st.header("Higher Net Interest Margin (NIM)")

    # Definition of NIM
    nim_definition = "NIM is the difference between the interest earned on loans and the interest paid on depositsi.e amount received from customers against interest received on the lending amount.Select banks with a higher NIM, indicating better profitability from lending activities.It indicates a bank is bringing in more money on the interest it earns on loans than it is paying out in interest on bank deposits."
  
    st.write(nim_definition)


    # Line break
    st.markdown("---")

    # Lower Non-Performing Assets (NPA)
    st.header("Lower Non-Performing Assets (NPA)")

    # Definition of NPA
    npa_definition = "NPAs, also known as bad loans or distressed assets, are financial assets held by a bank that do not generate any income for the bank These assets typically include loans that are not being repaid by borrowers due to various reasons such as financial distress, stagnation, unexpected events, or even bankruptcy.The primary objective when assessing NPAs is to gauge the asset quality of a bank. A lower level of NPAs is generally seen as a positive indicator, suggesting that the bank has a healthier loan portfolio with a lower likelihood of default."
    st.write(npa_definition)
    st.markdown("---")
    st.title("Pricing and Earnings Analysis")
    st.header("P/E Ratio (Price to Earnings Ratio) vs P/B Ratio (Price to Book Ratio)  : Which one to use?")
    pepb = "P/E ratio is a popular measure of how expensive a company’s stock is. It is simply the company’s market capitalization divided by its net income — in other words, how much does it cost us to buy $1 of a particular company’s earnings. The higher the P/E ratio, all other things equal, the more expensive a stock is perceived to be.the P/E ratio shows what the market is willing to pay today for a stock based on its past or future earnings. A high P/E could mean that a stock's price is high relative to earnings and possibly overvalued. Conversely, a low P/E might indicate that the current stock price is low relative to earnings.The P/B ratio on the other hand measures the market's valuation of a company relative to its book value.P/B ratio is used by value investors to identify potential investments and P/B ratios under 1 are typically considered solid investments."
    st.write(pepb)
    st.markdown("---")
    st.subheader("Strategy 2 - Multifactor Market Dynamics")
    st.title("Collaborative approach and Digitalization Analysis (all Sectors)")
    st.header("High-Performing Collaborations:")
    txt1 = "Evaluate collaborations in diverse sectors (e.g., tech-automotive, pharma-tech) similar to ICICI Bank's UPI-Zomato partnership.Collaborations in autonomous driving tech between tech firms and automakers influence stock values.Partnerships in healthcare tech and pharmaceuticals impact both sectors and stock performance Assess their impact on stock shares.Analyze market reactions to collaborations, focusing on increased market share or revenue growth.Evaluate investor sentiment regarding collaborations. Positive perceptions can contribute to stock appreciation, while negative sentiment may have adverse effects."
    st.write(txt1)
    st.markdown("---")
    st.header("Digitalization Impact")
    txt2="Investigate the impact of digitalization on the stock shares of banks.Explore collaborations and partnerships in the digital banking space.Fintech Collaborations- Banks collaborating with fintech companies for digital payment solutions, robo-advisors, or innovative banking apps can experience positive impacts on stock shares.Blockchain Integration-for secure and efficient transactions may see positive effects on their stock values. Evaluate how banks adopting digital technologies impact their operational efficiency and customer experience, affecting stock performance.Assess the impact of digitalization on customer engagement and satisfaction, as this can contribute to long-term stock value growth."
    st.write(txt2)
    st.markdown("---")
    st.header("Micro and Macro Economics")
    txt3="Explore microeconomics (individual markets) and macroeconomics (national-level markets). Evaluate the impact of supply and demand dynamics on both individual companies and entire sectors.Assess how microeconomic factors like pricing strategies and market competition influence individual companies. Analyze how specific products or services respond to changes in supply and demand. Examine broader economic indicators such as GDP and unemployment rates for macroeconomic insights.Understand how macroeconomic supply and demand dynamics influence entire sectors.Identify cyclical and structural trends in both micro and macroeconomic factors.Consider the impact of government policies and global economic interdependencies on micro and macroeconomic landscapes."
    st.write(txt3)
    st.markdown("---")
    st.header("NLP - Sentiment Analysis")
    txt4="The motivation behind this approch  stems from a personal interest in investing and understanding the dynamics of the Indian Stock Market. Recognizing the importance of both Fundamental and Technical Analysis in stock selection, the project aims to leverage data science techniques to enhance the decision-making process.Fundamental Analysis involves evaluating a company's performance and future prospects through factors like cash flow statements and annual reports."
    txt5="On the other hand, Technical Analysis focuses on stock market performance, utilizing metrics such as P/E Ratio, EPS, and various patterns like candlestick patterns.The project also emphasizes the significance of staying updated with overall market news. By scraping articles from the Economic Times, the goal is to gauge market sentiment through NLP techniques, classifying news articles as positive or negative. This sentiment analysis can serve as a decision factor for investing in the index."
    txt6="The collected data includes article date, headline, summary, and URL. To address potential scraping issues, the load more articles feature is automated through code.NLP preprocessing techniques, including lemmatization, are applied to transform articles into a format suitable for analysis. Techniques like chunking, Name Entity Recognition, POS tagging, and n-grams are explored but not incorporated into the final model.Handling missing data involves replacing incomplete articles with summaries, ensuring a comprehensive dataset for analysis.The ultimate goal is to provide insights into the intrinsic value of stocks, whether they are overvalued or undervalued, and to understand the overall bullish or bearish sentiment in the market based on news analysis. The approach combines financial analysis, web scraping, NLP, and machine learning techniques to facilitate informed investment decisions."
    st.write(txt4)
    st.write(txt5)
    st.write(txt6)
    
    st.markdown("---")
    st.header("Quarterly Earnings Analysis")
    st.write("Monitor and analyze quarterly earnings reports for insights.Adjust strategy based on emerging trends in individual companies.Backtest the strategy using historical data.Assess performance under various market conditions.")
    st.markdown("---")
    st.subheader("Strategy 3 -Personalised Investment Insights")
    st.header("Stock investment")
    st.write("A personalized stock investment strategy by first collecting user preferences for investment range, duration, and sectors of interest. It validates user inputs and, upon successful validation, retrieves historical stock data for a predefined list of banks. Utilizing the Prophet time-series forecasting model, it predicts future stock prices and potential profits for each bank over a specified duration. Then rank the stocks based on predicted profits, identifies the best buying option in the finance sector, and integrates with a language model to generate a phrase recommending the selected bank as a stock to invest in for a 6-month range. The results, including predicted profit, buying price, and safe exit selling price, are presented to the user for informed decision-making. Overall, it  combines financial analysis, machine learning, and natural language generation to provide users with tailored investment insights and a specific stock recommendation.")
    st.markdown("---")


    st.write("Work by")
    st.subheader('[![Keerthana G GitHub](https://img.shields.io/badge/Keerthana_G-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KeerthanaG23)') 
    st.subheader('[![Karthick N G GitHub](https://img.shields.io/badge/Karthick_N_G-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Karthick-ng)')

elif(selected=="Algorithmic Trading"):
    st.subheader("Algorithmic Trading")
    st.markdown("<h1 style='text-align: left; color: black;'>Traditional Algorithms</h1>",unsafe_allow_html=True)
    tickers = stock_df["Symbol"]
    comp=stock_df["Company Name"]
    search_query = st.text_input('Search for a company (e.g.TATA)')
    if (len(search_query)>0 or (search_query==" ")):
        filtered_options = [option for option in comp if search_query.lower() in option.lower()]
        selected_options = st.radio('Select option:', filtered_options)
        if len(selected_options)==0:
            st.write("Opps! We couldn't find that.")
        # st.write('Selected options:', selected_options)
        dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
        val = dict_csv.get(selected_options)  
        symb=val
        # st.write(symb[:])
        
        st.markdown("<h1 style='text-align: left; color: white;'>Traditional Approach</h1>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:18px; text-decoration: underline;'>Momentum/Dual Moving Average Crossover Strategy</span>", unsafe_allow_html=True)
        stock = yf.Ticker(symb)
        # print(stock)
        data = stock.history(start=start,end=end)
    #print(data.head())
        # st.write(data)
        data['momentum'] = data['Close'].pct_change()
        figure = make_subplots(rows=2, cols=1)
        figure.add_trace(go.Scatter(x=data.index,
                                    y=data['Close'],
                                    name='Close Price'))
        figure.add_trace(go.Scatter(x=data.index,
                                    y=data['momentum'],
                                    name='Momentum',
                                    yaxis='y2'))

        # Adding the buy and sell signals
        buy_markers = go.Scatter(x=data.loc[data['momentum'] > 0].index,
                                y=data.loc[data['momentum'] > 0]['Close'],
                                mode='markers', name='Buy',
                                marker=dict(color='green', symbol='triangle-up'))

        sell_markers = go.Scatter(x=data.loc[data['momentum'] < 0].index,
                                y=data.loc[data['momentum'] < 0]['Close'],
                                mode='markers', name='Sell',
                                marker=dict(color='red', symbol='triangle-down'))

        figure.add_trace(buy_markers)
        figure.add_trace(sell_markers)

        # Updating layout
        figure.update_layout(title='Algorithmic Trading using Momentum Strategy',
                            xaxis_title='Date',
                            yaxis_title='Price')
        figure.update_yaxes(title="Momentum", secondary_y=True)
        # Using Streamlit to display the plot
        st.plotly_chart(figure)
        sma30 = data['Close'].rolling(window=30).mean()
        sma100 = data['Close'].rolling(window=100).mean()
        x=pd.DataFrame()
        x['Company']=data['Close']
        x['SMA30']=sma30
        x['SMA100']=sma100
        plot_data = pd.DataFrame({'Close': data['Close'], 'SMA30': sma30, 'SMA100': sma100})
        fig = px.line(plot_data, x=plot_data.index, y=['Close', 'SMA30', 'SMA100'], labels={'value': 'Close Price USD($)'},
              title="Close Price History", template="plotly_dark")
        st.plotly_chart(fig)
        # st.write("Signaling when to buy or sell stock")


        def buySell(x):
            sigPriceBuy = []
            sigPriceSell = []
            flag = -1
            for i in range(len(x)):
                if x ['SMA30'][i] > x['SMA100'][i]:
                    if flag != 1:
                        sigPriceBuy.append(x['Company'][i])
                        sigPriceSell.append(np.nan)
                        flag = 1
                    else:
                        sigPriceBuy.append(np.nan)
                        sigPriceSell.append(np.nan)
                elif x['SMA30'][i] < x['SMA100'][i]:
                    if flag != 0:
                        sigPriceBuy.append(np.nan)
                        sigPriceSell.append(x['Company'][i])
                        flag = 0
                    else:
                        sigPriceBuy.append(np.nan)
                        sigPriceSell.append(np.nan)
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            return(sigPriceBuy, sigPriceSell)
        buySell=buySell(x)
        x['Buy Signal Price'] = buySell[0]
        x['Sell Signal Price'] = buySell[1]
        # st.write(x)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.title(' Close Price History Buy and Sell Signals')
        plt.style.use('classic')
        plt.figure(figsize=(12, 5))
        plt.plot(x['Company'], alpha=0.35)
        plt.plot(x['SMA30'], label='SMA30', alpha=0.35)
        plt.plot(x['SMA100'], label='SMA100', alpha=0.35)
        plt.scatter(x.index, x['Buy Signal Price'], label='Buy', marker='^', color='green')
        plt.scatter(x.index, x['Sell Signal Price'], label='Sell', marker='v', color='red')
        plt.title(' Close Price History Buy and Sell Signals')
        plt.xlabel(f"{start} - {end}")
        plt.ylabel("Adj Close Price USD($)")
        plt.legend(loc='upper left')
        st.pyplot()  
        st.markdown("<h1 style='text-align: left; color: black;'>Integrated Adaptive Trading Algorithm (IATA)</h1>",unsafe_allow_html=True)
        st.subheader("1. Random data sampling (Using static data for performance)")
        data = {'Bank': ['JPMorgan', 'Bank of America', 'Wells Fargo', 'Citigroup', 'Goldman Sachs', 
             'Morgan Stanley', 'Barclays', 'HSBC', 'UBS', 'Deutsche Bank',
             'Bank of China', 'ICBC', 'BNP Paribas', 'Santander', 'Mitsubishi UFJ',
             'Bank of Montreal', 'PNC Financial', 'BBVA', 'ING Group', 'Commonwealth Bank',
             'Sumitomo Mitsui', 'Royal Bank of Canada', 'Societe Generale', 'Toronto-Dominion',
             'Westpac Banking'],'NIM': [2.5, 3.0, 2.8, 2.2, 3.5, 2.0, 3.2, 2.7, 3.3, 2.4, 2.9, 3.1, 2.6, 3.4, 2.3, 3.2, 2.8, 3.0, 2.1, 3.0, 2.5, 3.3, 2.7, 2.9, 3.1],'NPA': [3.2, 2.5, 4.0, 1.8, 5.0, 2.0, 3.8, 4.5, 2.3, 3.5, 2.8, 4.2, 3.0, 2.7, 3.9, 1.6, 4.5, 2.4, 3.2, 3.0, 2.1, 4.0, 3.3, 2.9, 3.7],'P/B Ratio': [1.2, 1.5, 1.0, 1.8, 0.8, 1.7, 1.3, 1.1, 1.4, 1.6, 1.2, 1.1, 1.5, 1.3, 1.0, 1.8, 0.9, 1.4, 1.6, 1.2, 1.5, 1.8, 1.1, 1.3, 1.7],}
        col1, col2 = st.columns([1, 1])
        df = pd.DataFrame(data)
        # col1.write(df)
        
        df['NIM_NPA_Difference'] = df['NIM'] - df['NPA']
        
        # col2.write(df)
        
        
        df_sorted = df.sort_values(by='P/B Ratio', ascending=False)
        col1.write(df_sorted[['Bank','NIM','NPA','P/B Ratio']])
        col2.subheader("Valuation Metric")
        col2.write("Low P/B Ratio - Undervaluation, Potential Bargain, Financial Distress, Value Investing Opportunity")
        col2.write("High P/B Ratio - Market Confidence, Expectations of Future Growth, Industry Comparison, Potential Overvaluation")
        col2.write("Banks with higher Price-to-Book (P/B) Ratios are viewed favorably, indicating investor confidence and positive growth expectations. The positive difference between Net Interest Margin (NIM) and Non-Performing Assets (NPA) underscores effective income generation and risk management. A combination of a low P/B Ratio and a positive NIM-NPA difference suggests financial robustness, making these banks potentially attractive investments. However, thorough analysis considering external factors is crucial. Diversification across banks with varying P/B Ratios and NIM-NPA differences can mitigate risks. Investors should also monitor regulatory conditions and economic trends for informed decision-making.")
        max_diff_row = df.loc[df['NIM_NPA_Difference'].idxmax()]
        col1.subheader("2. Bank with minimum P/B Ratio & Higher NIM, Lower NPA")
        col1.success(df[(df['Bank'] == max_diff_row['Bank']) & (df['NIM_NPA_Difference'] > 0)].iloc[0]['Bank'])

        def generate_random_sector():
            sectors = ['Technology', 'E-commerce', 'Automotive', 'Healthcare', 'Consumer Goods', 'Beverages']
            return random.choice(sectors)
        companys=['Apple', 'Microsoft', 'Amazon', 'Google', 'Tesla', 'Facebook', 'Johnson & Johnson', 'Procter & Gamble', 'Intel', 'Coca-Cola',
                        'IBM', 'General Electric', 'Netflix', 'Ford', 'Oracle', 'Pfizer', 'Walmart', 'Boeing', 'AT&T', 'General Motors',
                        'Merck', 'Verizon', 'Chevron', 'Cisco', 'PepsiCo']

        sectors = ['Technology', 'E-commerce', 'Automotive', 'Healthcare', 'Consumer Goods', 'Beverages']

        sec_company = []
        for company in companys:
            if company in ['Amazon', 'Walmart', 'Alibaba']:
                sec_company.append('E-commerce')
            elif company in ['Tesla', 'Ford', 'General Motors']:
                sec_company.append('Automotive')
            elif company in ['Johnson & Johnson', 'Pfizer', 'Merck']:
                sec_company.append('Healthcare')
            elif company in ['Procter & Gamble', 'Coca-Cola', 'PepsiCo']:
                sec_company.append('Consumer Goods')
            elif company in ['Coca-Cola', 'PepsiCo']:
                sec_company.append('Beverages')
            else:
                sec_company.append('Technology')

        # Synthetic data for 25 companies
        companies_data = {
            'Company': ['Apple', 'Microsoft', 'Amazon', 'Google', 'Tesla', 'Facebook', 'Johnson & Johnson', 'Procter & Gamble', 'Intel', 'Coca-Cola',
                        'IBM', 'General Electric', 'Netflix', 'Ford', 'Oracle', 'Pfizer', 'Walmart', 'Boeing', 'AT&T', 'General Motors',
                        'Merck', 'Verizon', 'Chevron', 'Cisco', 'PepsiCo'],
            'Sector':  sec_company,
            'P/E Ratio': [random.randint(20, 40) for _ in range(25)],
            'Q1 Earnings': [round(random.uniform(1.5, 5.0), 2) for _ in range(25)],
            'Q2 Earnings': [round(random.uniform(1.5, 5.0), 2) for _ in range(25)],
            'Q3 Earnings': [round(random.uniform(1.5, 5.0), 2) for _ in range(25)],
            'Q4 Earnings': [round(random.uniform(1.5, 5.0), 2) for _ in range(25)],
        }

        # Convert to DataFrame
        companies_df = pd.DataFrame(companies_data)
        col1.subheader("3. Representing the sum of quarterly earnings for each company:")
        col1.write(companies_df)
        companies_df['Yearly']=companies_df['Q1 Earnings']+companies_df['Q2 Earnings']+companies_df['Q3 Earnings']+companies_df['Q4 Earnings']
        # col2.write(companies_df)
        sector_sum = companies_df.groupby('Sector')['Yearly'].mean()

        sorted_sector_sum = sector_sum.sort_values(ascending=False)
        col2.subheader("4. Average P/E ratio each sector")
        col2.write(sorted_sector_sum)
        companies_df_sorted = companies_df.sort_values(by='Sector', ascending=False)
        col2.subheader("5. Selecting companies which have lesser average earnings than its sector earnings")
        filtered_companies = companies_df[companies_df['Yearly'] < companies_df['Sector'].map(sector_sum)]
        filtered_companies.reset_index(drop=True)
        col2.write(filtered_companies)
        col1.write("High P/E Ratio - High Growth Expectations, Positive Market Sentiment, Industry Leaders, Tech and Growth Stocks")
        col1.write("Low P/E Ratio - Value Investing Opportunities, Stable or Mature Industries, Cyclical Industries, Perceived Risks")
        st.title("EquiAlgo: Precision in Indian Equities Algorithmic Trading")

        cufflinks.go_offline()
        cufflinks.set_config_file(world_readable=True, theme='pearl')
        from datetime import datetime
        
        st.image("https://imgur.com/fEgI9b6.png")
        st.write("The NIFTY 50 index is National Stock Exchange of India's benchmark broad based stock market index for the Indian equity market. NIFTY 50 - This represents the first 50 companies from the NIFTY 100 based on full market capitalisation from the eligible universe to track the performance of companies having large market caps.")
        col1, col2 = st.columns([1, 1])
        nifty_50 = pd.read_csv('./datasets/NIFTY 50.csv',parse_dates=["Date"])
        # nifty_50.head()
        col1.write(nifty_50)
        columns_info = {
    "Open and Close Columns": "Indicate the opening and closing price of the stocks on a particular day.",
    "High and Low Columns": "Provide the highest and the lowest price for the stock on a particular day, respectively.",
    "Volume Column": "Tells us the total volume of stocks traded on a particular day.",
    "Turnover Column": "Refers to the total value of stocks traded during a specific period of time.",
    "P/E Ratio": "Relates a company's share price to its earnings per share.",
    "P/B Ratio": "Measures the market's valuation of a company relative to its book value.",
    "Dividend Yield": "The amount of money a company pays shareholders (over the course of a year) for owning a share of its stock divided by its current stock price—displayed as a percentage."
}
        
        for column, description in columns_info.items():    
            with col2:
               st.success(f"{column}: {description}")
     
        nifty_50.isnull().sum()
        nifty_50.fillna(method='ffill',inplace=True)


        # Assume nifty_50 is your DataFrame
        col1.title("Visualising the NIFTY 50 data")
        # Plot NIFTY 50 High vs Low Trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nifty_50['Date'],
            y=nifty_50['High'],
            name='High Price',
            line=dict(color='blue'),
            opacity=0.8
        ))

        fig.add_trace(go.Scatter(
            x=nifty_50['Date'],
            y=nifty_50['Low'],
            name='Low Price',
            line=dict(color='orange'),
            opacity=0.8
        ))

        fig.update_layout(title_text='NIFTY 50 High vs Close Trend', plot_bgcolor='rgb(250, 242, 242)', yaxis_title='Value')

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Plot NIFTY 50 Closing Price
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nifty_50['Date'],
            y=nifty_50['Close'],
            name='Closing Price',
            line=dict(color='blue'),
            opacity=0.8
        ))

        fig.update_layout(title_text='NIFTY 50 Closing Price', plot_bgcolor='rgb(250, 242, 242)', yaxis_title='Value')

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Plot P/E vs P/B Ratio
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nifty_50['Date'],
            y=nifty_50['P/E'],
            name='P/E Ratio',
            line=dict(color='green'),
            opacity=0.8
        ))

        fig.add_trace(go.Scatter(
            x=nifty_50['Date'],
            y=nifty_50['P/B'],
            name='P/B Ratio',
            line=dict(color='orange'),
            opacity=0.8
        ))

        fig.update_layout(title_text='P/E vs P/B Ratio', plot_bgcolor='rgb(250, 242, 242)', yaxis_title='Value')

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Additional text information
        st.write("Whether P/E is better or P/B depends on the industry in question. Sometimes you need to take both into account to get an accurate picture of a company’s health and its financial prospects.")
        st.subheader("Market Performance post 2019")

        # Market performance post-2019
        nifty_50_2019 = nifty_50[nifty_50['Date'] >= '2019-01-01']
        df = nifty_50_2019

        # Candlestick plot
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'])])

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Line plot with Range Slider and Selectors
        fig = px.line(nifty_50_2019, x='Date', y='Close', title='Time Series with Range Slider and Selectors')

        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(plot_bgcolor='rgb(250, 242, 242)',
                        title='NIFTY_50: Major single day falls - 2019 onwards',
                        yaxis_title='NIFTY 50 Stock',
                        shapes=[dict(x0='2020-03-23', x1='2020-03-23', y0=0, y1=1, xref='x', yref='paper', line_width=2,
                                        opacity=0.3, line_color='red', editable=False),
                                dict(x0='2019-09-3', x1='2019-09-3', y0=0, y1=1, xref='x', yref='paper', line_width=3,
                                        opacity=0.3, line_color='red'),
                                dict(x0='2020-02-1', x1='2020-02-1', y0=0, y1=1, xref='x', yref='paper', line_width=3,
                                        opacity=0.3, line_color='red'),
                                dict(x0='2020-03-12', x1='2020-03-12', y0=0, y1=1, xref='x', yref='paper', line_width=3,
                                        opacity=0.3, line_color='red')],
                        annotations=[dict(x='2020-03-23', y=0.5, xref='x', yref='paper',
                                            showarrow=False, xanchor='left', text='Lockdown Phase-1 announced'),
                                    dict(x='2019-09-3', y=0.05, xref='x', yref='paper',
                                            showarrow=False, xanchor='left', text='Multiple PSU Bank Merger Announcements'),
                                    dict(x='2020-02-1', y=0.5, xref='x', yref='paper',
                                            showarrow=False, xanchor='right', text='Union Budget, coronavirus pandemic'),
                                    dict(x='2020-03-12', y=0.3, xref='x', yref='paper',
                                            showarrow=False, xanchor='right', text='Coronavirus declared Pandemic by WHO')]
                        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.subheader("Year 2020 saw a massive fall in the NIFTY 50 index which is prominent in the graph above. The effect can be seen majorly when the first phase of the lockdown started.")

        # Assume nifty_50_2019 is your DataFrame

        # Plot Time Series with Range Slider and Selectors
        fig = px.line(nifty_50_2019, x='Date', y='Close', title='NIFTY_50: Major single day gains - 2019 onwards')

        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        fig.update_layout(plot_bgcolor='rgb(250, 242, 242)',
                        title='NIFTY_50 : Major single day gains - 2019 onwards',
                        yaxis_title='NIFTY 50 Stock',
                        shapes=[dict(x0='2019-05-20', x1='2019-05-20', y0=0, y1=1, xref='x', yref='paper', line_width=2,
                                        opacity=0.3, line_color='green', editable=False),
                                dict(x0='2020-05-23', x1='2020-05-23', y0=0, y1=1, xref='x', yref='paper', line_width=3,
                                        opacity=0.3, line_color='green'),
                                dict(x0='2019-09-20', x1='2019-09-20', y0=0, y1=1, xref='x', yref='paper', line_width=3,
                                        opacity=0.3, line_color='green'),
                                dict(x0='2020-04-07', x1='2020-04-07', y0=0, y1=1, xref='x', yref='paper', line_width=3,
                                        opacity=0.3, line_color='green')],
                        annotations=[dict(x='2019-05-20', y=0.54, xref='x', yref='paper',
                                            showarrow=False, xanchor='right', text='Exit-Polls predict majority'),
                                    dict(x='2019-05-20', y=0.5, xref='x', yref='paper',
                                            showarrow=False, xanchor='right', text='for BJP government'),
                                    dict(x='2019-09-3', y=0.08, xref='x', yref='paper',
                                            showarrow=False, xanchor='left', text='2019 General Elections'),
                                    dict(x='2019-09-3', y=0.05, xref='x', yref='paper',
                                            showarrow=False, xanchor='left', text='results announced'),
                                    dict(x='2019-09-20', y=0.54, xref='x', yref='paper',
                                            showarrow=False, xanchor='left', text='cut in the corporate tax rate announced'),
                                    dict(x='2020-04-07', y=0.3, xref='x', yref='paper',
                                            showarrow=False, xanchor='right', text='Italy Coronavirus Nos went down')]
                        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)
        st.write("Due to reduction in the number of the Coronavirus cases in India, a positive sentiment was generated which translated to gain in the NIFTY index.")
        st.header("Performance of other  Sectoral Indices in 2020")
        nifty_auto = pd.read_csv('./datasets/NIFTY AUTO.csv',parse_dates=["Date"])
        nifty_bank = pd.read_csv('./datasets/NIFTY BANK.csv',parse_dates=["Date"])
        nifty_fmcg = pd.read_csv('./datasets/NIFTY FMCG.csv',parse_dates=["Date"])
        nifty_IT = pd.read_csv('./datasets/NIFTY IT.csv',parse_dates=["Date"])
        nifty_metal = pd.read_csv('./datasets/NIFTY METAL.csv',parse_dates=["Date"])
        nifty_pharma = pd.read_csv('./datasets/NIFTY PHARMA.csv',parse_dates=["Date"])
        nifty_auto_2019 = nifty_auto[nifty_auto['Date'] > '2019-12-31']
        nifty_bank_2019 = nifty_bank[nifty_bank['Date'] > '2019-12-31']
        nifty_fmcg_2019 = nifty_fmcg[nifty_fmcg['Date'] > '2019-12-31']
        nifty_IT_2019 = nifty_IT[nifty_IT['Date'] > '2019-12-31']
        nifty_metal_2019 = nifty_metal[nifty_metal['Date'] > '2019-12-31']
        nifty_pharma_2019 = nifty_pharma[nifty_pharma['Date'] > '2019-12-31']

        d = {'NIFTY Auto index': nifty_auto_2019['Close'].values, 
            'NIFTY Bank index': nifty_bank_2019['Close'].values,
            'NIFTY FMCG index': nifty_fmcg_2019['Close'].values,
            'NIFTY IT index': nifty_IT_2019['Close'].values,
            'NIFTY Metal index': nifty_metal_2019['Close'].values,
            'NIFTY Pharma index': nifty_pharma_2019['Close'].values,
            }
        df = pd.DataFrame(data=d)
        df.index=nifty_auto_2019['Date']
        fig = px.box(df)
        st.plotly_chart(fig)
        # Display concise information
        st.title("Investment Strategy")

        # Explanation of P/E ratio
        st.write(
            "The Price-to-Earnings (P/E) ratio is a crucial metric in investment decisions, providing a snapshot of a company's valuation. "
            "This ratio, calculated by dividing the stock price by earnings per share (EPS), serves as a key indicator for investors. "
            "A high P/E may signal market optimism about future growth, prompting a closer look at company fundamentals. Conversely, a low P/E might suggest undervaluation, though thorough investigation is necessary to understand the underlying reasons."
        )

        # Strategy details
        st.write(
            "When employing P/E ratios in your strategy, it's essential to consider sector benchmarks. "
            "Comparing a company's P/E against the sector average provides insights into relative valuation. "
            "A high P/E compared to the sector may be justified by strong company performance, while a lower P/E warrants investigation into potential undervaluation."
        )

        # Economic considerations
        st.write(
            "Macro and microeconomic factors play crucial roles in this strategy. "
            "Macro trends, such as economic conditions and industry outlooks, impact investor sentiment and, consequently, stock valuations. "
            "On the micro level, delving into company fundamentals, competitive positioning, and management quality is imperative."
        )

        # Risk assessment
        st.write(
            "In the investment decision-making process, it's vital to assess risks. "
            "Analyzing both macroeconomic trends and industry-specific risks helps formulate a comprehensive risk management strategy. "
            "Adopting a long-term perspective aligned with financial goals ensures that investments withstand short-term market fluctuations."
        )

        # Conclusion
        st.write(
            "By integrating P/E analysis with macro and microeconomic considerations, investors can make well-informed decisions, fostering a balanced and resilient investment portfolio. "
            "Regular reassessment and adaptation to changing market dynamics are integral to successful long-term investing."
        )   

    else:
        st.write("Please search for trading strategies...")


        
elif selected == "Personalized Investment Insights":
    st.subheader("Personalized Investment Insights")
    
    default_sectors_of_interest = ["Finance"]

    # User inputs
    investment_range_options = ["Low", "Medium", "High"]
    investment_range = st.selectbox("Select Investment Range", investment_range_options)

    duration_options = ["Short-term", "Mid-term","Long-term" ]

    duration = st.radio("Select Duration", duration_options)

    sectors_of_interest_options = ["Tech", "Healthcare", "Finance", "Energy"]
    sectors_of_interest = st.multiselect("Select Sectors of Interest", sectors_of_interest_options, default=default_sectors_of_interest)



    # Display User Input
    st.write(f"*Selected Investment Price Range:* {investment_range}")
    st.write(f"*Selected Duration:* {duration} ")
    st.write(f"*Selected Sectors of Interest:* {sectors_of_interest[0]}")

    if (investment_range == 'Low' and duration == 'Mid-term' and 'Finance' in sectors_of_interest):
         with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['YESBANK.NS',
                    'UJJIVANSFB.NS',
                    'UCOBANK.NS',
                    'MAHABANK.NS',
                    'CENTRALBK.NS',
                    'PSB.NS',
                    'IOB.NS',
                    'EDELWEISS.NS',
                    'IDFCFIRSTB.NS',
                    'IDBI.NS',
                    'JMFINANCIL.NS']

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 0.5
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)
            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = '...Generate a phrase showcasing the bank as a stock to invest for a 6month range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":300, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')

    elif (investment_range == 'Low' and duration == 'Long-term' and 'Finance' in sectors_of_interest):
        with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['YESBANK.NS',
                    'UJJIVANSFB.NS',
                    'UCOBANK.NS',
                    'MAHABANK.NS',
                    'CENTRALBK.NS',
                    'PSB.NS',
                    'IOB.NS',
                    'EDELWEISS.NS',
                    'IDFCFIRSTB.NS',
                    'IDBI.NS',
                    'JMFINANCIL.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 1.5
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)

            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = f'...Generate a phrase showcasing the bank as a stock to invest for a {period} days range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":250, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')

    elif (investment_range == 'Low' and duration == 'Short-term' and 'Finance' in sectors_of_interest):
        with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['YESBANK.NS',
                    'UJJIVANSFB.NS',
                    'UCOBANK.NS',
                    'MAHABANK.NS',
                    'CENTRALBK.NS',
                    'PSB.NS',
                    'IOB.NS',
                    'EDELWEISS.NS',
                    'IDFCFIRSTB.NS',
                    'IDBI.NS',
                    'JMFINANCIL.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            period = int(0.03 * 365)

            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 0.03
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)

            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = f'...Generate a phrase showcasing the bank as a stock to invest for a {period} days range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":250, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')



    elif (investment_range == 'Medium' and duration == 'Mid-term' and 'Finance' in sectors_of_interest):
         with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['PNB.NS',
                    'DCBBANK.NS',
                    'BANKINDIA.NS',
                    'CUB.NS',
                    'J&KBANK.NS',
                    'UNIONBANK.NS',
                    'FEDERALBNK.NS',
                    'MANAPPURAM.NS',
                    'KARURVYSYA.NS',
                    'ABCAPITAL.NS',
                    'BANDHANBNK.NS',
                    'KTKBANK.NS',
                    'RBLBANK.NS',
                    'BANKBARODA.NS',
                    'M&MFIN.NS',
                    'FINOPB.NS',
                    'CSBBANK.NS',
                    'PFC.NS',
                    'INDIANB.NS',
                    'ICICIPRULI.NS',
                    'AUBANK.NS',
                    'CANBK.NS',
                    'IIFL.NS',
                    'PNBHOUSING.NS',
                    'SBIN.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 0.5
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)
            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = '...Generate a phrase showcasing the bank as a stock to invest for a 6month range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":300, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')

    elif (investment_range == 'Medium' and duration == 'Long-term' and 'Finance' in sectors_of_interest):
        with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['PNB.NS',
                    'DCBBANK.NS',
                    'BANKINDIA.NS',
                    'CUB.NS',
                    'J&KBANK.NS',
                    'UNIONBANK.NS',
                    'FEDERALBNK.NS',
                    'MANAPPURAM.NS',
                    'KARURVYSYA.NS',
                    'ABCAPITAL.NS',
                    'BANDHANBNK.NS',
                    'KTKBANK.NS',
                    'RBLBANK.NS',
                    'BANKBARODA.NS',
                    'M&MFIN.NS',
                    'FINOPB.NS',
                    'CSBBANK.NS',
                    'PFC.NS',
                    'INDIANB.NS',
                    'ICICIPRULI.NS',
                    'AUBANK.NS',
                    'CANBK.NS',
                    'IIFL.NS',
                    'PNBHOUSING.NS',
                    'SBIN.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 1.5
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)

            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = f'...Generate a phrase showcasing the bank as a stock to invest for a {period} days range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":250, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')

    elif (investment_range == 'Medium' and duration == 'Short-term' and 'Finance' in sectors_of_interest):
        with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['PNB.NS',
                    'DCBBANK.NS',
                    'BANKINDIA.NS',
                    'CUB.NS',
                    'J&KBANK.NS',
                    'UNIONBANK.NS',
                    'FEDERALBNK.NS',
                    'MANAPPURAM.NS',
                    'KARURVYSYA.NS',
                    'ABCAPITAL.NS',
                    'BANDHANBNK.NS',
                    'KTKBANK.NS',
                    'RBLBANK.NS',
                    'BANKBARODA.NS',
                    'M&MFIN.NS',
                    'FINOPB.NS',
                    'CSBBANK.NS',
                    'PFC.NS',
                    'INDIANB.NS',
                    'ICICIPRULI.NS',
                    'AUBANK.NS',
                    'CANBK.NS',
                    'IIFL.NS',
                    'PNBHOUSING.NS',
                    ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            period = int(0.03 * 365)

            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 0.03
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)

            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = f'...Generate a phrase showcasing the bank as a stock to invest for a {period} days range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":250, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')



    elif (investment_range == 'High' and duration == 'Mid-term' and 'Finance' in sectors_of_interest):
         with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['SBIN.NS' ,
                    'MFSL.NS',
                    'ICICIBANK.NS',
                    'AXISBANK.NS',
                    'MUTHOOTFIN.NS',
                    'BAJAJFINSV.NS',
                    'MOTILALOFS.NS',
                    'KOTAKBANK.NS',
                    'HDFCAMC.NS',
                    'SUNDARMFIN.NS',
                    'BAJAJHLDNG.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 0.5
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)
            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = '...Generate a phrase showcasing the bank as a stock to invest for a 6month range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":300, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')

    elif (investment_range == 'High' and duration == 'Long-term' and 'Finance' in sectors_of_interest):
        with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['SBIN.NS' ,
                    'MFSL.NS',
                    'ICICIBANK.NS',
                    'AXISBANK.NS',
                    'MUTHOOTFIN.NS',
                    'BAJAJFINSV.NS',
                    'MOTILALOFS.NS',
                    'KOTAKBANK.NS',
                    'HDFCAMC.NS',
                    'SUNDARMFIN.NS',
                    'BAJAJHLDNG.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 1.5
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)

            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = f'...Generate a phrase showcasing the bank as a stock to invest for a {period} days range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":250, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')

    elif (investment_range == 'High' and duration == 'Short-term' and 'Finance' in sectors_of_interest):
        with st.spinner('Loading...'):
            time.sleep(2)  # Simulating some processing time, replace with actual data processing

            start = datetime.date(2019, 1, 1)
            end_f = datetime.date.today()
            end= end_f - timedelta(days=1)
            stock_df = pd.read_csv("./datasets/TickersData.csv")
            tickers = stock_df["Company Name"]
            yesterday = end_f - timedelta(days=2)
            array = ['SBIN.NS' ,
                    'MFSL.NS',
                    'ICICIBANK.NS',
                    'AXISBANK.NS',
                    'MUTHOOTFIN.NS',
                    'BAJAJFINSV.NS',
                    'MOTILALOFS.NS',
                    'KOTAKBANK.NS',
                    'HDFCAMC.NS',
                    'SUNDARMFIN.NS',
                    'BAJAJHLDNG.NS' ]

            dict_csv = pd.read_csv('./datasets/TickersData.csv', header=None, index_col=0).to_dict()[1]
            symb_list = []
            for i in array:
                val = dict_csv.get(i)
                symb_list.append(val)

            data = yf.download(array, start=yesterday, end=end)
            data = data['Close']
            data.reset_index(inplace=True)
            melted_data = pd.melt(data, id_vars=['Date'], value_vars=array, var_name='Ticker', value_name='Price')

            sorted_data = melted_data.sort_values(by='Price', ascending=True)

            new_t = []
            for i in sorted_data['Ticker']:
                new_t.append(i)

            result_df = pd.DataFrame(columns=['Ticker', 'PredictedProfit', 'Buying_Price', 'Exit_safe_price'])

            a = new_t
            period = int(0.03 * 365)

            for i in a:
                data = yf.download(i, start=start, end=end)
                data.reset_index(inplace=True)
                n_years = 0.03
                period = int(n_years * 365)
                df_train = data[['Date', 'Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})  # rename columns

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                end = date.today()

                filtered_df = forecast[forecast['ds'] > end_f]
                maxx_pred = filtered_df['yhat_upper'].max()
                a = sorted_data[sorted_data['Ticker'] == i]
                predicted_profit = maxx_pred - a['Price'].iloc[0]

                buy_price = sorted_data[sorted_data["Ticker"] == i]['Price'].iloc[0]
                result_df.loc[len(result_df)] = [i, predicted_profit, buy_price, maxx_pred]
            result_df=result_df.sort_values(by='PredictedProfit', ascending=False)

            os.environ["REPLICATE_API_TOKEN"] = "r8_drcPJJI0QkgcMSShXbRc9WmbtQdpjdV2e7nJv"
            pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
            input=result_df[:1]
            prompt_input = f'...Generate a phrase showcasing the bank as a stock to invest for a {period} days range.provide response to just this and nothing else'

            # Generate LLM response
            output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', # LLM model
                                    input={"prompt": f"{pre_prompt} {input} {prompt_input} Assistant: ", # Prompts
                                    "temperature":0.2, "top_p":0.9, "max_length":250, "repetition_penalty":1})
            
            full_response = ""

            for item in output:
                full_response += item

            st.success(full_response)
            
            st.write("## Result Summary")
            st.write(result_df.reset_index(drop=True))

            st.write('## Best Buying option in Finance sector')
            cname = result_df[:1]['Ticker'].iloc[0]
            st.write(f"*Company Name:* {cname}")

            profit = result_df[:1]['PredictedProfit'].iloc[0]
            st.write(f'*Predicted Profit:* {round(profit,2)}')

            buy_p = result_df[:1]['Buying_Price'].iloc[0]
            st.write(f'*Buying Price(today):* {round(buy_p,2)}')

            sell_p = result_df[:1]['Exit_safe_price'].iloc[0]
            st.write(f'*Safe Exit Selling Price:* {round(sell_p,2)}')
    else:
        st.write("poda")
        st.warning("## PLEASE SELECT All INPUT FIELDS")
        


elif(selected=="Multifactor Market Dynamics"):

    st.markdown('<h1 style="text-align:center; color:Gray;">Trade Based on News Headlines Using NLP</h1>', unsafe_allow_html=True)
    html_message = '<p style="color:black; font-size:20px; font-weight:bold; text-align:left;">Scrap news data from an indian news website and vectorize each article.Assign score to each word in the text based on how positive or negative it is. It then combines these scores to give an overall sentiment score for the text.Get a grip around which sectors are bullish and bearish.</p>'

    data = {
    'Symbol': ['AU', 'BM', 'BX', 'CD', 'CDGS', 'CG', 'CPSE', 'EG', 'FMCG', 'FN', 'HC', 'ID', 'II', 'IT', 'MT', 'ONG', 'PSU', 'PWR', 'RE', 'TC', 'Teck', 'UT'],
    'Sector': ['Auto', 'Basic Materials', 'Bankex', 'Consumer Durables', 'Consumer Discretionary Goods & Services', 'Capital Goods', 'CPSE', 'Energy', 'Fast Moving Consumer Goods', 'Financials', 'Healthcare', 'Industrials', 'India Infrastructure', 'Information Technology', 'Metal', 'Oil & Gas', 'PSU', 'Power', 'Realty', 'Telecom', 'Teck', 'Utilities'],
    'Number_of_Stocks': [15, 189, 10, 12, 297, 25, 52, 27, 81, 139, 96, 203, 30, 62, 10, 10, 56, 11, 10, 17, 28, 24]}

    df = pd.DataFrame(data)

    # Streamlit UI
    # st.title('Number of Stocks by Sector')

    # Dropdown to select sector
    selected_sector = st.selectbox('Select a Sector:', df['Sector'])

    # Display the number of stocks for the selected sector
    if selected_sector in df['Sector'].values:
        num_stocks = df.loc[df['Sector'] == selected_sector, 'Number_of_Stocks'].values[0]
        st.write(f'The number of stocks in {selected_sector} sector is: {num_stocks}')
    else:
        st.write(f'Sorry, the selected sector "{selected_sector}" is not available in the dataset.')
        # Display the HTML message
    st.markdown(html_message, unsafe_allow_html=True)


    model = BertForSequenceClassification.from_pretrained(r"ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    sentences = ["RBI loses 5% economy" ]
    results = nlp(sentences)
    print(results)
    url = 'https://economictimes.indiatimes.com/markets/stocks/news'


    def get_economic_times_headlines(url, num_pages=3):
        all_headlines = []

        for page in range(1, num_pages + 1):
            page_url = f"{url}?page={page}"
            response = requests.get(page_url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                headlines = soup.find_all('div', class_='eachStory')  # Example selector
                headlines_array = [headline.find('a').text.strip() for headline in headlines]
                all_headlines.extend(headlines_array)
            else:
                print(f"Failed to retrieve content for page {page}. Status code: {response.status_code}")

        return all_headlines    
    news_url = "https://economictimes.indiatimes.com/industry/banking/finance/banking"
    all_headlines = get_economic_times_headlines(news_url, num_pages=3)
    results = nlp(all_headlines)
    labels = [item['label'] for item in results]
    # print(labels)
    df = pd.DataFrame({'Headlines': all_headlines, 'Labels': labels})
    st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        {df.to_html(classes='dataframe', escape=False)}
    </div>
    """,
    unsafe_allow_html=True)

# Print the DataFrame
    # st.write(df)
    p_ct=df[df['Labels']=='positive'].count()
    n_ct=df[df['Labels']=='negative'].count()
    if (len(p_ct)>len(n_ct)):
       st.markdown("<br>", unsafe_allow_html=True)
       st.success(f"Positive News: The {selected_sector} sector is currently in a BULL market.")
       


    else:
        st.markdown("<br>", unsafe_allow_html=True)
        # Custom alert message using Streamlit
        st.error(f"Alert: The {selected_sector} sector is currently in a BEAR market.")
else:
   
    col1, col2 = st.columns([1, 1])
    col2.markdown(
    """
    <style>
        body {
            color: #333;
            background-color: #f8f9fa;
        }
        .st-bw {
            background-color: #8270e6 ;
            color: #ffffff;
        }
        .st-cc {
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True)
    
    col1.subheader("Stay updated with live market data and insights")
    sym = stock_df
    col1.write(sym[:10])
    # ticker_symbol = col2.text_input("Choose a symbol (with .NS)", "AAPL")
    ticker_symbol = col2.selectbox("Choose a symbol (with .NS)", stock_df['Symbol'])
    col2.image("./static/bull.jpg",width=400)
    data = yf.download(ticker_symbol, start=start, end=end)
    st.subheader(f"Stock Price Data for {ticker_symbol}")
    st.dataframe(data.tail())
    st.subheader(f"Stock Price Chart for {ticker_symbol}")
    st.line_chart(data['Close'])
    st.header("Market Insights")
    st.header("Latest News")
    if(ticker_symbol!="AAPL"):
        ticker_symbol = ticker_symbol[:-3]
    news_api_key = '8b0eb0ffb4964f63bb1f32e6fc16cf3e'  
    news_url = f'https://newsapi.org/v2/everything?q={ticker_symbol}&apiKey={news_api_key}&sortBy=publishedAt'
    response = requests.get(news_url)
    news_data = response.json()

    # Display news articles
    if 'articles' in news_data:
        for article in news_data['articles']:
            st.write(f"**{article['title']}**")
            st.write(article['description'])
            st.write(f"Source: {article['source']['name']}, Published At: {article['publishedAt']}")
            st.markdown("---")
    else:
        st.warning("Unable to fetch news at the moment. Please try again later.")

    # Footer
    st.markdown("---")
    st.markdown("Created by **Karthick N G** & **Keerthana G**  ")


   
