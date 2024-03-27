import datetime
import streamlit as st, pandas as pd, numpy as np
from datetime import date
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go

st.title('Financial Dashboard 📈')
st.image('abc.jpg')


ticker=st.sidebar.text_input('Name of Stock',value="AAPL")
start_date= st.sidebar.date_input('Start Date',datetime.date(2024, 1, 1))
end_date= st.sidebar.date_input('End Date')
data=yf.download(ticker,start=start_date,end=end_date)
api_key = 'kR3pOO0UkPTyu5b4gq6pYCmk4tk8mga6'
requestString = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
request = requests.get(requestString)
json_data = request.json()
data1 = json_data[0]
#st.write(data1)
Introduction, Liquidity_Ratio, Monte_Carlo_Simulation, Predictions = st.tabs(["Introduction","Liquidity_Ratio","Monte_Carlo_Simulation", "Predictions"])
 
with Introduction:
    with st.expander("About Company"):
        st.write(data1["description"])
    fig=px.line(data,x=data.index,y=data['Adj Close'],title= ticker)
    st.plotly_chart(fig)
    with st.expander("Price Movements"):
        data2 = data.copy()
        data2['% Change'] = data['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)

with Liquidity_Ratio:
    # Balance Sheet
    with st.expander("Liquidity_Ratio"):
        balance_sheet_url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={api_key}'
        balance_sheet_request = requests.get(balance_sheet_url)
        balance_sheet_data = balance_sheet_request.json()
        bs = pd.DataFrame(balance_sheet_data)
        bs.set_index('date', inplace=True)
        st.subheader('Balance Sheet')
        st.write(bs)
    # Income Statement
        income_statement_url = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={api_key}'
        income_statement_request = requests.get(income_statement_url)
        income_statement_data = income_statement_request.json()
        is1 = pd.DataFrame(income_statement_data)
        is1.set_index('date', inplace=True)
        st.subheader('Income Statement')
        st.write(is1)
    # Cash Flow Statement
        cash_flow_url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={api_key}'
        cash_flow_request = requests.get(cash_flow_url)
        cash_flow_data = cash_flow_request.json()
        cf = pd.DataFrame(cash_flow_data)
        cf.set_index('date', inplace=True)
        st.subheader('Cash Flow Statement')
        st.write(cf)


    # Check if bs contains data
    if not bs.empty:
    # Calculate ratios
        current_ratio = round(bs.iloc[0]['totalCurrentAssets'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
        quick_ratio = round((bs.iloc[0]['totalCurrentAssets'] - bs.iloc[0]['inventory']) / bs.iloc[0]['totalCurrentLiabilities'], 2)
        cash_ratio = round(bs.iloc[0]['cashAndCashEquivalents'] / bs.iloc[0]['totalCurrentLiabilities'], 2)

    # Create a table
        table_data = {
            "Metric": [ "Current Ratio", "Quick Ratio", "Cash Ratio", "Total Current Assets", "Total Current Liabilities","inventory", "cashAndCashEquivalents"],
            "Value": [
                current_ratio,
                quick_ratio,
                cash_ratio,
                bs.iloc[0]['totalCurrentAssets'],
                bs.iloc[0]['totalCurrentLiabilities'],
                bs.iloc[0]['inventory'],
                bs.iloc[0]['cashAndCashEquivalents'],
            ]
        }

    # Display the table
        st.table(table_data)
    else:
        st.write("Balance sheet data is empty. Please check your data.")




with Monte_Carlo_Simulation:
    data['Adj Close'].plot(figsize=(10, 6), title=f"Last year Stock Price chart of {ticker}")

# Calculating daily returns
    data['Returns'] = data['Adj Close'].pct_change()

# Setting the number of simulations and the number of days to project
    n_simulations = 100
    n_days = 30

    last_price = data['Adj Close'][-1]
    simulation_df = pd.DataFrame()

    for x in range(n_simulations):
        count = 0
        daily_volatility = data['Returns'].std()

        price_series = []

    # Generating the price list for the next year
        price = last_price * (1 + np.random.normal(0, daily_volatility))
        price_series.append(price)

        for y in range(n_days):
            if count == n_days + 1:
                break
            price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
            price_series.append(price)
            count += 1

        simulation_df[x] = price_series

# Plotting the simulation
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_df)
    plt.axhline(y=last_price, color='r', linestyle='-')
    plt.title(f'Monte Carlo Simulation for {ticker} Stock')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.tight_layout()

# Display the plot in Streamlit
    st.pyplot(plt)