import datetime
import streamlit as st, pandas as pd, numpy as np
from millify import millify
from datetime import date, timedelta
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
#from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from scipy import stats


st.set_page_config(
    page_title='Financial Analytics Dashboard',
    page_icon='📈',
    layout="centered",
)
st.title('Financial Analytics Dashboard 📈')
st.image('\stock_market.jpeg')

value = "JPM"
ticker = st.text_input('Enter Stock Ticker:', value=value).upper()

def generate_card(text: str) -> None:
    """
    Generates a styled card with a title and icon.

    Parameters:
        text (str): The title text for the card.
    """
    st.markdown(f"""
        <div style='border: 1px solid #e6e6e6; border-radius: 5px; padding: 10px; display: flex; justify-content: center; align-items: center'>
            <i class='fas fa-chart-line' style='font-size: 24px; color: #0072C6; margin-right: 10px'></i>
            <h3 style='text-align: center'>{text}</h3>
        </div>
         """, unsafe_allow_html=True)

# Function to convert selected duration to start and end dates
def get_start_end_dates(selected_duration):
    end_date = datetime.date.today()
    if selected_duration == "1 Year":
        start_date = end_date - datetime.timedelta(days=365)
    elif selected_duration == "5 Years":
        start_date = end_date - datetime.timedelta(days=5*365)
    elif selected_duration == "10 Years":
        start_date = end_date - datetime.timedelta(days=10*365)
    else:
        # For custom date selection, the function will return None.
        return None, None
    return start_date, end_date

# Add Multiple date selection option to the select box
selected_duration = st.selectbox('Select Duration', ["1 Year", "5 Years", "10 Years", "Custom Date"])

# Convert selected duration to start and end dates
start_date, end_date = get_start_end_dates(selected_duration)

# If the user selects "Custom Date", display a date input widget to select the start date
if selected_duration == "Custom Date":
    start_date = st.date_input("Select Start Date", value=datetime.date.today() - datetime.timedelta(days=365), min_value=datetime.date(1900, 1, 1), max_value=datetime.date.today())
    # Assuming you want to keep the end date as today for the custom date selection
    end_date = datetime.date.today()

data=yf.download(ticker,start=start_date,end=end_date)
# api_key = 'kR3pOO0UkPTyu5b4gq6pYCmk4tk8mga6' # Santhosh
api_key = 'eUMyuzTsDHEK7kI8KYxpBbaVS9yxrF4P' # Rituraj
requestString = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
request = requests.get(requestString)
json_data = request.json()
data1 = json_data[0]
# st.write(data1)


# Calculate the start and end dates for the last 10 years
#end_date = datetime.today()
start_date = end_date - timedelta(days=365.25*10)

# Download the monthly stock data for the last 10 years
stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
market_data = yf.download('^GSPC', start=start_date, end=end_date, interval="1mo")

col1, col2 = st.columns((8,2.5))
with col1:
    generate_card(data1['companyName'])
with col2:
    # display image and make it clickable
    image_html = f"<a href='{data1['website']}' target='_blank'><img src='{data1['image']}' alt='{data1['companyName']}' style='background-color: lightgrey; display: block; margin: auto;' height='80' width='80'></a>"
    st.markdown(image_html, unsafe_allow_html=True)

    # image_html = f"<a href='{data1['website']}' target='_blank'><img src='{data1['image']}' alt='{data1['companyName']}' height='75' width='75'></a>"
    # st.markdown(image_html, unsafe_allow_html=True)

col3, col4, col5 = st.columns((2 ,2.2 ,2.2))

with col3:
    generate_card(f"Price <br> {data1['price']:.2f}")

with col4:
    market_cap_formatted = millify(data1['mktCap'], precision=2)
    generate_card(f"Market Cap \n {market_cap_formatted}")
    # empty_lines(2)

with col5:
    # empty_lines(1)
    generate_card(data1['industry'])            
    # empty_lines(2)

st.write("\n\n")

Introduction, Financial_Ratio, Monte_Carlo_Simulation, Predictions, GoldenCross = st.tabs(["Overview","Financial Ratios","Monte Carlo Simulation", "Stock Price Predictions", "Golden Cross Signals"])

with Introduction:
    
    # with st.write("About Company"):
    company_name = data1['companyName']
    st.subheader(f"About {company_name}")
    st.write(data1["description"])
    st.markdown('---')
    st.subheader("Stock Data")
    fig=px.line(data,x=data.index,y=data['Adj Close'],title= f'Line chart of Adj. Close Price of {company_name} stock:')
    st.plotly_chart(fig)
    st.markdown('---')
    
    with st.expander(f"{company_name} Stock Data - OHLC"):
        data2 = data.copy()
        data2['% Change'] = data['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2,wide=True)

## Balance Sheet
    
balance_sheet_url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?apikey={api_key}'
balance_sheet_request = requests.get(balance_sheet_url)
balance_sheet_data = balance_sheet_request.json()
bs = pd.DataFrame(balance_sheet_data)
bs.set_index('date', inplace=True)

## Income Statement
income_statement_url = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?apikey={api_key}'
income_statement_request = requests.get(income_statement_url)
income_statement_data = income_statement_request.json()
is1 = pd.DataFrame(income_statement_data)
is1.set_index('date', inplace=True)

## Cash Flow Statement
cash_flow_url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?apikey={api_key}'
cash_flow_request = requests.get(cash_flow_url)
cash_flow_data = cash_flow_request.json()
cf = pd.DataFrame(cash_flow_data)
cf.set_index('date', inplace=True)

def create_bar_chart(df, column_name):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df[column_name], name=column_name))
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Cash Flow (USD)',
                      barmode='relative')  # 'relative' for stacked bar chart, 'group' for grouped bar chart
    return fig

with Financial_Ratio:

# Check if bs contains data
    st.subheader("Liquidity Ratios")    
    if not bs.empty:
    # Calculate ratios
        current_ratio = round(bs.iloc[0]['totalCurrentAssets'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
        quick_ratio = round((bs.iloc[0]['totalCurrentAssets'] - bs.iloc[0]['inventory']) / bs.iloc[0]['totalCurrentLiabilities'], 2)
        cash_ratio = round(bs.iloc[0]['cashAndCashEquivalents'] / bs.iloc[0]['totalCurrentLiabilities'], 2)

    # Define formulas for each ratio
        current_formula = "Total Current Assets / Total Current Liabilities"
        quick_formula = "(Total Current Assets - Inventory) / Total Current Liabilities"
        cash_formula = "Cash and Cash Equivalents / Total Current Liabilities"

    # Create a DataFrame for the table
        table_data = pd.DataFrame({
            "Metric": ["Current Ratio", "Quick Ratio", "Cash Ratio"],
            "Value": [current_ratio, quick_ratio, cash_ratio],
            "Formula": [current_formula, quick_formula, cash_formula]
        })

    # Display the table without index numbers
        st.dataframe(table_data)
        st.markdown('---')
    else:
        st.write("Balance sheet data is empty. Please check your data.")

   
    # Assuming is1, bs, and cf are dictionaries containing income statement, balance sheet, and cash flow statement data respectively
    #st.write(data1)
    st.subheader("Profitability Ratios")    
    if not is1.empty:
        if not bs.empty:
            if not cf.empty:
            # Calculate ratios
                current_ratio = round(bs.iloc[0]['totalCurrentAssets'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
                quick_ratio = round((bs.iloc[0]['totalCurrentAssets'] - bs.iloc[0]['inventory']) / bs.iloc[0]['totalCurrentLiabilities'], 2)
                cash_ratio = round(bs.iloc[0]['cashAndCashEquivalents'] / bs.iloc[0]['totalCurrentLiabilities'], 2)
                gpm = round((is1.iloc[0]['grossProfit'] / is1.iloc[0]['revenue']) * 100, 2)
                opm = round((is1.iloc[0]['operatingIncome'] / is1.iloc[0]['revenue']) * 100, 2)
                roa = round((is1.iloc[0]['netIncome'] / bs.iloc[0]['totalAssets']) * 100, 2)
                roe = round((is1.iloc[0]['netIncome'] - cf.iloc[0]['dividendsPaid']) / bs.iloc[0]['totalStockholdersEquity'] * 100, 2)
                ros = round((is1.iloc[0]['grossProfit'] - is1.iloc[0]['operatingExpenses']) / is1.iloc[0]['revenue'] * 100, 2)
                roi = round((is1.iloc[0]['netIncome'] / bs.iloc[0]['totalInvestments']) * 100, 2)

            # Define formulas for each ratio
                gpm_formula = "(Gross Profit / Revenue) * 100"
                opm_formula = "(Operating Income / Revenue) * 100"
                roa_formula = "(Net Income / Total Assets) * 100"
                roe_formula = "(Net Income - Dividends) / Total Stockholders Equity) * 100"
                ros_formula = "((Operating Income - Operating Expenses) / Revenue) * 100"
                roi_formula = "(Net Income / Cost of Investments or Investment Base) * 100"


            # Create a DataFrame for the table
                table_data = pd.DataFrame({
                    "Metric": ["Gross Profit Margin (%)", "Operating Profit Margin (%)", "Return on Assets (%)", "Return on Equity (%)", "Return on Sales (%)", "Return on Investment (%)"],
                    "Value": [gpm, opm, roa, roe, ros, roi],
                    "Formula": [gpm_formula, opm_formula, roa_formula, roe_formula, ros_formula, roi_formula]
                })

            # Display the table without index numbers
                st.dataframe(table_data)
                st.markdown('---')
            else:
                st.write("Income statement data is empty. Please check your data.")

# Assuming is1, bs, cf, and company_info are dictionaries containing data for income statement, balance sheet, cash flow statement, and company information respectively

    st.subheader("Earning Ratio")

    # Check if income statement (is1) contains data
    if not is1.empty:
        # Check if balance sheet (bs) and cash flow statement (cf) contain data
        if not bs.empty and not cf.empty:
            # Calculate ratios
            pe_ratio = round(data1['price'] / is1.iloc[0]['eps'], 2)
            dividend_payout_ratio = round(abs(cf.iloc[0]['dividendsPaid']) / is1.iloc[0]['netIncome'], 2)
            debt_to_equity_ratio = round(bs.iloc[0]['totalDebt'] / bs.iloc[0]['totalStockholdersEquity'], 2)
            sustainable_growth_rate = roe * (1 - dividend_payout_ratio)

            # Define formulas for each ratio
            pe_ratio_formula = "Price / Earnings per Share"
            dividend_payout_ratio_formula = "(Dividends Paid / Net Income)"
            debt_to_equity_ratio_formula = "(Total Debt / Total Stockholders Equity)"
            sustainable_growth_rate_formula = "(ROE * (1 - Dividend Payout Ratio))"

            # Create a DataFrame for the table
            table_data = pd.DataFrame({
                "Metric": ["Price-to-Earnings Ratio", "Dividend Payout Ratio",
                        "Debt-to-Equity Ratio", "Sustainable Growth Rate"],
                "Value": [pe_ratio, dividend_payout_ratio, debt_to_equity_ratio, sustainable_growth_rate],
                "Formula": [pe_ratio_formula, dividend_payout_ratio_formula,
                            debt_to_equity_ratio_formula, sustainable_growth_rate_formula]
            })

            # Display the table without index numbers
            st.dataframe(table_data)
            st.markdown('---')
        else:
            st.write("Balance sheet or cash flow statement data is empty. Please check your data.")
    


    
    st.subheader("Capital Asset Pricing Model - CAPM")

    # Ensure is1 DataFrame is not empty
    if not is1.empty:
        market_data['Returns'] = market_data['Adj Close'].pct_change()
        stock_data['Returns'] = stock_data['Adj Close'].pct_change()
        market_returns = market_data['Returns'].values[1:]
        stock_returns = stock_data['Returns'].values[1:]
        # CAPM and WACC

        beta = round(stats.linregress(market_returns, stock_returns)[0], 4)

        rf_rate = st.number_input(f"Risk Free Rate:",value=1.74)
        # Canada Long Term Real Return Bonds Rate - https://ycharts.com/indicators/canada_long_term_real_return_bonds_rate
        st.write(f"<small>Reference: https://ycharts.com/indicators/canada_long_term_real_return_bonds_rate</small>", unsafe_allow_html=True)
        rf = rf_rate
        
        gm = stats.gmean(1 + market_returns) - 1
        period = 12
        nominal_rate = gm * period
        rm = (((1 + (nominal_rate / period)) ** period) - 1) * 100
        cost_of_equity = round(rf + beta * (rm - rf), 2)

        st.write(f"<span style='font-size:1.2em;'>Risk Free Rate: {rf_rate}% <br>Beta - β: {beta:.2f} <br>Market Return (S&P 500) - Rm : {rm:.2f}%</span>", unsafe_allow_html=True)
        
        st.write(f"<span style='font-size:1.2em;'><b>Expected Return on Asset - Ra: {cost_of_equity}%</b></span>", unsafe_allow_html=True)
        st.write(f"[Expected Return on Asset = Risk-free Rate + Beta * (Market Return - Risk-free Rate]")
        
        st.markdown('---')
                

        # Check if 'totalEquity' key exists in bs DataFrame
        if 'totalEquity' in bs.columns:
            equity = bs.iloc[0]['totalEquity']
            debt = bs.iloc[0]['totalDebt']

            # Total Funding
            total_funding = equity + debt

            # Check if 'interestExpense' key exists in is1 DataFrame
            if 'interestExpense' in is1.columns:
                cost_of_debt = round(is1.iloc[0]['interestExpense'] / bs.iloc[0]['totalDebt'], 4) * 100
                
                # Check if 'incomeTaxExpense' and 'incomeBeforeTax' keys exist in is1 DataFrame
                if 'incomeTaxExpense' in is1.columns and 'incomeBeforeTax' in is1.columns:
                    # Calculate income tax rate
                    income_tax_rate = round(is1.iloc[0]['incomeTaxExpense'] / is1.iloc[0]['incomeBeforeTax'], 4)

                    # Define the calculate_wacc function
                    def calculate_wacc(market_value_of_equity, market_value_of_debt, cost_of_equity, cost_of_debt, corporate_tax_rate):
                        global equity_ratio,debt_ratio
                        total_value = market_value_of_equity + market_value_of_debt
                        equity_ratio = market_value_of_equity / total_value
                        debt_ratio = market_value_of_debt / total_value
                        
                        wacc = (equity_ratio * cost_of_equity) + (debt_ratio * cost_of_debt * (1 - corporate_tax_rate))
                        return wacc

                    # Calculate WACC
                    wacc_value = calculate_wacc(market_value_of_equity=equity,
                                                market_value_of_debt=debt,
                                                cost_of_equity=cost_of_equity,
                                                cost_of_debt=cost_of_debt,
                                                corporate_tax_rate=income_tax_rate)
                    
                    st.subheader("Weighted Average Cost of Capital - WACC")
                    
                    # st.write(str(equity))
                    # st.write(debt)
                    
                    st.write(f"""<span style='font-size:1.2em;'>                             
                             Equity: ${equity:,.2f} <br>
                             Debt: ${debt:,.2f} <br>
                             Total Assets: ${total_funding:,.2f} <br>
                             Weight of Equity: {equity_ratio:.2f} <br>
                             Weight of Debt: {debt_ratio:.2f} <br>
                             Cost of Equity: {cost_of_equity}% - <small>[Expected Return on Asset]</small><br>
                             Cost of Debt: {cost_of_debt}% - <small>[Interest Expense/Total Debt]</small><br>
                             Tax Rate: {income_tax_rate*100}% - <small>[Tax Expense/Income Before Tax]</small><br>
                             WACC: {wacc_value:.2f}% <br></span>""", unsafe_allow_html=True)
                    
                    st.markdown('---')

                  
                    st.subheader('Cash Flow Analysis')
                    selected_cash_flow = st.selectbox('Select Cash Flow Type', [ 'Free Cash Flow','Operating Activities', 'Investing Activities', 'Financing Activities'])

                    # Display corresponding chart based on the selected cash flow type
                    if selected_cash_flow == 'Operating Activities':
                        st.subheader('Cash Flow from Operating Activities')
                        st.plotly_chart(create_bar_chart(cf, 'netCashProvidedByOperatingActivities'))
                        st.markdown('These are the day-to-day business activities of a company, such as sales, purchases, and expenses. Operating cash flow reflects the cash generated or used by these activities, excluding investments and financing.')
                    elif selected_cash_flow == 'Investing Activities':
                        st.subheader('Cash Flow from Investing Activities')
                        st.plotly_chart(create_bar_chart(cf, 'netCashUsedForInvestingActivites'))
                        st.markdown('These activities involve the purchase and sale of long-term assets, such as property, plant, and equipment, as well as investments in securities. Investing cash flow indicates the cash spent on or received from these investments.')
                    elif selected_cash_flow == 'Financing Activities':
                        st.subheader('Cash Flow from Financing Activities')
                        st.plotly_chart(create_bar_chart(cf, 'netCashUsedProvidedByFinancingActivities'))
                        st.markdown('These activities involve raising capital and repaying debt, as well as transactions with shareholders. Financing cash flow shows the cash received from or used for financing the company''s operations.')
                    else:  # Free Cash Flow
                        st.subheader('Free Cash Flow Over Time')
                        st.plotly_chart(create_bar_chart(cf, 'freeCashFlow'))
                        st.markdown('Free Cash Flow (FCF) is the cash a company has left over after paying for its operations and investments, showing how much money it can use for growth or to pay investors.')
                        st.markdown('---')

                else:
                    st.write("Error: 'incomeTaxExpense' or 'incomeBeforeTax' not found in DataFrame.")
            else:
                st.write("Error: 'interestExpense' not found in DataFrame.")
        else:
            st.write("Error: 'totalEquity' not found in DataFrame.")
    else:
        st.write("Error: DataFrame 'is1' is empty.")

    st.subheader("Company Financial Statements - Past 5 years")
    with st.expander("Click to expand"):
        st.subheader('Balance Sheet')
        st.write(bs.T)
        st.subheader('Income Statement')
        st.write(is1.T)
        st.subheader('Cash Flow Statement')
        st.write(cf.T)
    
    
#### Monte Carlo Simulation section

with Monte_Carlo_Simulation:
    def fetch_stock_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    # Function to perform Monte Carlo simulation
    def monte_carlo_simulation(data, n_simulations, n_days):
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
                if count == n_days+1:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_volatility))
                price_series.append(price)
                count += 1
            
            simulation_df[x] = price_series
        
        return simulation_df

    # Main Streamlit app
    def main():
        st.header('Stock Price Prediction with Monte Carlo Simulation')
        
        # Input text boxes for user input
        #ticker = st.text_input('Enter Stock Ticker (e.g., AAPL for Apple)', 'AAPL')
        n_simulations = st.number_input('Number of Simulations', min_value=1, max_value=1000, value=100, step=1)
        n_days = st.number_input('Number of Days to Project', min_value=1, max_value=365, value=30, step=1)
        
        # Fetching data
        #end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Plotting last 1 year stock price chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Stock Price'))
        fig1.update_layout(title=f'Last 1 Year Stock Price Chart of {ticker}', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig1)
        st.markdown('---')
        
        # Calculating returns and performing Monte Carlo simulation
        data['Returns'] = data['Adj Close'].pct_change()
        simulation_df = monte_carlo_simulation(data, n_simulations, n_days)
        
        # Plotting Monte Carlo simulation
            # Plotting Monte Carlo simulation
        fig2 = go.Figure()
        for col in simulation_df.columns:
            x_values = list(range(n_days+1))
            fig2.add_trace(go.Scatter(x=x_values, y=simulation_df[col], mode='lines', name=f'Simulation {col+1}'))
        fig2.add_trace(go.Scatter(x=x_values, y=[data['Adj Close'][-1]]*(n_days+1), mode='lines', name='Last Price', line=dict(color='red', dash='dash')))
        fig2.update_layout(title=f'Monte Carlo Simulation for {ticker} Stock', xaxis_title='Day', yaxis_title='Price')
        st.plotly_chart(fig2)


    if __name__ == "__main__":
        main()


#### FB-Prophet section

with Predictions:
    # Function to fetch stock data
    def fetch_stock_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    # Function to plot the actual time series of Adjusted Close prices
    def plot_actual_time_series(df,ticker):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adjusted Closing Price'))
        fig.update_layout(title=f'Actual Time Series of Adjusted Close Prices of {ticker} stock', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)
        st.markdown('---')

    # Function to plot actual vs predicted values
    def plot_actual_vs_predicted(df, forecast,ticker):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted'))
        # fig.update_layout(title=f'Actual vs Predicted Adj. Close Price of {ticker} stock', xaxis_title='Date', yaxis_title='Price')

        fig.update_layout(
        title=f'Actual vs Predicted Adj. Close Price of {ticker} stock',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        )
    )

        st.plotly_chart(fig)
        st.markdown('---')

    ## Function to plot the components of the model
    ## Function to plot the components of the model using Plotly
    ############################################################################################################################
    # def plot_model_components(model, forecast):
    #     components = ['trend', 'yearly', 'weekly', 'daily']
    #     fig = go.Figure()
    #     for component in components:
    #         if component in forecast:
    #             fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[component], mode='lines', name=component.capitalize()))
    #     fig.update_layout(title='Model Components', xaxis_title='Date', yaxis_title='Price')
    #     st.plotly_chart(fig)

#####################################################################################################################################

    def plot_model_components(model, forecast):
        components = ['trend', 'yearly', 'weekly', 'daily']
        for component in components:
            if component in forecast.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[component], mode='lines', name=component.capitalize()))
                
                if f"{component}_lower" in forecast.columns and f"{component}_upper" in forecast.columns:
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[f"{component}_upper"], fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast[f"{component}_lower"], fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False, name="Confidence Interval"))
                fig.update_layout(title=f'{component.capitalize()} Component', xaxis_title='Date', yaxis_title='Value', height=400)
                st.plotly_chart(fig)

############################################################################################################################################

    # Main Streamlit app
    def main():
        st.title('Stock Price Prediction with Prophet')
        
        # Input text box for user input
        ticker = st.text_input('Enter Stock Ticker', 'JPM')
        
        # Fetching data
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.DateOffset(years=1)
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Plot actual time series
        plot_actual_time_series(data,ticker)
        
        # Prepare data for Prophet
        df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        
        # Initialize the Model
        model = Prophet(daily_seasonality=False, yearly_seasonality=True)
        
        # Fit the Model
        model.fit(df_prophet)
        
        # Set the duration of predictions
        period = 30
        
        # Make a DataFrame to hold predictions
        future = model.make_future_dataframe(periods=period)
        
        # Predict
        forecast = model.predict(future)
        
        # Plot actual vs predicted values
        plot_actual_vs_predicted(data, forecast,ticker)

        st.subheader('Plot the components of Prophet')
        
        # Plotting the components of the model
        plot_model_components(model, forecast)

    if __name__ == "__main__":
        main()

#with GoldenCross:

df_gc = data[['Adj Close']]

## Calculate 20-days and 50-days moving average.
df_gc['EMA20'] = df_gc['Adj Close'].ewm(span = 20, adjust = False).mean()
df_gc['EMA50'] = df_gc['Adj Close'].ewm(span = 50, adjust = False).mean()

df_gc['Signal'] = 0.0  
df_gc['Signal'] = np.where(df_gc['EMA20'] > df_gc['EMA50'], 1.0, 0.0)
df_gc['Position'] = df_gc['Signal'].diff()

with GoldenCross:
    plot_actual_time_series(data,ticker)

    trace_adj_close = go.Scatter(x=df_gc.index, y=df_gc['Adj Close'], mode='lines', name='Adj Close', line=dict(color='black', width=1))
    trace_ema20 = go.Scatter(x=df_gc.index, y=df_gc['EMA20'], mode='lines', name='EMA 20-day', line=dict(color='blue', width=1))
    trace_ema50 = go.Scatter(x=df_gc.index, y=df_gc['EMA50'], mode='lines', name='EMA 50-day', line=dict(color='green', width=1))

    # Create traces for buy and sell signals
    trace_buy_signals = go.Scatter(x=df_gc[df_gc['Position'] == 1].index, y=df_gc['EMA20'][df_gc['Position'] == 1], mode='markers', name='buy', marker=dict(color='green', size=15, symbol='triangle-up'))
    trace_sell_signals = go.Scatter(x=df_gc[df_gc['Position'] == -1].index, y=df_gc['EMA50'][df_gc['Position'] == -1], mode='markers', name='sell', marker=dict(color='red', size=15, symbol='triangle-down'))

    # Combine all traces into a list
    data = [trace_adj_close, trace_ema20, trace_ema50, trace_buy_signals, trace_sell_signals]

    # Layout of the plot
    layout = go.Layout(
        title=f'Gloden Cross for {ticker} stocks',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price in USD'),
        plot_bgcolor='grey',
        font=dict(size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=800, height=600
    )

    # Create figure and add data and layout
    fig = go.Figure(data=data, layout=layout)

    # Display the figure in Streamlit
    st.plotly_chart(fig)
    st.markdown('---')



    df_position = df_gc[(df_gc['Position'] == 1) | (df_gc['Position'] == -1)].copy()
    df_position['Position'] = df_position['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')


    # Display the sub-heading and the DataFrame in Streamlit
    st.subheader(f'Trading signals for {ticker} stocks')
    st.dataframe(df_position)
