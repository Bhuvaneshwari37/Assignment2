import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sqlalchemy import create_engine

# Connect to the SQLite database
engine = create_engine('mysql+pymysql://root:Sqlbr*123@localhost/StockMarket')

# Load data from the SQL database

stocks_df = pd.read_sql('SELECT * FROM stocks', engine)

# Calculate necessary metrics
stocks_df['Daily_Return'] = stocks_df.groupby('ticker')['close'].pct_change()
stocks_df['Yearly_Return'] = stocks_df.groupby('ticker')['close'].transform(lambda x: x.iloc[-1] / x.iloc[0] - 1)
stocks_df['Volatility'] = stocks_df.groupby('ticker')['Daily_Return'].transform(np.std)

# Create Key Metrics DataFrames
top_10_green = stocks_df.groupby('ticker').last().sort_values(by='Yearly_Return', ascending=False).head(10)
top_10_loss = stocks_df.groupby('ticker').last().sort_values(by='Yearly_Return').head(10)
num_green_stocks = (stocks_df['Yearly_Return'] > 0).sum()
num_red_stocks = (stocks_df['Yearly_Return'] < 0).sum()
avg_price = stocks_df['close'].mean()
avg_volume = stocks_df['volume'].mean()

# Streamlit app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Go to', [
        'Key Metrics', 
        'Volatility Analysis', 
        'Yearly Cumulative Return', 
        'Stock Price Correlation', 
        'Top 5 Gainers and Losers by Month'])

    if page == 'Key Metrics':
        key_metrics_page()
    elif page == 'Volatility Analysis':
        volatility_analysis_page()
    elif page == 'Yearly Cumulative Return':
        yearly_cumulative_return_page()
    elif page == 'Stock Price Correlation':
        stock_price_correlation_page()
    elif page == 'Top 5 Gainers and Losers by Month':
        top_gainers_losers_page()

def key_metrics_page():
    st.title('Key Metrics')
    st.subheader('Top 10 Green Stocks')
    st.dataframe(top_10_green[['Yearly_Return']])
    st.subheader('Top 10 Loss Stocks')
    st.dataframe(top_10_loss[['Yearly_Return']])
    st.subheader('Market Summary')
    st.write(f"Number of Green Stocks: {num_green_stocks}")
    st.write(f"Number of Red Stocks: {num_red_stocks}")
    st.write(f"Average Price: {avg_price}")
    st.write(f"Average Volume: {avg_volume}")

def volatility_analysis_page():
    st.title('Volatility Analysis')
    top_10_volatile = stocks_df.groupby('ticker').last().sort_values(by='Volatility', ascending=False).head(10)
    st.bar_chart(top_10_volatile[['Volatility']])

def yearly_cumulative_return_page():
    st.title('Yearly Cumulative Return for Top 5 Performing Stocks')
    stocks_df['Cumulative_Return'] = stocks_df.groupby('ticker')['Daily_Return'].transform(lambda x: (x + 1).cumprod() - 1)
    top_5_performers = stocks_df.groupby('ticker').last().sort_values(by='Cumulative_Return', ascending=False).head(5).index
    top_5_data = stocks_df[stocks_df['ticker'].isin(top_5_performers)]

    st.bar_chart(top_5_data.groupby('ticker')['Cumulative_Return'].last())

def stock_price_correlation_page():
    st.title('Stock Price Correlation Heatmap')
    correlation_matrix = stocks_df.pivot_table(index='date', columns='ticker', values='close').corr()
    fig, ax = plt.subplots(figsize=(60, 30))
    sns.heatmap(correlation_matrix, ax=ax, cmap='coolwarm', annot=True)
    st.pyplot(fig)

def top_gainers_losers_page():
    st.title('Top 5 Gainers and Losers by Month')
    stocks_df['Month'] = pd.to_datetime(stocks_df['date']).dt.to_period('M')
    monthly_returns = stocks_df.groupby(['Month', 'ticker'])['close'].last().pct_change().dropna()
    
    for month in monthly_returns.index.levels[0]:
        monthly_data = monthly_returns.loc[month].sort_values(ascending=False)
        top_gainers = monthly_data.head(5)
        top_losers = monthly_data.tail(5)
        
        st.subheader(f'Top 5 Gainers in {month}')
        st.bar_chart(top_gainers)
        st.subheader(f'Top 5 Losers in {month}')
        st.bar_chart(top_losers)

if __name__ == '__main__':
    main()
