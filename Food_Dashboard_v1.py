import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans
import numpy as np

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    country_data = pd.read_csv('avg_prices_by_country_year.csv')
    commodity_data = pd.read_csv('commodity_global_trends.csv')
    volatility_data = pd.read_csv('commodity_country_volatility.csv')
    return country_data, commodity_data, volatility_data

country_data, commodity_data, volatility_data = load_data()

# ----------------------------
# Theme Switch
# ----------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

st.sidebar.title("Settings")
theme = st.sidebar.radio("Select Theme", ['Light', 'Dark'])
st.session_state.theme = theme

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Global Food Price Dashboard", layout="wide")
st.title("🌍 Global Food Price Analytics Dashboard")

# ----------------------------
# Sidebar Tabs
# ----------------------------
st.sidebar.header("Dashboard Tabs")
tab = st.sidebar.radio("Select Tab", ["Overview", "Country Explorer", "Commodity Forecast", "Volatility Clustering", "World Map"])

# ----------------------------
# Helper: Display Summary Cards
# ----------------------------
def display_summary(df, value_col='Avg_Price_USD'):
    avg_val = round(df[value_col].mean(),2)
    max_val = round(df[value_col].max(),2)
    min_val = round(df[value_col].min(),2)
    std_val = round(df[value_col].std(),2)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Price (USD)", avg_val)
    col2.metric("Max Price (USD)", max_val)
    col3.metric("Min Price (USD)", min_val)
    col4.metric("Price Volatility (STD)", std_val)

# ----------------------------
# TAB 1: Overview
# ----------------------------
if tab == "Overview":
    st.subheader("Global Commodity Trends")
    selected_commodities = st.multiselect("Select Commodities", options=commodity_data['Commodity'].unique(),
                                         default=['Rice', 'Wheat', 'Maize'])
    filtered = commodity_data[commodity_data['Commodity'].isin(selected_commodities)]
    
    # Summary Cards
    display_summary(filtered)
    
    # Line Chart
    fig = px.line(filtered, x='year', y='Avg_Price_USD', color='Commodity', markers=True,
                  title="Global Commodity Price Trends", labels={'year':'Year', 'Avg_Price_USD':'Price (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Download CSV
    st.download_button(
        label="Download Filtered Data",
        data=filtered.to_csv(index=False).encode('utf-8'),
        file_name="overview_filtered.csv",
        mime='text/csv'
    )

# ----------------------------
# TAB 2: Country Explorer
# ----------------------------
elif tab == "Country Explorer":
    st.subheader("Country Level Analysis")
    country = st.selectbox("Select Country", country_data['Country'].unique())
    country_filtered = country_data[country_data['Country']==country]
    
    # Summary Cards
    display_summary(country_filtered)
    
    # Line Chart
    fig = px.line(country_filtered, x='year', y='Avg_Price_USD', color='Commodity', markers=True,
                  title=f"{country} Commodity Prices Over Time", labels={'year':'Year','Avg_Price_USD':'Price (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 Expensive Commodities
    latest_year = country_filtered['year'].max()
    top5 = country_filtered[country_filtered['year']==latest_year].sort_values(by='Avg_Price_USD', ascending=False).head(5)
    st.markdown("**Top 5 Most Expensive Commodities:**")
    st.table(top5[['Commodity','Avg_Price_USD']].reset_index(drop=True))
    
    # Download CSV
    st.download_button(
        label="Download Country Data",
        data=country_filtered.to_csv(index=False).encode('utf-8'),
        file_name=f"{country}_data.csv",
        mime='text/csv'
    )

# ----------------------------
# TAB 3: Commodity Forecast
# ----------------------------
elif tab == "Commodity Forecast":
    st.subheader("Commodity Price Forecast")
    commodity = st.selectbox("Select Commodity", commodity_data['Commodity'].unique())
    df = commodity_data[commodity_data['Commodity']==commodity][['year','Avg_Price_USD']].rename(columns={'year':'ds','Avg_Price_USD':'y'})
    df['ds'] = pd.to_datetime(df['ds'], format='%Y')
    
    # Summary Cards
    display_summary(df, value_col='y')
    
    # Prophet Forecast
    m = Prophet(yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=5, freq='Y')  # forecast 5 years ahead
    forecast = m.predict(future)
    
    fig = px.line(forecast, x='ds', y='yhat', title=f"{commodity} Price Forecast (Next 5 Years)")
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dash', color='lightgreen'), name='Upper CI')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(dash='dash', color='lightcoral'), name='Lower CI')
    st.plotly_chart(fig, use_container_width=True)
    
    # Download forecast
    st.download_button(
        label="Download Forecast Data",
        data=forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False).encode('utf-8'),
        file_name=f"{commodity}_forecast.csv",
        mime='text/csv'
    )

# ----------------------------
# TAB 4: Volatility Clustering
# ----------------------------
elif tab == "Volatility Clustering":
    st.subheader("Commodity Volatility Clustering")

    selected_commodity = st.selectbox(
        "Select Commodity",
        sorted(volatility_data['Commodity'].unique())
    )

    num_clusters = st.slider("Number of Clusters", 2, 6, 3)

    filtered_df = volatility_data[
        volatility_data['Commodity'] == selected_commodity
    ].copy()

    filtered_df['Price_STD_safe'] = (
        filtered_df['Price_STD']
        .fillna(0)
        .replace(0, 0.1)
    )

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(15, 70))
    filtered_df['Bubble_Size'] = scaler.fit_transform(
        filtered_df[['Price_STD_safe']]
    )

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    filtered_df['Cluster'] = kmeans.fit_predict(
        filtered_df[['Price_STD_safe']]
    )

    display_summary(filtered_df, value_col='Price_STD')

    fig = px.scatter(
        filtered_df,
        x='Country',
        y='Price_STD',
        color='Cluster',
        size='Bubble_Size',
        hover_data=['Price_STD'],
        title=f"Volatility Clusters for {selected_commodity}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Filtered Volatility Data",
        filtered_df.to_csv(index=False).encode("utf-8"),
        f"{selected_commodity}_volatility_clusters.csv",
        "text/csv"
    )

# ----------------------------
# TAB 5: World Map
# ----------------------------
elif tab == "World Map":
    st.subheader("Average Commodity Prices Across Countries")
    commodity = st.selectbox("Select Commodity for Map", commodity_data['Commodity'].unique())
    
    latest_year = country_data['year'].max()
    map_data = country_data[(country_data['Commodity']==commodity) & (country_data['year']==latest_year)]
    
    # Summary Cards
    display_summary(map_data)
    
    fig = px.choropleth(map_data, locations='Country', locationmode='country names',
                        color='Avg_Price_USD', hover_name='Country',
                        color_continuous_scale='Viridis', title=f"{commodity} Average Price in {latest_year}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.download_button(
        label="Download Map Data",
        data=map_data.to_csv(index=False).encode('utf-8'),
        file_name=f"{commodity}_map_data.csv",
        mime='text/csv'
    )


