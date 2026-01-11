import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pycountry

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

st.sidebar.title("Dashboard Controls")

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Global Food Price Dashboard", layout="wide")
st.title("🌍 Global Food Price Analytics Dashboard")

st.markdown("""
### 📌 Key Takeaways

- **Global food prices show long-term upward trends** across staple commodities, reflecting inflationary pressure, supply chain stress, and global demand growth.

- **Price behavior varies significantly by country and commodity**, highlighting the importance of localized market analysis rather than global averages.

- **Forecasting reveals persistent uncertainty**, especially for commodities with historically volatile pricing, underscoring the need for proactive monitoring.

- **Volatility clustering identifies high-risk markets**, where sharp price fluctuations may threaten food affordability and food security.

- **Geographic visualizations are intentionally restricted** when country-level coverage is insufficient, ensuring analytical integrity over misleading visuals.
""")


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
    
    display_summary(filtered)
    
    fig = px.line(filtered, x='year', y='Avg_Price_USD', color='Commodity', markers=True,
                  title="Global Commodity Price Trends", labels={'year':'Year', 'Avg_Price_USD':'Price (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    display_summary(country_filtered)
    
    fig = px.line(country_filtered, x='year', y='Avg_Price_USD', color='Commodity', markers=True,
                  title=f"{country} Commodity Prices Over Time", labels={'year':'Year','Avg_Price_USD':'Price (USD)'})
    st.plotly_chart(fig, use_container_width=True)
    
    latest_year = country_filtered['year'].max()
    top5 = country_filtered[country_filtered['year']==latest_year].sort_values(by='Avg_Price_USD', ascending=False).head(5)
    st.markdown("**Top 5 Most Expensive Commodities:**")
    st.table(top5[['Commodity','Avg_Price_USD']].reset_index(drop=True))
    
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
    
    display_summary(df, value_col='y')
    
    m = Prophet(yearly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=5, freq='Y')
    forecast = m.predict(future)
    
    fig = px.line(forecast, x='ds', y='yhat', title=f"{commodity} Price Forecast (Next 5 Years)")
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dash', color='lightgreen'), name='Upper CI')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(dash='dash', color='lightcoral'), name='Lower CI')
    st.plotly_chart(fig, use_container_width=True)
    
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
    st.subheader("Commodity-Specific Volatility Clustering")

    commodity = st.selectbox(
        "Select Commodity",
        sorted(volatility_data['Commodity'].unique())
    )

    num_clusters = st.slider("Number of Clusters", 2, 5, 3)

    filtered_data = volatility_data[
        volatility_data['Commodity'] == commodity
    ].copy()

    if 'Price_STD' not in filtered_data.columns:
        st.error("Volatility data not available.")
        st.stop()

    # ---- Clustering on volatility only ----
    X = filtered_data[['Price_STD']].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=20
    )

    filtered_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # ---- Visualization ----
    fig = px.scatter(
        filtered_data,
        x='Price_STD',
        y='Country',
        color='Cluster',
        title=f"{commodity} – Country Volatility Clusters",
        labels={'Price_STD': 'Price Volatility (STD)'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Cluster Interpretation ----
    cluster_summary = (
        filtered_data
        .groupby('Cluster')['Price_STD']
        .mean()
        .round(2)
    )

    st.markdown("### 📊 Cluster Interpretation")

    q1 = cluster_summary.quantile(0.33)
    q2 = cluster_summary.quantile(0.66)

    for cluster_id, avg_std in cluster_summary.items():
        if avg_std <= q1:
            label = "🟢 Low Volatility (Stable)"
        elif avg_std <= q2:
            label = "🟡 Medium Volatility"
        else:
            label = "🔴 High Volatility (Risky)"

        st.write(
            f"**Cluster {cluster_id} → {label}**  \n"
            f"- Average Price Volatility (STD): {avg_std}"
        )

    st.markdown("### 🧠 Key Insights")

    st.markdown("""
    - **Low Volatility Countries (🟢 Stable):**  
    These countries show relatively stable food prices over time, indicating stronger supply chains,
    better market regulation, or government price controls.

    - **Medium Volatility Countries (🟡 Medium Risk):**  
    Prices fluctuate moderately, often due to seasonal effects, import dependency,
    or partial exposure to global price shocks.

    - **High Volatility Countries (🔴 Risky):**  
    These countries experience large price swings, which may be driven by economic instability,
    climate impacts, conflict, or heavy reliance on food imports.

    📌 **Why this matters:**  
    Volatility directly affects food affordability and food security. Identifying high-risk countries
    helps policymakers, NGOs, and analysts prioritize intervention and monitoring efforts.
    """)

# ----------------------------
# TAB 5: World Map (Animated + ISO3 + Top 5)
# ----------------------------

elif tab == "World Map":
    st.subheader("🌍 Animated Commodity Price Map")

    COUNTRY_FIXES = {
        "United States of America": "USA",
        "United States": "USA",
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "Viet Nam": "VNM",
        "Iran (Islamic Republic of)": "IRN",
        "Congo (DRC)": "COD",
        "Congo": "COG",
        "Bolivia (Plurinational State of)": "BOL",
        "Tanzania, United Republic of": "TZA",
        "Syrian Arab Republic": "SYR",
        "Lao People's Democratic Republic": "LAO",
        "Korea, Republic of": "KOR",
        "Korea, Democratic People's Republic of": "PRK"
    }

    def country_to_iso3(name):
        if name in COUNTRY_FIXES:
            return COUNTRY_FIXES[name]
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    # --------- NEW: find only commodities with valid ISO map data ----------
    @st.cache_data
    def get_mappable_commodities(df):
        temp = df.copy()
        temp['ISO3'] = temp['Country'].apply(country_to_iso3)
        temp = temp.dropna(subset=['ISO3'])

        valid = (
            temp
            .groupby('Commodity')['ISO3']
            .nunique()
            .reset_index(name='country_count')
            .query('country_count >= 5')
        )

        return valid['Commodity'].tolist()

    valid_commodities = get_mappable_commodities(country_data)

    if not valid_commodities:
        st.warning(
    "🗺️ Map visualization is unavailable for this dataset.\n\n"
    "Reason: The data is market-level with sparse country coverage per commodity. "
    "After geographic standardization, no commodity meets the minimum threshold "
    "for reliable global comparison.\n\n"
    "This restriction is intentional to prevent misleading geographic insights."
)
        st.stop()

    commodity = st.selectbox("Select Commodity", sorted(valid_commodities))

    # --------- Build map data ----------
    map_data = country_data[
        (country_data['Commodity'] == commodity) &
        (country_data['Avg_Price_USD'].notna())
    ].copy()

    map_data['ISO3'] = map_data['Country'].apply(country_to_iso3)
    map_data = map_data.dropna(subset=['ISO3'])

    if map_data.empty:
        st.warning(
            "Map data could not be rendered due to country name standardization issues. "
            "This does not affect underlying price analytics."
        )
        st.stop()
    st.caption(
        "⚠️ Map visualizations are exploratory due to uneven geographic coverage in the dataset."
    )

    # --------- Summary + Map ----------
    display_summary(map_data)

    fig = px.choropleth(
        map_data,
        locations="ISO3",
        color="Avg_Price_USD",
        animation_frame="year",
        hover_name="Country",
        color_continuous_scale="Viridis",
        range_color=(
            map_data['Avg_Price_USD'].min(),
            map_data['Avg_Price_USD'].max()
        ),
        title=f"{commodity} Prices Over Time"
    )

    fig.update_layout(
        geo=dict(projection_type="natural earth"),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------- Top 5 panel ----------
    year = st.slider(
        "Select Year",
        int(map_data['year'].min()),
        int(map_data['year'].max()),
        int(map_data['year'].max())
    )

    top5 = (
        map_data[map_data['year'] == year]
        .sort_values('Avg_Price_USD', ascending=False)
        .head(5)
    )

    st.markdown(f"### 🏆 Top 5 Most Expensive Countries in {year}")
    st.table(top5[['Country', 'Avg_Price_USD']].reset_index(drop=True))

