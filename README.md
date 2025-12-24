# 🌍 Global Food Price Analytics Dashboard

An interactive data analytics dashboard for exploring global food price trends, volatility patterns, and price forecasting using Python and Streamlit.

This project analyzes historical food price data to surface insights related to price stability, market risk, and food security, while explicitly handling real-world data limitations.

## 📌 Project Overview

Global food prices are influenced by multiple factors such as supply chains, climate, geopolitics, and economic stability. This dashboard enables users to:

- Analyze long-term commodity price trends
- Explore country-level price behavior
- Identify high-risk regions using volatility clustering
- Forecast future commodity prices using time-series models
- Understand where data limitations prevent reliable geographic insights

---

## 🧩 Project Structure & Iterations

This project was developed in two iterations, both preserved intentionally.

### 🔹 Version 1 (V1) – Exploratory Prototype

**File**: `Food_Dashboard_v1.py`

V1 represents the initial exploratory phase, where the primary goal was to:

- Understand the dataset
- Experiment with multiple visualizations
- Test feature ideas quickly

**Characteristics of V1:**

- Rapid experimentation
- Some incomplete features and edge-case bugs
- Minimal data validation
- Served as a learning and discovery phase

V1 is intentionally retained to demonstrate the evolution of analytical thinking, not as a polished product.

### 🔹 Version 2 (V2) – Refined Analytics Dashboard

**File**: `Food_Dashboard_v2.py`

V2 is the refined and portfolio-ready version, built by addressing limitations discovered in V1.

**Key improvements in V2:**

- Cleaned and aggregated datasets for performance
- Clear separation between analysis and visualization
- Robust handling of missing and sparse data
- Explicit prevention of misleading visualizations
- Stronger analytical storytelling through insights

V2 reflects intentional design decisions rather than feature accumulation.

---

## 📊 Dashboard Features

### 1️⃣ Global Overview

- Long-term price trends for selected commodities
- Summary statistics (average, min, max, volatility)
- Downloadable filtered datasets

### 2️⃣ Country Explorer

- Country-wise commodity price analysis
- Time-series visualization by commodity
- Identification of the most expensive commodities per country

### 3️⃣ Commodity Price Forecasting

- Time-series forecasting using Facebook Prophet
- 5-year forward price projections
- Confidence intervals to reflect uncertainty

⚠️ **Note:** Forecasts are indicative, not predictive guarantees.

### 4️⃣ Volatility Clustering (Analytical Core)

Countries are clustered based on:

- Average commodity price
- Price volatility (standard deviation)

**Cluster Interpretation:**

- 🟢 **Stable**: Relatively consistent prices, often indicating stronger supply chains or regulation.
- 🟡 **Medium Risk**: Moderate fluctuations driven by seasonal effects or partial exposure to global shocks.
- 🔴 **Risky**: High volatility, potentially linked to economic instability, climate stress, or import dependency.

⚠️ **Why this matters:**  
Price volatility directly impacts food affordability and food security. Identifying high-risk regions helps prioritize monitoring and intervention.

### 5️⃣ World Map (Intentionally Restricted)

A global price map was evaluated but intentionally limited due to data constraints.

**Why the map is restricted:**

- The dataset is market-level, not consistently country-level.
- Many commodities lack sufficient geographic coverage.
- Enabling the map would risk misleading interpretations.

This restriction reflects a **data ethics decision**, not a technical limitation.

---

## 🗂️ Data Source

- **Dataset**: Global Food Prices Dataset
- **Source**: [Kaggle](https://www.kaggle.com/)
- **Granularity**: Market-level price observations aggregated into country- and commodity-level summaries
- Due to dataset size (~400MB), pre-aggregated CSV files are used for dashboard performance.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – dashboard framework
- **Pandas & NumPy** – data processing
- **Plotly** – interactive visualizations
- **Scikit-learn** – clustering (KMeans)
- **Prophet** – time-series forecasting

---

## 🎯 Key Takeaways

- Not all datasets support all visualizations — knowing when not to visualize is a strength.
- Volatility analysis provides deeper insight than averages alone.
- Iterative development leads to better analytical decisions.
- Transparency about data limitations builds trust.

---

## 🚀 Future Improvements (Optional)

- External macroeconomic data integration
- Region-level aggregation
- Improved geographic standardization
- Model comparison for forecasting

---

## 👤 Author Notes

This project emphasizes analytical thinking, data integrity, and clarity over complexity.  
V1 and V2 together demonstrate both learning progression and decision-making maturity.
