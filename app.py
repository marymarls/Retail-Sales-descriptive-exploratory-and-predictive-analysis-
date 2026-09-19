import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Retail Sales Intelligence", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("SuperStoreOrders.csv", encoding="latin-1")
    df.columns = df.columns.str.replace('ï»¿', '', regex=False).str.strip()
    df['sales'] = df['sales'].astype(str).str.replace(',', '', regex=False)
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
    df = df.dropna(subset=['sales', 'country'])
    df['order_date'] = pd.to_datetime(df['order_date'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['order_date'])
    return df

df = load_data()

st.sidebar.title("🛍️ Retail Sales Intelligence")
page = st.sidebar.radio("Navigate to:", ["🗺️ Map", "👥 Segments", "📈 Forecast"])

st.title(page)

if page == "🗺️ Map":
    country_year = df.groupby(['country', 'year'], as_index=False).agg(
        total_sales=('sales', 'sum'),
        total_profit=('profit', 'sum'),
        order_count=('order_id', 'count')
    )

    fig = px.choropleth(
        country_year,
        locations="country",
        locationmode="country names",
        color="total_sales",
        color_continuous_scale="OrRd",
        animation_frame="year",
        hover_data={"total_profit": ":.0f", "order_count": True},
        title="Total Sales by Country Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("This map shows, for each country, total sales revenue generated from products purchased by customers located in that country, aggregated by year.")
