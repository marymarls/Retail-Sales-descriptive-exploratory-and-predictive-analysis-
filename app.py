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
    country_year['year'] = country_year['year'].astype(int)
    country_year = country_year.sort_values('year')

    fig = px.choropleth(
        country_year,
        locations="country",
        locationmode="country names",
        color="total_sales",
        color_continuous_scale="OrRd",
        animation_frame="year",
        category_orders={"year": sorted(country_year['year'].unique())},
        hover_data={"total_profit": ":.0f", "order_count": True},
        title="Total Sales by Country Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("This map shows, for each country, total sales revenue generated from products purchased by customers located in that country, aggregated by year.")


elif page == "👥 Segments":
    customer_summary = df.groupby(['customer_name', 'country'], as_index=False).agg(
        total_sales=('sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_discount=('discount', 'mean'),
        order_count=('order_id', 'nunique'),
        total_quantity=('quantity', 'sum')
    )

    features = ['total_sales', 'total_profit', 'avg_discount', 'order_count', 'total_quantity']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_summary[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_summary['segment_cluster'] = kmeans.fit_predict(scaled_features)

    # Dynamically rank clusters by average profit, then average sales, to assign consistent labels
    cluster_stats = customer_summary.groupby('segment_cluster')[['total_profit', 'total_sales']].mean()
    cluster_stats = cluster_stats.sort_values('total_profit')

    ordered_labels = ["Unprofitable Discount-Driven", "Efficient Small-Spenders", "Core Regular Customers", "High-Value VIP"]
    label_map = dict(zip(cluster_stats.index, ordered_labels))
    customer_summary['segment_name'] = customer_summary['segment_cluster'].map(label_map)

    fig = px.scatter(
        customer_summary,
        x="total_sales",
        y="total_profit",
        color="segment_name",
        size="order_count",
        hover_data=["customer_name", "country", "avg_discount"],
        title="Customer Segments: Sales vs Profit"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Customers are grouped into 4 behavioral segments based on sales, profit, discount usage, order frequency, and quantity purchased.")

    st.subheader("Segment Averages")
    st.dataframe(customer_summary.groupby('segment_name')[features].mean().round(2))
