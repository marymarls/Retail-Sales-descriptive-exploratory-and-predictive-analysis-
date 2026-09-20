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
page = st.sidebar.radio("Navigate to:", ["ℹ️ About", "🗺️ Map", "👥 Segments", "📈 Forecast"])

st.title(page)


if page == "ℹ️ About":
    st.markdown("""
    ### Retail Sales Intelligence
    A geographic and predictive analysis of global retail performance, built on the **Global Superstore** dataset (2011–2014, ~20,000 orders across 141 countries).

    **What this app does:**
    - 🗺️ **Map** : Visualizes total sales and profit by country, animated across years, to identify strong and weak markets.
    - 👥 **Segments** : Uses K-Means clustering to group customers into 4 behavioral segments based on spending, profit, and discount patterns — surfacing which customers drive real profit vs. which are unprofitable despite high sales.
    - 📈 **Forecast** : Uses Holt-Winters Exponential Smoothing to project sales 6 months ahead, validated against historical hold-out data (7.9% MAPE).

    **Methodology & limitations:**
    - Data is simulated/sample retail data, not from a real company, customer names are reused across countries and shouldn't be treated as unique individual identifiers.
    - The forecast is a demonstration of time-series methodology; production forecasting would benefit from more granular (e.g. regional) models and longer history.

    **Tools used:** Python, Pandas, Plotly, Scikit-learn, Statsmodels, Streamlit

    Built by Mariam Touati as a personal project applying data analytics and machine learning to retail intelligence.
    """)

elif page == "🗺️ Map":
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

    # ============================================================
    # 1. CUSTOMER-LEVEL AGGREGATION
    # ============================================================

    customer_summary = df.groupby(
        ['customer_name', 'country'],
        as_index=False
    ).agg(
        total_sales=('sales', 'sum'),
        total_profit=('profit', 'sum'),
        avg_discount=('discount', 'mean'),
        order_count=('order_id', 'nunique'),
        total_quantity=('quantity', 'sum')
    )

    # ============================================================
    # 2. CUSTOMER PROFIT MARGIN
    # ============================================================

    customer_summary['profit_margin'] = (
        customer_summary['total_profit']
        / customer_summary['total_sales']
        * 100
    )

    # ============================================================
    # 3. K-MEANS CLUSTERING
    # ============================================================

    features = [
        'total_sales',
        'total_profit',
        'avg_discount',
        'order_count',
        'total_quantity'
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        customer_summary[features]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    customer_summary['segment_cluster'] = (
        kmeans.fit_predict(scaled_features)
    )

    # ============================================================
    # 4. DYNAMIC SEGMENT LABELS
    # ============================================================

    cluster_stats = (
        customer_summary
        .groupby('segment_cluster')[['total_profit', 'total_sales']]
        .mean()
        .sort_values('total_profit')
    )

    ordered_labels = [
        "Unprofitable Discount-Driven",
        "Efficient Small-Spenders",
        "Core Regular Customers",
        "High-Value VIP"
    ]

    label_map = dict(
        zip(cluster_stats.index, ordered_labels)
    )

    customer_summary['segment_name'] = (
        customer_summary['segment_cluster']
        .map(label_map)
    )

    # ============================================================
    # 5. PAGE HEADER
    # ============================================================

    st.title("Customer Intelligence")

    st.caption(
        "Customer profiles based on purchasing behavior, "
        "profitability, discount usage, order frequency, "
        "and quantity purchased."
    )

    # ============================================================
    # 6. EXECUTIVE KPIs
    # ============================================================

    total_profiles = len(customer_summary)
    total_sales = customer_summary['total_sales'].sum()
    total_profit = customer_summary['total_profit'].sum()

    overall_margin = (
        total_profit / total_sales * 100
        if total_sales != 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Customer Profiles",
        f"{total_profiles:,}"
    )

    col2.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    col3.metric(
        "Total Profit",
        f"{total_profit:,.0f}"
    )

    col4.metric(
        "Profit Margin",
        f"{overall_margin:.1f}%"
    )

    st.divider()

    # ============================================================
    # 7. SEGMENT SUMMARY TABLE
    # ============================================================

    segment_summary = (
        customer_summary
        .groupby('segment_name')
        .agg(
            customer_profiles=('customer_name', 'count'),
            total_sales=('total_sales', 'sum'),
            total_profit=('total_profit', 'sum'),
            avg_sales=('total_sales', 'mean'),
            avg_profit=('total_profit', 'mean'),
            avg_discount=('avg_discount', 'mean'),
            avg_orders=('order_count', 'mean'),
            avg_quantity=('total_quantity', 'mean')
        )
        .reset_index()
    )

    segment_summary['sales_share'] = (
        segment_summary['total_sales']
        / segment_summary['total_sales'].sum()
        * 100
    )

    # Aggregate margin is more meaningful than averaging
    # individual customer margins.
    segment_summary['profit_margin'] = (
        segment_summary['total_profit']
        / segment_summary['total_sales']
        * 100
    )

    # ============================================================
    # 8. SEGMENT DISTRIBUTION
    # ============================================================

    st.subheader("Customer Profile Distribution")

    segment_counts = (
        customer_summary['segment_name']
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        'segment_name',
        'customer_count'
    ]

    segment_counts['share'] = (
        segment_counts['customer_count']
        / total_profiles
        * 100
    )

    fig_distribution = px.bar(
        segment_counts,
        x='customer_count',
        y='segment_name',
        orientation='h',
        text='share',
        title="Customer Profiles by Segment",
        labels={
            'customer_count': 'Customer Profiles',
            'segment_name': 'Segment'
        }
    )

    fig_distribution.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )

    fig_distribution.update_layout(
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True
    )

    # ============================================================
    # 9. BUSINESS CONTRIBUTION
    # ============================================================

    st.subheader("Business Contribution by Segment")

    segment_chart = segment_summary[
        ['segment_name', 'total_sales', 'total_profit']
    ].melt(
        id_vars='segment_name',
        var_name='metric',
        value_name='value'
    )

    fig_contribution = px.bar(
        segment_chart,
        x='segment_name',
        y='value',
        color='metric',
        barmode='group',
        title="Sales and Profit Contribution"
    )

    st.plotly_chart(
        fig_contribution,
        use_container_width=True
    )

    # ============================================================
    # 10. AUTOMATIC BUSINESS INSIGHTS
    # ============================================================

    st.subheader("Key Business Insights")

    vip = segment_summary[
        segment_summary['segment_name'] == 'High-Value VIP'
    ].iloc[0]

    unprofitable = segment_summary[
        segment_summary['segment_name']
        == 'Unprofitable Discount-Driven'
    ].iloc[0]

    small_spenders = segment_summary[
        segment_summary['segment_name']
        == 'Efficient Small-Spenders'
    ].iloc[0]

    core = segment_summary[
        segment_summary['segment_name']
        == 'Core Regular Customers'
    ].iloc[0]

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.info(
            f"**High-Value VIP** profiles represent "
            f"**{vip['customer_profiles'] / total_profiles * 100:.1f}%** "
            f"of customer profiles and generate "
            f"**{vip['sales_share']:.1f}%** of total sales."
        )

        st.warning(
            f"**Unprofitable Discount-Driven** profiles have an "
            f"average discount of **{unprofitable['avg_discount'] * 100:.1f}%** "
            f"and an average profit of "
            f"**{unprofitable['avg_profit']:,.0f}**."
        )

    with insight_col2:

        st.success(
            f"**Efficient Small-Spenders** represent "
            f"**{small_spenders['customer_profiles'] / total_profiles * 100:.1f}%** "
            f"of profiles while maintaining positive average profit."
        )

        st.info(
            f"**Core Regular Customers** generate an average of "
            f"**{core['avg_orders']:.1f} orders** per profile "
            f"with average sales of "
            f"**{core['avg_sales']:,.0f}**."
        )

    # ============================================================
    # 11. SEGMENT EXPLORER
    # ============================================================

    st.divider()

    st.subheader("Explore a Customer Segment")

    selected_segment = st.selectbox(
        "Select a segment",
        ["All Segments"]
        + sorted(customer_summary['segment_name'].unique())
    )

    if selected_segment == "All Segments":

        filtered_customers = customer_summary

    else:

        filtered_customers = customer_summary[
            customer_summary['segment_name'] == selected_segment
        ]

    # ============================================================
    # 12. SELECTED SEGMENT KPIs
    # ============================================================

    selected_profiles = len(filtered_customers)

    selected_sales = filtered_customers['total_sales'].sum()
    selected_profit = filtered_customers['total_profit'].sum()

    selected_avg_sales = (
        filtered_customers['total_sales'].mean()
    )

    selected_avg_profit = (
        filtered_customers['total_profit'].mean()
    )

    selected_avg_discount = (
        filtered_customers['avg_discount'].mean()
    )

    selected_avg_orders = (
        filtered_customers['order_count'].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Profiles",
        f"{selected_profiles:,}"
    )

    col2.metric(
        "Average Sales",
        f"{selected_avg_sales:,.0f}"
    )

    col3.metric(
        "Average Profit",
        f"{selected_avg_profit:,.0f}"
    )

    col4.metric(
        "Average Discount",
        f"{selected_avg_discount * 100:.1f}%"
    )

    # ============================================================
    # 13. SALES VS PROFIT DRILL-DOWN
    # ============================================================

    st.subheader("Customer-Level Sales vs Profit")

    fig = px.scatter(
        filtered_customers,
        x='total_sales',
        y='total_profit',
        color='segment_name',
        size='order_count',
        hover_data=[
            'customer_name',
            'country',
            'total_sales',
            'total_profit',
            'profit_margin',
            'avg_discount',
            'order_count',
            'total_quantity',
            'segment_name'
        ],
        title="Customer Segments: Sales vs Profit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ============================================================
    # 14. DISCOUNT VS PROFITABILITY
    # ============================================================

    st.subheader("Discount vs Profitability")

    fig_discount = px.scatter(
        segment_summary,
        x='avg_discount',
        y='avg_profit',
        size='customer_profiles',
        color='segment_name',
        text='segment_name',
        title="Average Discount vs Average Profit by Segment",
        labels={
            'avg_discount': 'Average Discount',
            'avg_profit': 'Average Profit'
        }
    )

    fig_discount.update_xaxes(
        tickformat='.0%'
    )

    fig_discount.update_traces(
        textposition='top center'
    )

    st.plotly_chart(
        fig_discount,
        use_container_width=True
    )

    # ============================================================
    # 15. SEGMENT PROFILE
    # ============================================================

    st.subheader("Segment Profile")

    profile_columns = [
        'segment_name',
        'customer_profiles',
        'total_sales',
        'total_profit',
        'profit_margin',
        'avg_sales',
        'avg_profit',
        'avg_discount',
        'avg_orders',
        'avg_quantity'
    ]

    st.dataframe(
        segment_summary[profile_columns].round(2),
        use_container_width=True
    )

    # ============================================================
    # 16. METHODOLOGY
    # ============================================================

    with st.expander("ℹ️ Segmentation methodology"):

        st.write(
            """
            Customers are represented by the combination of
            customer name and country because customer names may
            be reused across countries in the source dataset.

            Customer profiles are segmented using K-Means clustering
            with four clusters.

            The clustering variables are:

            • Total sales
            • Total profit
            • Average discount
            • Order frequency
            • Total quantity purchased

            Before clustering, the variables are standardized using
            StandardScaler so that variables with different scales
            do not disproportionately influence the clustering.

            The four clusters are interpreted as behavioral customer
            profiles rather than verified unique individuals.
            """
        )



elif page == "📈 Forecast":
    monthly_sales = df.groupby(df['order_date'].dt.to_period('M'))['sales'].sum().reset_index()
    monthly_sales['order_date'] = monthly_sales['order_date'].dt.to_timestamp()

    train = monthly_sales.iloc[:-6]
    test = monthly_sales.iloc[-6:]

    model = ExponentialSmoothing(train['sales'], trend='add', seasonal='add', seasonal_periods=12)
    results = model.fit()
    test_forecast = results.forecast(6)

    mae = (test['sales'].values - test_forecast.values)
    mape = (abs(mae) / test['sales'].values).mean() * 100

    full_model = ExponentialSmoothing(monthly_sales['sales'], trend='add', seasonal='add', seasonal_periods=12)
    full_results = full_model.fit()
    future_forecast = full_results.forecast(6)

    future_dates = pd.date_range(start=monthly_sales['order_date'].iloc[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_sales['order_date'], y=monthly_sales['sales'], mode='lines', name='Historical Sales'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_forecast.values, mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.update_layout(title='Global Sales: Historical + Forecast')
    st.plotly_chart(fig, use_container_width=True)

    st.info(f" Model validation: tested on the last 6 known months, this model's average error (MAPE) was **{mape:.1f}%**. Forecast shown above is refit on all available data ({len(monthly_sales)} months, 2011–2014).")
