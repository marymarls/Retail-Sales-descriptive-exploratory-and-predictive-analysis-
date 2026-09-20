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

st.sidebar.title("Retail Sales Analytics")
page = st.sidebar.radio("Navigate to:", ["About", "Map", "Segments", "Sales Analysis"])

st.title(page)


if page == "About":

    st.title("Retail Sales Analytics")

    st.markdown("""
   

    **Retail Sales Analytics** is an interactive business intelligence
    application designed to explore historical retail performance,
    customer behavior, and sales patterns using the **Global Superstore**
    dataset.

    The application combines **data analysis, visualization, and
    unsupervised machine learning** to transform transactional data into
    business-oriented insights.
    """)

    st.divider()

    
    # project overview
    

    st.subheader("What this application does")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### 1. Geographic Analysis

        Explore sales and profit performance across countries and
        observe how market performance changes over time.

        The interactive map helps identify:

        - High-performing markets
        - Lower-performing markets
        - Geographic differences in profitability
        - Changes in performance across years
        """)

    with col2:

        st.markdown("""
        ### 2. Customer Segmentation

        Uses **K-Means clustering** to identify four behavioral customer
        profiles based on:

        - Total sales
        - Total profit
        - Average discount
        - Order frequency
        - Quantity purchased
        """)

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("""
        ### 3. Sales Analysis

        Examines historical sales patterns through:

        - Monthly sales trends
        - Yearly performance
        - Year-over-year growth
        - Recurring monthly patterns
        - Highest and lowest observed periods

        This page focuses on **descriptive analysis**.
        """)

    with col4:

        st.markdown("""
        ### 4. Business Intelligence

        The application transforms analytical results into business
        insights through:

        - KPI indicators
        - Segment comparisons
        - Sales and profit contribution
        - Interactive customer exploration
        - Automatically generated observations
        """)

    st.divider()

    
    # where did i get my dataset?
    

    st.subheader("Where did i get my dataset?")

    st.markdown("""
    The application uses the **Global Superstore** dataset, covering
    approximately **2011–2014** and containing around **20,000 orders
    across 141 countries**.
    """)

    st.info(
        "The dataset is a public/sample retail dataset and DOES NOT"
        "represent the operations of a REAL company."
    )

   
    # methodology i followed
    

    st.subheader("Methodology I followed")

    st.markdown("""
    ### Customer Segmentation

    Customer profiles are created by grouping transactions using the
    combination of **customer name and country**.

    The resulting behavioral profiles are standardized using
    `StandardScaler` and segmented using **K-Means clustering with
    4 clusters**.

    The four resulting profiles are interpreted according to their
    observed sales, profitability, discount behavior, order frequency,
    and purchase volume.

    ### Sales Analysis

    Sales are aggregated at the monthly and yearly levels to examine
    historical trends and recurring patterns.

    No future sales values are predicted in the current version of the
    application.
    """)

    
    # LIMITATIONS
   

    st.subheader("Data Limitations")

    st.warning("""
    **Important interpretation notes:**

    - The dataset is sample/public retail data and is not connected to
      a real company's operational systems.
    - Customer names may be reused across countries. Therefore,
      customer-level analysis uses the combination of **customer name
      + country** as a behavioral profile identifier.
    - Customer segments should therefore be interpreted as
      **customer-country behavioral profiles**, not verified individual
      customers.
    - Historical patterns shown in the application describe the
      available dataset and should not automatically be interpreted as
      predictions of future performance.
    """)

   
    # TECHNOLOGIES
    

    st.subheader("Technologies")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        **Data & Analysis**

        - Python
        - Pandas
        - NumPy
        """)

    with tech_col2:
        st.markdown("""
        **Machine Learning**

        - Scikit-learn
        - StandardScaler
        - K-Means clustering
        """)

    with tech_col3:
        st.markdown("""
        **Visualization & App**

        - Plotly
        - Streamlit
        """)

    st.divider()

    
    # About me
    

    st.subheader("Project")

    st.markdown("""
    Built by **Mariam Touati** as a personal project exploring the
    intersection of **Business Intelligence, data analytics,
    visualization, and machine learning**.
    """)


elif page == "Map":
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


elif page == "Segments":

    # ------------------------------------------------------------
    # 1. CUSTOMER-LEVEL AGGREGATION
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # 2. CUSTOMER PROFIT MARGIN
    # ------------------------------------------------------------

    customer_summary['profit_margin'] = (
        customer_summary['total_profit']
        / customer_summary['total_sales']
        * 100
    )


    # ------------------------------------------------------------
    # 3. K-MEANS CLUSTERING
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # 4. DYNAMIC SEGMENT LABELS
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # 5. PAGE HEADER
    # ------------------------------------------------------------

    st.title("Customer Intelligence")

    st.caption(
        "Customer profiles based on purchasing behavior, "
        "profitability, discount usage, order frequency, "
        "and quantity purchased."
    )


    # ------------------------------------------------------------
    # 6. EXECUTIVE KPIs
    # ------------------------------------------------------------

    total_profiles = len(customer_summary)

    total_sales = customer_summary['total_sales'].sum()

    total_profit = customer_summary['total_profit'].sum()

    overall_margin = (
        total_profit / total_sales * 100
        if total_sales != 0
        else 0
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


    # ------------------------------------------------------------
    # 7. SEGMENT SUMMARY TABLE
    # ------------------------------------------------------------

    segment_summary = (
        customer_summary
        .groupby('segment_name')
        .agg(
            customer_profiles=('customer_name', 'count'),
            total_sales=('total_sales', 'sum'),
            total_profit=('total_profit', 'sum'),
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

    segment_summary['profit_share'] = (
        segment_summary['total_profit']
        / segment_summary['total_profit'].sum()
        * 100
    )

    # Aggregate margin is more meaningful than averaging
    # individual customer margins.

    segment_summary['profit_margin'] = (
        segment_summary['total_profit']
        / segment_summary['total_sales']
        * 100
    )


    # ------------------------------------------------------------
    # 8. SEGMENT DISTRIBUTION
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # 9. BUSINESS CONTRIBUTION
    # ------------------------------------------------------------

    st.subheader("Segment Economic Contribution")

    st.caption(
        "How much of total sales and profit is generated by each customer segment?"
    )

    contribution_data = segment_summary[
        ["segment_name", "sales_share", "profit_share"]
    ].melt(
        id_vars="segment_name",
        var_name="metric",
        value_name="share"
    )

    contribution_data["metric"] = contribution_data["metric"].map({
        "sales_share": "Sales Share",
        "profit_share": "Profit Share"
    })

    fig = px.bar(
        contribution_data,
        x="segment_name",
        y="share",
        color="metric",
        barmode="group",
        labels={
            "segment_name": "Customer Segment",
            "share": "Share (%)",
            "metric": ""
        },
        title="Sales Share vs Profit Share by Segment"
    )

    fig.update_yaxes(ticksuffix="%")

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ------------------------------------------------------------
    # 10. AUTOMATIC BUSINESS INSIGHTS
    # ------------------------------------------------------------

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
            f"and a profit margin of "
            f"**{unprofitable['profit_margin']:.1f}%**."
        )

    with insight_col2:

        st.success(
            f"**Efficient Small-Spenders** represent "
            f"**{small_spenders['customer_profiles'] / total_profiles * 100:.1f}%** "
            f"of profiles while maintaining a profit margin of "
            f"**{small_spenders['profit_margin']:.1f}%**."
        )

        st.info(
            f"**Core Regular Customers** generate an average of "
            f"**{core['avg_orders']:.1f} orders** per profile "
            f"and account for **{core['sales_share']:.1f}%** "
            f"of total sales."
        )


    # ------------------------------------------------------------
    # 11. SEGMENT EXPLORER
    # ------------------------------------------------------------

    st.divider()

    st.subheader("Explore a Customer Segment")

    selected_segment = st.selectbox(
        "Select a segment",
        ["All Segments"]
        + sorted(customer_summary["segment_name"].unique())
    )

    if selected_segment == "All Segments":

        filtered_customers = customer_summary

    else:

        filtered_customers = customer_summary[
            customer_summary["segment_name"] == selected_segment
        ]


    # ------------------------------------------------------------
    # 12. SELECTED SEGMENT KPIs
    # ------------------------------------------------------------

    if selected_segment == "All Segments":

        selected_profiles = len(filtered_customers)

        selected_sales_share = 100.0

        selected_total_sales = filtered_customers["total_sales"].sum()

        selected_total_profit = filtered_customers["total_profit"].sum()

        selected_profit_margin = (
            selected_total_profit
            / selected_total_sales
            * 100
            if selected_total_sales != 0
            else 0
        )

        selected_avg_orders = (
            filtered_customers["order_count"].mean()
        )

        selected_avg_discount = (
            filtered_customers["avg_discount"].mean()
        )

    else:

        selected = segment_summary[
            segment_summary["segment_name"] == selected_segment
        ].iloc[0]

        selected_profiles = int(
            selected["customer_profiles"]
        )

        selected_sales_share = (
            selected["sales_share"]
        )

        selected_profit_margin = (
            selected["profit_margin"]
        )

        selected_avg_orders = (
            selected["avg_orders"]
        )

        selected_avg_discount = (
            selected["avg_discount"]
        )


    # KPI CARDS

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Profiles",
        f"{selected_profiles:,}"
    )

    col2.metric(
        "Sales Share",
        f"{selected_sales_share:.1f}%"
    )

    col3.metric(
        "Profit Margin",
        f"{selected_profit_margin:.1f}%"
    )

    col4.metric(
        "Average Orders",
        f"{selected_avg_orders:.1f}"
    )

    col5.metric(
        "Average Discount",
        f"{selected_avg_discount * 100:.1f}%"
    )


    # KPI EXPLANATIONS

    # KPI EXPLANATIONS

st.markdown(
    """
    <div style="color: #888; font-size: 0.82rem; margin-top: -10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; text-align: center;">
            <div style="width: 20%;">→ How large is the segment?</div>
            <div style="width: 20%;">→ How important is it commercially?</div>
            <div style="width: 20%;">→ How profitable is the segment?</div>
            <div style="width: 20%;">→ How engaged/frequent are they?</div>
            <div style="width: 20%;">→ How much discounting is associated with them?</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


    # ------------------------------------------------------------
    # 13. SALES VS PROFIT DRILL-DOWN
    # ------------------------------------------------------------

    st.subheader("Customer-Level Sales vs Profit")

    fig = px.scatter(
        filtered_customers,
        x="total_sales",
        y="total_profit",
        color="segment_name",
        size="order_count",
        hover_data=[
            "customer_name",
            "country",
            "avg_discount",
        ],
        title="Customer Profiles: Sales vs Profit"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # HOW TO INTERPRET THE SCATTERPLOT

    st.markdown(
        """
        <div style="
            color: #888;
            font-size: 0.85rem;
            line-height: 1.6;
            margin-top: -5px;
        ">

            <b style="color: #666;">High sales + high profit</b><br>
            Potentially valuable profiles.
            <br><br>

            <b style="color: #666;">High sales + low/negative profit</b><br>
            Potentially problematic profiles.
            <br><br>

            <b style="color: #666;">Low sales + positive profit</b><br>
            Small but efficient profiles.
            <br><br>

            <b style="color: #666;">Low sales + negative profit</b><br>
            Low-value and potentially unprofitable profiles.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------
    # 14. DISCOUNT VS PROFITABILITY
    # ------------------------------------------------------------

    st.subheader("Discount Behavior vs Profitability")

    st.markdown(
        """
        <div style="
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 10px;
        ">

            <b style="color: #666;">Business question:</b>
            Does higher discounting appear to be associated with lower
            profitability across customer segments?

        </div>
        """,
        unsafe_allow_html=True
    )

    discount_profit = (
        customer_summary
        .groupby("segment_name")
        .agg(
            avg_discount=("avg_discount", "mean"),
            avg_profit=("total_profit", "mean")
        )
        .reset_index()
    )

    fig = px.scatter(
        discount_profit,
        x="avg_discount",
        y="avg_profit",
        color="segment_name",
        hover_data=["segment_name"],
        labels={
            "avg_discount": "Average Discount",
            "avg_profit": "Average Profit",
            "segment_name": "Customer Segment"
        },
        title="Discount Behavior vs Average Profit"
    )

    fig.update_xaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ------------------------------------------------------------
    # 15. SEGMENT PROFILE
    # ------------------------------------------------------------

    st.subheader("Segment Profile")

    profile_columns = [
        'segment_name',
        'customer_profiles',
        'total_sales',
        'total_profit',
        'sales_share',
        'profit_share',
        'profit_margin',
        'avg_discount',
        'avg_orders',
        'avg_quantity'
    ]

    st.dataframe(
        segment_summary[profile_columns].round(2),
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------------
    # 16. METHODOLOGY
    # ------------------------------------------------------------

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


elif page == "Sales Analysis":

    
# 1. MONTHLY SALES AGGREGATION


    monthly_sales = (
        df.groupby(df['order_date'].dt.to_period('M'))['sales']
        .sum()
        .reset_index()
    )

    monthly_sales['order_date'] = (
        monthly_sales['order_date'].dt.to_timestamp()
    )

    
# 2. BASIC SALES METRICS

    
    total_sales = monthly_sales['sales'].sum()
    average_monthly_sales = monthly_sales['sales'].mean()

    best_month = monthly_sales.loc[
        monthly_sales['sales'].idxmax()
    ]

    worst_month = monthly_sales.loc[
        monthly_sales['sales'].idxmin()
    ]

    
# 3. PAGE HEADER
    

    st.title("Sales Trends & Analysis")

    st.caption(
        "Descriptive analysis of historical sales patterns, "
        "monthly performance, seasonality, and growth."
    )


# 4. KPI CARDS
    

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"{total_sales:,.0f}"
    )

    col2.metric(
        "Average Monthly Sales",
        f"{average_monthly_sales:,.0f}"
    )

    col3.metric(
        "Best Month",
        best_month['order_date'].strftime("%b %Y")
    )

    col4.metric(
        "Lowest Month",
        worst_month['order_date'].strftime("%b %Y")
    )

    st.divider()

    
# 5. MONTHLY SALES TREND
    

    st.subheader("Monthly Sales Trend")

    fig = px.line(
        monthly_sales,
        x='order_date',
        y='sales',
        markers=True,
        title="Monthly Sales Performance",
        labels={
            'order_date': 'Month',
            'sales': 'Sales'
        }
    )

    fig.update_layout(
        hovermode='x unified'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

   
# 6. YEARLY PERFORMANCE
    

    yearly_sales = (
        df.groupby(df['order_date'].dt.year)['sales']
        .sum()
        .reset_index()
    )

    yearly_sales.columns = ['year', 'sales']

    yearly_sales['growth'] = (
        yearly_sales['sales'].pct_change() * 100
    )

    st.subheader("Yearly Sales Performance")

    fig_year = px.bar(
        yearly_sales,
        x='year',
        y='sales',
        text='sales',
        title="Total Sales by Year",
        labels={
            'year': 'Year',
            'sales': 'Sales'
        }
    )

    fig_year.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside'
    )

    st.plotly_chart(
        fig_year,
        use_container_width=True
    )

    
# 7. MONTHLY SEASONAL PATTERN
    

    monthly_sales['month'] = (
        monthly_sales['order_date'].dt.month
    )

    monthly_sales['month_name'] = (
        monthly_sales['order_date'].dt.strftime('%b')
    )

    seasonal_pattern = (
        monthly_sales
        .groupby(
            ['month', 'month_name'],
            as_index=False
        )['sales']
        .mean()
        .sort_values('month')
    )

    st.subheader("Average Sales by Month")

    fig_season = px.bar(
        seasonal_pattern,
        x='month_name',
        y='sales',
        title="Average Monthly Sales Pattern",
        labels={
            'month_name': 'Month',
            'sales': 'Average Sales'
        }
    )

    st.plotly_chart(
        fig_season,
        use_container_width=True
    )

    
# 8. AUTOMATIC BUSINESS INSIGHTS
   

    highest_season_month = seasonal_pattern.loc[
        seasonal_pattern['sales'].idxmax()
    ]

    lowest_season_month = seasonal_pattern.loc[
        seasonal_pattern['sales'].idxmin()
    ]

    st.subheader("Key Sales Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.info(
            f"**Highest observed month:** "
            f"{best_month['order_date'].strftime('%B %Y')} "
            f"with sales of "
            f"**{best_month['sales']:,.0f}**."
        )

        st.success(
            f"**Strongest recurring month:** "
            f"{highest_season_month['month_name']} "
            f"has the highest average sales across the "
            f"available years."
        )

    with insight_col2:

        st.warning(
            f"**Lowest observed month:** "
            f"{worst_month['order_date'].strftime('%B %Y')} "
            f"with sales of "
            f"**{worst_month['sales']:,.0f}**."
        )

        st.info(
            f"**Lowest recurring month:** "
            f"{lowest_season_month['month_name']} "
            f"has the lowest average sales across the "
            f"available years."
        )

    
# 9. YEAR-OVER-YEAR GROWTH TABLE
    

    st.subheader("Year-over-Year Performance")

    yearly_display = yearly_sales.copy()

    yearly_display['sales'] = (
        yearly_display['sales'].round(0)
    )

    yearly_display['growth'] = (
        yearly_display['growth'].round(1)
    )

    yearly_display['growth'] = (
        yearly_display['growth'].apply(
            lambda x: f"{x:.1f}%"
            if pd.notna(x)
            else "—"
        )
    )

    yearly_display.columns = [
        "Year",
        "Total Sales",
        "YoY Growth"
    ]

    st.dataframe(
        yearly_display,
        use_container_width=True,
        hide_index=True
    )

    
# 10. METHODOLOGY
    

    with st.expander("ℹ️ Analysis methodology"):

        st.write(
            """
            This page focuses on descriptive analysis rather than
            forecasting.

            Monthly sales are aggregated from the order date and
            analyzed over the available historical period.

            The analysis examines:

            • Monthly sales trends
            • Yearly sales performance
            • Year-over-year growth
            • Recurring monthly sales patterns
            • Highest and lowest observed periods

            No future sales values are predicted. The results describe
            historical patterns present in the dataset.
            """
        )
