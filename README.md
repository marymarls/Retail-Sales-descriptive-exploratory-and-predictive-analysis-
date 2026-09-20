# Retail-Sales-descriptive-exploratory-and-predictive-analysis-
# Retail Analytics

An interactive Business Intelligence application for exploring global retail performance, customer behavior, and historical sales patterns.

## Overview

**Retail Analytics** is a data analytics and visualization project built with Python and Streamlit using the Global Superstore dataset.

The application transforms transactional retail data into interactive dashboards and business-oriented insights through:

- 🌍 Geographic sales and profit analysis
- 👥 Customer behavioral segmentation
- 📊 Historical sales trend analysis
-  Business KPIs and insights

##  Features

###  Geographic Analysis

Explore sales and profit performance across countries through an interactive map.

The analysis allows users to examine:

- Sales by country
- Profit by country
- Performance across different years
- Geographic differences in retail performance

###  Customer Segmentation

Customers are grouped into four behavioral profiles using **K-Means clustering**.

The segmentation is based on:

- Total sales
- Total profit
- Average discount
- Order frequency
- Total quantity purchased

The resulting profiles are:

- **High-Value VIP**
- **Core Regular Customers**
- **Efficient Small-Spenders**
- **Unprofitable Discount-Driven**

Before clustering, the features are standardized using `StandardScaler`.

###  Sales Analysis

The application performs descriptive analysis of historical sales data, including:

- Monthly sales trends
- Yearly sales performance
- Year-over-year growth
- Recurring monthly patterns
- Highest and lowest observed periods

No future sales forecasting is performed in the current version.

##  Methodology

The project combines:

**Data Preparation → Exploratory Analysis → Customer Segmentation → Interactive Visualization → Business Insights**

Customer profiles are created using the combination of **customer name and country**, since customer names may be reused across countries in the source dataset.

K-Means clustering is then applied to standardized behavioral features.

The application uses four clusters to provide granular and business-interpretable customer profiles.

##  Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Plotly**
- **Streamlit**

##  Dataset

The project uses the **Global Superstore** dataset, covering approximately **2011–2014** and containing around **20,000 orders across 141 countries**.

The dataset is a public/sample retail dataset and does not represent the operations of a real company.

##  Limitations

- The dataset is sample/public retail data rather than data from a real company.
- Customer names may be reused across countries.
- Customer segmentation therefore represents **customer-country behavioral profiles**, rather than verified unique individuals.
- Historical patterns describe the available dataset and should not automatically be interpreted as predictions of future performance.
