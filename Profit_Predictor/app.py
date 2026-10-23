import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import html
from pathlib import Path
from pathlib import Path
from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

MODEL_PATH = (
    Path(__file__).parent
    / "notebooks"
    / "models"
    / "trained_models.joblib"
)

trained_models = joblib.load(
    Path(__file__).parent / "notebooks" / "models" / "trained_models_compressed.joblib"
)


st.set_page_config(
    page_title="Profit Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- THEME & STYLING ----------
st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: #E5E7EB;
}

[data-testid="stHeader"] {
    background: #0B1220;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

[data-testid="stSidebar"] {
    background: #0E192B;
    border-right: 1px solid #24344A;
}

[data-testid="stSidebar"] h2 {
    color: #5EEAD4;
}

h1, h2, h3 {
    color: #F1F5F9;
}

p, label, li {
    color: #C5D0DF;
}

.hero {
    background: linear-gradient(130deg, #14243D, #10353A);
    border: 1px solid #28505B;
    border-radius: 20px;
    padding: 35px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #5EEAD4;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: 36px;
    line-height: 1.3;
    color: #FFFFFF;
}

.hero p {
    color: #C0CEDD;
    line-height: 1.8;
}

.section-heading {
    color: #F1F5F9;
    font-size: 23px;
    font-weight: 700;
    margin: 24px 0 15px 0;
}

.info-card {
    background: #111C2F;
    border: 1px solid #253750;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 15px;
    min-height: 150px;
}

.info-card h3 {
    color: #5EEAD4;
    font-size: 18px;
}

.info-card p, .info-card li {
    color: #C5D0DF;
    font-size: 14px;
    line-height: 1.8;
}

.metric-card {
    background: #111C2F;
    border: 1px solid #253750;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.filtered-record-card {
    background: #111C2F;
    border: 1px solid #25354C;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0 18px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.filtered-record-label {
    color: #C5D0DF;
    font-size: 15px;
    font-weight: 500;
}

.filtered-record-number {
    color: #5EEAD4;
    font-size: 24px;
    font-weight: 700;
}

.stDownloadButton > button:hover {
    background: #5EEAD4;
    color: #0B1220;
    border: 1px solid #5EEAD4;
    box-shadow: 0 0 14px rgba(45, 212, 191, 0.2);
}

.metric-number {
    color: #5EEAD4;
    font-size: 27px;
    font-weight: 700;
}

.metric-label {
    color: #A9B8CC;
    font-size: 13px;
}



.stDownloadButton > button {
    background: #4F46E5;
    color: #FFFFFF;
    border: 1px solid #6366F1;
    border-radius: 10px;
    padding: 13px 22px;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.3px;
    line-height: 1.5;
    width: 100%;
    transition: all 0.2s ease;
}

.stDownloadButton > button p {
    color: #FFFFFF;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.3px;
}


.stDownloadButton > button:hover {
    background: #6366F1;
    color: #FFFFFF;
    border: 1px solid #818CF8;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.15);
    transform: translateY(-1px);
}

.stDownloadButton > button:hover p {
    color: #FFFFFF;
    font-size: 16px;
    font-weight: 700;
    transition: font-size 0.2s ease;
}


.trading-chart-title {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 25px 0 10px 0;
}

.trading-panel {
    background: #080F1C;
    border: 1px solid #26364D;
    border-radius: 12px;
    padding: 15px 15px 5px 15px;
    margin-bottom: 22px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"] {
    background: #080F1C;
    border-radius: 10px;
}

[data-testid="stVegaLiteChart"] canvas,
[data-testid="stVegaLiteChart"] svg {
    background: #080F1C;
}

.trading-note {
    color: #8292A8;
    font-size: 12px;
    margin-top: 4px;
}

/* ===== PROFIT PREDICTOR: HEADING THEME ===== */

/* Main page headings */
h1 {
    color: #F8FAFC !important;
    font-size: 34px !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
    padding-bottom: 12px;
    border-bottom: 2px solid #14B8A6;
    margin-bottom: 24px !important;
}

/* Section headings */
h2 {
    color: #F8FAFC !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    padding-left: 12px;
    border-left: 4px solid #14B8A6;
    margin-top: 28px !important;
    margin-bottom: 16px !important;
}

/* Subheadings */
h3 {
    color: #5EEAD4 !important;
    font-size: 20px !important;
    font-weight: 650 !important;
    margin-top: 20px !important;
    margin-bottom: 12px !important;
}

/* Smaller headings */
h4, h5, h6 {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}

/* Paragraph and description text */
p {
    color: #94A3B8;
    line-height: 1.7;
}

/* Consistent card styling */
div[data-testid="stMetric"],
div[data-testid="stExpander"] {
    background-color: #111C2F;
    border: 1px solid #263449;
    border-radius: 12px;
    padding: 12px;
}

/* Teal accent for metric labels */
div[data-testid="stMetricLabel"] {
    color: #5EEAD4 !important;
}

/* Smooth appearance */
h1, h2, h3 {
    transition: color 0.2s ease;
}

/* Dataset Explorer heading */
.page-heading {
    display: flex;
    align-items: stretch;
    gap: 16px;
    margin: 12px 0 28px 0;
    padding: 22px;
    background: linear-gradient(135deg, #111C2F, #0D1929);
    border: 1px solid #263449;
    border-radius: 14px;
}

.heading-accent {
    width: 5px;
    min-width: 5px;
    border-radius: 8px;
    background: #14B8A6;
}

.heading-label {
    color: #5EEAD4;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 7px;
}

.custom-heading {
    color: #F8FAFC !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 8px 0 !important;
}

.heading-description {
    color: #94A3B8 !important;
    font-size: 14px;
    margin: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 📊 PROFIT PREDICTOR")
    st.caption("Sales Analytics & Machine Learning")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "🗂️ Dataset Explorer","📊 Sales Dashboard","📦 Product & Regional Analysis","👥 Customer & Delivery Analysis","🤖 Model Evaluation","💰 Profit Prediction"],
        label_visibility="visible"
    )

    st.markdown("---")
    st.caption("Global Superstore Project")
    st.caption("Data Science & Machine Learning")


# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/Global Superstore.csv")


# ---------- PAGE 1: HOME ----------
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">DATA SCIENCE • MACHINE LEARNING</div>
        <h1>Sales Performance Analytics<br>&amp; Profit Prediction</h1>
        <p>
            Explore sales performance, discover profit patterns,
            and use machine learning to estimate expected profit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Project at a Glance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">51,290</div>
            <div class="metric-label">Dataset Records</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">27</div>
            <div class="metric-label">Dataset Columns</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-number">3</div>
            <div class="metric-label">Regression Models</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">01 — Business Problem</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>Why this project?</h3>
        <p>
            Businesses need to identify profitable products, understand
            regional performance, and evaluate how discounts relate to
            profit. This project analyzes historical sales data to support
            better business decisions and profit prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">02 — Project Objectives</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Business Analytics</h3>
            <ul>
                <li>Analyze sales and profit.</li>
                <li>Compare product categories and regions.</li>
                <li>Explore discount and profit patterns.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Machine Learning</h3>
            <ul>
                <li>Compare regression models.</li>
                <li>Evaluate prediction performance.</li>
                <li>Develop a profit prediction feature.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">03 — Dataset Description</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>Global Superstore Dataset</h3>
        <p>
            This dataset contains 51,290 records and 27 columns, including
            Sales, Profit, Quantity, Discount, Category, Product, Customer,
            and Region information.
        </p>
        <p>
            <b>Data limitation:</b> The supplied date fields require
            validation. Some information mentioned in the assignment
            is not available in this dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">04 — Project Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🎓 Student Details</h3>
            <p><b>Student Name:</b> SHAHZAIN</p>
            <p><b>Course:</b> Data Science and Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🧭 Project Scope</h3>
            <p>Exploratory Data Analysis</p>
            <p>Model Evaluation</p>
            <p>Profit Prediction</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        PROFIT PREDICTOR • DATA-DRIVEN BUSINESS INSIGHTS
    </div>
    """, unsafe_allow_html=True)


# ---------- PAGE 2: DATASET EXPLORER ----------
elif page == "🗂️ Dataset Explorer":

    
    st.markdown("""
    <div class="page-heading">
        <div class="heading-accent"></div>
        <div>
            <div class="heading-label">DATA INSIGHTS</div>
            <h1 class="custom-heading">Dataset Explorer</h1>
            <p class="heading-description">
                Explore your dataset, inspect columns, and understand data quality.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    try:
        df = load_data()

        # Overview
        
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(df):,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{len(df.columns)}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{int(df.isna().sum().sum())}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🔎 Filter Dataset")

        filtered_df = df.copy()

        col1, col2 = st.columns(2)

        with col1:
            regions = sorted(
                df["Region"].dropna().astype(str).unique().tolist()
            )

            selected_regions = st.multiselect(
                "Select Region",
                regions,
                default=regions
            )

        with col2:
            categories = sorted(
                df["Category"].dropna().astype(str).unique().tolist()
            )

            selected_categories = st.multiselect(
                "Select Product Category",
                categories,
                default=categories
            )

        filtered_df = filtered_df[
            filtered_df["Region"].astype(str).isin(selected_regions)
            & filtered_df["Category"].astype(str).isin(selected_categories)
        ]

        products = sorted(
            filtered_df["Product Name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_products = st.multiselect(
            "Select Products (optional)",
            options=products,
            placeholder="Leave empty to include all products"
        )

        if selected_products:
            filtered_df = filtered_df[
                filtered_df["Product Name"].astype(str).isin(
                    selected_products
                )
            ]

        st.markdown("---")
        st.subheader("📋 Data Preview")

        
        st.markdown(f"""
        <div class="filtered-record-card">
            <span class="filtered-record-label">
                📋 Filtered Records
            </span>
            <span class="filtered-record-number">
                {len(filtered_df):,}
            </span>
        </div>
        """, unsafe_allow_html=True)


        rows_to_show = st.selectbox(
            "Rows to display",
            [10, 25, 50, 100],
            index=0
        )

        st.dataframe(
            filtered_df.head(rows_to_show),
            use_container_width=True,
            hide_index=True
        )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        
        st.download_button(
            label="⬇️  Download Filtered Dataset (CSV)",
            data=csv_data,
            file_name="filtered_superstore_data.csv",
            mime="text/csv",
            use_container_width=True
        )


        st.caption(
            "Note: Date filtering will be added after the date fields "
            "have been correctly validated."
        )

    except FileNotFoundError:
        st.error(
            "Dataset not found. Check that the file is located at "
            "'data/Global Superstore.csv'."
        )

    except KeyError as error:
        st.error(f"A required dataset column is missing: {error}")

    st.markdown("""
    <div class="footer">
        DATASET EXPLORER • GLOBAL SUPERSTORE
    </div>
    """, unsafe_allow_html=True)

# ---------- PAGE 3: SALES DASHBOARD ----------
elif page == "📊 Sales Dashboard":

    
    
    st.markdown("""
    <div class="page-heading">
        <div class="heading-accent"></div>
        <div>
            <div class="heading-label">SALES ANALYTICS</div>
            <h1 class="custom-heading">Sales Dashboard</h1>
            <p class="heading-description">
                Analyze sales trends, compare product categories, and explore regional performance through interactive visualizations.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    try:
        df = load_data()

        # Calculate KPIs
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()
        total_orders = df["Order ID"].nunique()

        avg_order_value = (
            total_sales / total_orders if total_orders else 0
        )

        avg_profit_per_order = (
            total_profit / total_orders if total_orders else 0
        )

        # KPI Cards
        
        # Calculate additional KPIs
        total_customers = df["Customer ID"].nunique()
        avg_discount = df["Discount"].mean()

        avg_items_per_order = (
            df.groupby("Order ID")["Quantity"].sum().mean()
        )

        order_profit = df.groupby("Order ID")["Profit"].sum()
        profitable_orders_pct = (
            (order_profit > 0).mean() * 100
        )

        # Row 1: 2 cards
        st.markdown("### Business Performance")
        k1, k2 = st.columns(2)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💰 Total Sales</div>
                <div class="metric-number">${total_sales:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📈 Total Profit</div>
                <div class="metric-number">${total_profit:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 2: 3 cards
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🧾 Total Orders</div>
                <div class="metric-number">{total_orders:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🛒 Average Order Value</div>
                <div class="metric-number">${avg_order_value:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💵 Average Profit per Order</div>
                <div class="metric-number">${avg_profit_per_order:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 3: 2 cards
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2 = st.columns(2)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">👥 Total Customers</div>
                <div class="metric-number">{total_customers:,}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🏷️ Average Discount</div>
                <div class="metric-number">{avg_discount:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 4: 2 cards
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2 = st.columns(2)

        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📦 Average Items per Order</div>
                <div class="metric-number">{avg_items_per_order:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">✅ Profitable Orders</div>
                <div class="metric-number">{profitable_orders_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

       
# ----------  CHARTS ----------

    # Sales by Category
        st.markdown(
            '<div class="trading-chart-title">01 / CATEGORY SALES</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="trading-panel">', unsafe_allow_html=True)

        category_sales = (
            df.groupby("Category", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=True)
        )

        st.bar_chart(
            category_sales,
            x="Category",
            y="Sales",
            color="#6366F1",
            horizontal=True
        )

        st.markdown('</div>', unsafe_allow_html=True)


    # Profit by Category
        st.markdown(
            '<div class="trading-chart-title">02 / CATEGORY PROFIT</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="trading-panel">', unsafe_allow_html=True)

        category_profit = (
            df.groupby("Category", as_index=False)["Profit"]
            .sum()
            .sort_values("Profit", ascending=True)
        )

        st.bar_chart(
            category_profit,
            x="Category",
            y="Profit",
            color="#22C55E",
            horizontal=True
        )

        st.markdown('</div>', unsafe_allow_html=True)


        # Sales by Region
        st.markdown(
            '<div class="trading-chart-title">03 / REGIONAL PERFORMANCE</div>',
            unsafe_allow_html=True
        )

        st.markdown('<div class="trading-panel">', unsafe_allow_html=True)

        region_sales = (
            df.groupby("Region", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=True)
        )

        st.bar_chart(
            region_sales,
            x="Region",
            y="Sales",
            color="#38BDF8",
            horizontal=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Dataset limitations
        st.markdown("---")
        st.markdown("### 📌 Data Availability")

        st.info(
            "Return Rate, Average Customer Rating, and Average Delivery "
            "Days cannot be calculated reliably because the required "
            "fields are not available in the supplied dataset."
        )

        st.markdown("""
        <div class="footer">
            SALES DASHBOARD • GLOBAL SUPERSTORE ANALYTICS
        </div>
        """, unsafe_allow_html=True)

    except (FileNotFoundError, KeyError) as error:
        st.error(
            f"Unable to build the dashboard. Check the dataset file "
            f"and required columns. Details: {error}"
        )


# ---------- PAGE 4: PRODUCT & REGIONAL ANALYSIS ----------
elif page == "📦 Product & Regional Analysis":

    df = load_data()

    # ---------- THEME ----------
    PANEL = "#111C2F"
    GRID = "#26364D"
    TEXT = "#E2E8F0"
    MUTED = "#94A3B8"
    TEAL = "#2DD4BF"
    GREEN = "#34D399"
    INDIGO = "#818CF8"
    BLUE = "#38BDF8"
    PURPLE = "#C084FC"

    
    
    st.markdown("""
    <div class="page-heading">
        <div class="heading-accent"></div>
        <div>
            <div class="heading-label">PRODUCT & REGIONAL INSIGHTS</div>
            <h1 class="custom-heading">Product and Regional Analysis</h1>
            <p class="heading-description">
                Explore product category performance, identify profitable products,
                and compare sales and profit across different regions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ---------- CHART STYLE ----------
    def trading_layout(fig, height=350):
        fig.update_layout(
            height=height,
            paper_bgcolor=PANEL,
            plot_bgcolor=PANEL,
            font=dict(
                family="Segoe UI, sans-serif",
                color=TEXT,
                size=11
            ),
            margin=dict(l=15, r=15, t=25, b=15),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=MUTED)
            ),
            hoverlabel=dict(
                bgcolor="#1E293B",
                font_color="#FFFFFF"
            ),
            xaxis=dict(
                gridcolor=GRID,
                zerolinecolor=GRID,
                tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED)
            ),
            yaxis=dict(
                gridcolor=GRID,
                zerolinecolor=GRID,
                tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED)
            )
        )
        return fig

    def chart_panel(title, subtitle=None):
        st.markdown(
            f"""
            <div style="
                background:{PANEL};
                border:1px solid {GRID};
                border-radius:10px;
                padding:14px 16px;
                margin-bottom:8px;
            ">
                <div style="
                    color:{TEXT};
                    font-size:14px;
                    font-weight:700;
                    letter-spacing:0.5px;
                ">{title}</div>
                <div style="
                    color:{MUTED};
                    font-size:11px;
                    margin-top:4px;
                ">{subtitle or ""}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------- KPI CALCULATIONS ----------
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()

    # ---------- TOP KPI CARDS ----------
    k1, k2, k3, k4 = st.columns(4)

    kpis = [
        (k1, "TOTAL SALES", f"${total_sales:,.0f}", TEAL),
        (k2, "TOTAL PROFIT", f"${total_profit:,.0f}", GREEN),
        (k3, "TOTAL ORDERS", f"{total_orders:,}", INDIGO),
        (k4, "CUSTOMERS", f"{total_customers:,}", BLUE),
    ]

    for col, label, value, accent in kpis:
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{PANEL};
                    border:1px solid {GRID};
                    border-top:3px solid {accent};
                    border-radius:10px;
                    padding:17px 14px;
                    min-height:100px;
                ">
                    <div style="
                        color:{MUTED};
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:1px;
                    ">{label}</div>
                    <div style="
                        color:{TEXT};
                        font-size:23px;
                        font-weight:700;
                        margin-top:12px;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- CATEGORY DONUT + SALES AREA ----------
    left, right = st.columns([1, 1.6], gap="medium")

    with left:
        chart_panel(
            "CATEGORY MIX",
            "Share of total sales by category"
        )

        category_mix = (
            df.groupby("Category", as_index=False)["Sales"]
            .sum()
        )

        fig_donut = px.pie(
            category_mix,
            names="Category",
            values="Sales",
            hole=0.70,
            color="Category",
            color_discrete_sequence=[INDIGO, TEAL, BLUE]
        )

        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent",
            marker=dict(
                line=dict(color=PANEL, width=3)
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Sales: $%{value:,.0f}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        )

        fig_donut.update_layout(
            annotations=[
                dict(
                    text="SALES",
                    x=0.5,
                    y=0.52,
                    showarrow=False,
                    font=dict(size=13, color=MUTED)
                )
            ],
            showlegend=True,
            legend=dict(
                orientation="h",
                y=-0.12,
                x=0.5,
                xanchor="center"
            )
        )

        fig_donut = trading_layout(fig_donut, 330)

        st.plotly_chart(
            fig_donut,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:
        chart_panel(
            "CATEGORY SALES PERFORMANCE",
            "Sales distribution across product categories"
        )

        category_trend = (
            df.groupby("Category", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales")
        )

        fig_area = go.Figure()

        fig_area.add_trace(
            go.Scatter(
                x=category_trend["Category"],
                y=category_trend["Sales"],
                mode="lines+markers",
                line=dict(color=TEAL, width=3, shape="spline"),
                marker=dict(
                    color=TEAL,
                    size=9,
                    line=dict(color=PANEL, width=2)
                ),
                fill="tozeroy",
                fillcolor="rgba(45,212,191,0.12)",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Sales: $%{y:,.0f}"
                    "<extra></extra>"
                )
            )
        )

        fig_area.update_layout(
            xaxis_title="Category",
            yaxis_title="Total Sales ($)",
            showlegend=False
        )

        fig_area = trading_layout(fig_area, 330)

        st.plotly_chart(
            fig_area,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("---")

    # ---------- TOP PRODUCTS ----------
    chart_panel(
        "TOP 10 PRODUCTS",
        "Switch between sales and profit to explore product rankings"
    )

    metric_choice = st.radio(
        "Ranking metric",
        ["Sales", "Profit"],
        horizontal=True,
        key="p4_product_metric"
    )

    top_products = (
        df.groupby("Product Name", as_index=False)[metric_choice]
        .sum()
        .nlargest(10, metric_choice)
        .sort_values(metric_choice, ascending=True)
    )

    fig_products = px.bar(
        top_products,
        x=metric_choice,
        y="Product Name",
        orientation="h",
        color=metric_choice,
        color_continuous_scale=["#26364D", INDIGO, TEAL],
        labels={
            metric_choice: f"Total {metric_choice} ($)",
            "Product Name": ""
        }
    )

    fig_products.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            + metric_choice
            + ": $%{x:,.2f}"
            + "<extra></extra>"
        ),
        marker_line_width=0
    )

    fig_products.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(
            categoryorder="total ascending",
            automargin=True
        )
    )

    fig_products = trading_layout(fig_products, 450)

    st.plotly_chart(
        fig_products,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.info(
        "Brand analysis is unavailable because this dataset has no "
        "Brand column. The chart above ranks individual products instead."
    )

    st.markdown("---")

    # ---------- REGIONAL SALES AND PROFIT ----------
    left, right = st.columns(2, gap="medium")

    regional = (
        df.groupby("Region", as_index=False)[["Sales", "Profit"]]
        .sum()
    )

    with left:
        chart_panel(
            "REGIONAL SALES",
            "Total sales by region"
        )

        regional_sales = regional.sort_values("Sales")

        fig_region_sales = px.bar(
            regional_sales,
            x="Sales",
            y="Region",
            orientation="h",
            color_discrete_sequence=[INDIGO],
            labels={"Sales": "Sales ($)", "Region": ""}
        )

        fig_region_sales.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Sales: $%{x:,.0f}"
                "<extra></extra>"
            )
        )

        fig_region_sales = trading_layout(
            fig_region_sales, 440
        )

        st.plotly_chart(
            fig_region_sales,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:
        chart_panel(
            "REGIONAL PROFIT",
            "Positive and negative profit by region"
        )

        regional_profit = regional.sort_values("Profit")

        fig_region_profit = px.bar(
            regional_profit,
            x="Profit",
            y="Region",
            orientation="h",
            color="Profit",
            color_continuous_scale=[
                "#F87171", "#26364D", GREEN
            ],
            labels={"Profit": "Profit ($)", "Region": ""}
        )

        fig_region_profit.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Profit: $%{x:,.0f}"
                "<extra></extra>"
            )
        )

        fig_region_profit.update_layout(
            coloraxis_showscale=False
        )

        fig_region_profit = trading_layout(
            fig_region_profit, 440
        )

        st.plotly_chart(
            fig_region_profit,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("---")

    # ---------- SUB-CATEGORY PERFORMANCE ----------
    chart_panel(
        "SUB-CATEGORY PROFITABILITY",
        "Compare sales and profit across product sub-categories"
    )

    subcategory = (
        df.groupby("Sub-Category", as_index=False)[["Sales", "Profit"]]
        .sum()
        .sort_values("Sales", ascending=True)
    )

    fig_subcategory = go.Figure()

    fig_subcategory.add_trace(
        go.Bar(
            x=subcategory["Sales"],
            y=subcategory["Sub-Category"],
            name="Sales",
            orientation="h",
            marker_color=INDIGO,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Sales: $%{x:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig_subcategory.add_trace(
        go.Scatter(
            x=subcategory["Profit"],
            y=subcategory["Sub-Category"],
            name="Profit",
            mode="markers",
            marker=dict(
                color=GREEN,
                size=10,
                symbol="diamond"
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Profit: $%{x:,.0f}"
                "<extra></extra>"
            )
        )
    )

    fig_subcategory.update_layout(
        barmode="overlay",
        xaxis_title="Amount ($)",
        yaxis_title="",
        legend=dict(orientation="h", y=1.08)
    )

    fig_subcategory = trading_layout(
        fig_subcategory, 520
    )

    st.plotly_chart(
        fig_subcategory,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    # ---------- DATA LIMITATIONS ----------
    st.markdown("---")

    st.caption(
        "Brand and salesperson details are not present in the supplied "
        "dataset. Salesperson performance is therefore not displayed. "
        "All charts use the available Global Superstore records."
    )

# ---------- PAGE 5: CUSTOMER & DELIVERY ANALYSIS ----------
elif page == "👥 Customer & Delivery Analysis":

    df = load_data()

    PANEL = "#111C2F"
    GRID = "#26364D"
    TEXT = "#E2E8F0"
    MUTED = "#94A3B8"
    TEAL = "#2DD4BF"
    INDIGO = "#818CF8"
    BLUE = "#38BDF8"
    PURPLE = "#C084FC"
    GREEN = "#34D399"

    
    st.markdown("""
    <div class="page-heading">
        <div class="heading-accent"></div>
        <div>
            <div class="heading-label">CUSTOMER & DELIVERY INSIGHTS</div>
            <h1 class="custom-heading">Customer & Delivery Analytics</h1>
            <p class="heading-description">
                Analyze customer behavior, shipping methods, and order-level
                insights to better understand business performance.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ---------- REUSABLE CHART STYLE ----------
    def customer_chart_style(fig, height=350):
        fig.update_layout(
            height=height,
            paper_bgcolor=PANEL,
            plot_bgcolor=PANEL,
            font=dict(
                family="Segoe UI, sans-serif",
                color=TEXT,
                size=11
            ),
            margin=dict(l=15, r=15, t=25, b=15),
            hoverlabel=dict(
                bgcolor="#1E293B",
                font_color="#FFFFFF"
            ),
            xaxis=dict(
                gridcolor=GRID,
                zerolinecolor=GRID,
                tickfont=dict(color=MUTED)
            ),
            yaxis=dict(
                gridcolor=GRID,
                zerolinecolor=GRID,
                tickfont=dict(color=MUTED)
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color=MUTED)
            )
        )
        return fig

    def customer_panel(title, subtitle):
        st.markdown(
            f"""
            <div style="
                background:{PANEL};
                border:1px solid {GRID};
                border-radius:10px;
                padding:14px 16px;
                margin-bottom:8px;
            ">
                <div style="
                    color:{TEXT};
                    font-size:14px;
                    font-weight:700;
                ">{title}</div>
                <div style="
                    color:{MUTED};
                    font-size:11px;
                    margin-top:4px;
                ">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------- KPI CARDS ----------
    total_customers = df["Customer ID"].nunique()
    total_orders = df["Order ID"].nunique()
    total_sales = df["Sales"].sum()
    average_customer_sales = (
        total_sales / total_customers if total_customers else 0
    )

    k1, k2, k3, k4 = st.columns(4)

    cards = [
        (k1, "UNIQUE CUSTOMERS", f"{total_customers:,}", TEAL),
        (k2, "TOTAL ORDERS", f"{total_orders:,}", INDIGO),
        (k3, "TOTAL SALES", f"${total_sales:,.0f}", BLUE),
        (
            k4,
            "AVG SALES / CUSTOMER",
            f"${average_customer_sales:,.2f}",
            PURPLE
        )
    ]

    for col, label, value, accent in cards:
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{PANEL};
                    border:1px solid {GRID};
                    border-top:3px solid {accent};
                    border-radius:10px;
                    padding:16px 12px;
                    min-height:100px;
                ">
                    <div style="
                        color:{MUTED};
                        font-size:10px;
                        font-weight:700;
                        letter-spacing:0.5px;
                    ">{label}</div>
                    <div style="
                        color:{TEXT};
                        font-size:21px;
                        font-weight:700;
                        margin-top:12px;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- CUSTOMER SEGMENT MIX ----------
    left, right = st.columns(2, gap="medium")

    with left:
        customer_panel(
            "CUSTOMER SEGMENT MIX",
            "Share of customers by segment"
        )

        segment_customers = (
            df.groupby("Segment", as_index=False)["Customer ID"]
            .nunique()
        )

        fig_segment = px.pie(
            segment_customers,
            names="Segment",
            values="Customer ID",
            hole=0.68,
            color="Segment",
            color_discrete_sequence=[INDIGO, TEAL, BLUE]
        )

        fig_segment.update_traces(
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Customers: %{value:,}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        )

        fig_segment.update_layout(
            showlegend=True,
            legend=dict(orientation="h", y=-0.12)
        )

        fig_segment = customer_chart_style(fig_segment, 340)

        st.plotly_chart(
            fig_segment,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:
        customer_panel(
            "CUSTOMER SEGMENT SALES",
            "Compare total sales across customer segments"
        )

        segment_sales = (
            df.groupby("Segment", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales")
        )

        fig_segment_sales = px.bar(
            segment_sales,
            x="Sales",
            y="Segment",
            orientation="h",
            color="Segment",
            color_discrete_sequence=[INDIGO, TEAL, BLUE],
            labels={"Sales": "Sales ($)", "Segment": ""}
        )

        fig_segment_sales.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Sales: $%{x:,.0f}"
                "<extra></extra>"
            )
        )

        fig_segment_sales = customer_chart_style(
            fig_segment_sales, 340
        )

        st.plotly_chart(
            fig_segment_sales,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("---")

    # ---------- TOP CUSTOMERS ----------
    customer_panel(
        "TOP 10 CUSTOMERS",
        "Rank customers by total sales or total profit"
    )

    customer_metric = st.radio(
        "Rank by",
        ["Sales", "Profit"],
        horizontal=True,
        key="p5_customer_metric"
    )

    top_customers = (
        df.groupby("Customer Name", as_index=False)[customer_metric]
        .sum()
        .nlargest(10, customer_metric)
        .sort_values(customer_metric)
    )

    fig_customers = px.bar(
        top_customers,
        x=customer_metric,
        y="Customer Name",
        orientation="h",
        color=customer_metric,
        color_continuous_scale=["#26364D", INDIGO, TEAL],
        labels={
            customer_metric: f"Total {customer_metric} ($)",
            "Customer Name": ""
        }
    )

    fig_customers.update_layout(coloraxis_showscale=False)

    fig_customers.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            + customer_metric
            + ": $%{x:,.2f}"
            + "<extra></extra>"
        )
    )

    fig_customers = customer_chart_style(
        fig_customers, 430
    )

    st.plotly_chart(
        fig_customers,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown("---")

    # ---------- SHIPPING MODE ANALYSIS ----------
    left, right = st.columns(2, gap="medium")

    with left:
        customer_panel(
            "SHIPPING MODE DISTRIBUTION",
            "Number of orders associated with each shipping mode"
        )

        shipping_counts = (
            df.groupby("Ship Mode")["Order ID"]
            .nunique()
            .reset_index(name="Orders")
            .sort_values("Orders")
        )

        fig_shipping = px.bar(
            shipping_counts,
            x="Orders",
            y="Ship Mode",
            orientation="h",
            color_discrete_sequence=[BLUE],
            labels={"Orders": "Unique orders", "Ship Mode": ""}
        )

        fig_shipping.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Orders: %{x:,}"
                "<extra></extra>"
            )
        )

        fig_shipping = customer_chart_style(
            fig_shipping, 360
        )

        st.plotly_chart(
            fig_shipping,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:
        customer_panel(
            "SHIPPING COST BY MODE",
            "Average shipping cost for each shipping mode"
        )

        shipping_cost = (
            df.groupby("Ship Mode", as_index=False)["Shipping Cost"]
            .mean()
            .sort_values("Shipping Cost")
        )

        fig_shipping_cost = px.bar(
            shipping_cost,
            x="Shipping Cost",
            y="Ship Mode",
            orientation="h",
            color_discrete_sequence=[PURPLE],
            labels={
                "Shipping Cost": "Average cost",
                "Ship Mode": ""
            }
        )

        fig_shipping_cost.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Average shipping cost: %{x:,.2f}"
                "<extra></extra>"
            )
        )

        fig_shipping_cost = customer_chart_style(
            fig_shipping_cost, 360
        )

        st.plotly_chart(
            fig_shipping_cost,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    st.markdown("---")

    # ---------- NEGATIVE PROFIT ANALYSIS ----------
    customer_panel(
        "NEGATIVE-PROFIT ORDERS",
        "Orders whose combined profit is below zero"
    )

    order_profit = (
        df.groupby("Order ID", as_index=False)["Profit"]
        .sum()
    )

    loss_orders = order_profit[
        order_profit["Profit"] < 0
    ]

    profitable_orders = order_profit[
        order_profit["Profit"] > 0
    ]

    loss_count = len(loss_orders)
    loss_pct = (
        loss_count / len(order_profit) * 100
        if len(order_profit) else 0
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Negative-Profit Orders", f"{loss_count:,}")

    with b:
        st.metric("Negative-Profit Order Rate", f"{loss_pct:.1f}%")

    with c:
        st.metric(
            "Positive-Profit Orders",
            f"{len(profitable_orders):,}"
        )

    st.info(
        "Negative profit does not prove that an order was returned. "
        "The dataset does not include return status, customer gender, "
        "age group, ratings, or reliable delivery dates. Those analyses "
        "cannot be calculated from the supplied fields."
    )

    st.caption(
        "CUSTOMER & DELIVERY ANALYTICS • GLOBAL SUPERSTORE"
    )

# ---------- PAGE 6: MODEL EVALUATION ----------
elif page == "🤖 Model Evaluation":

    import numpy as np
    import pandas as pd
    import joblib
    import plotly.express as px
    import streamlit as st
    from pathlib import Path

    # ---------- PREMIUM PAGE STYLING ----------
    st.markdown("""
    <style>
    .model-hero {
        padding: 26px 28px;
        border-radius: 18px;
        background: linear-gradient(
            120deg, #111C2F 0%, #172554 55%, #164E63 100%
        );
        border: 1px solid rgba(45,212,191,0.30);
        margin-bottom: 22px;
    }
    .model-hero h1 {
        color: #F8FAFC;
        font-size: 30px;
        margin: 0;
        font-weight: 800;
    }
    .model-hero p {
        color: #94A3B8;
        margin: 9px 0 0 0;
        font-size: 13px;
        letter-spacing: 1px;
    }
    .metric-card {
        background: linear-gradient(145deg, #111C2F, #0D1728);
        border: 1px solid #263750;
        border-radius: 16px;
        padding: 19px 16px;
        min-height: 130px;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 25px;
        font-weight: 800;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }
    .metric-note {
        color: #2DD4BF;
        font-size: 12px;
        margin-top: 7px;
    }
    .section-note {
        color: #94A3B8;
        font-size: 13px;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="model-hero">
        <h1>🤖 Model Intelligence Center</h1>
        <p>PROFIT PREDICTOR &nbsp; • &nbsp;
        PERFORMANCE ANALYTICS &nbsp; • &nbsp; MODEL DIAGNOSTICS</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- LOAD SAVED RESULTS ----------
    base_path = Path(__file__).parent
    models_dir = base_path / "notebooks" / "models"

    results_path = models_dir / "model_results.joblib"
    model_path = models_dir / "trained_models.joblib"
    feature_path = models_dir / "feature_info.joblib"

    if not results_path.exists():
        st.error("model_results.joblib not found in notebooks/models/.")
        st.stop()

    try:
        model_results = joblib.load(results_path)
    except Exception as e:
        st.error(f"Unable to load evaluation results: {e}")
        st.stop()

    if not isinstance(model_results, dict) or not model_results:
        st.error("The saved model evaluation results are empty or invalid.")
        st.stop()

    # ---------- PREPARE RESULTS TABLE ----------
    results = []

    for name, metrics in model_results.items():
        if not all(key in metrics for key in ["MAE", "RMSE", "R2"]):
            st.error(
                f"Metrics for {name} are incomplete. "
                "Recreate model_results.joblib from the notebook."
            )
            st.stop()

        results.append({
            "Model": name,
            "MAE": float(metrics["MAE"]),
            "MSE": float(metrics["RMSE"]) ** 2,
            "RMSE": float(metrics["RMSE"]),
            "R² Score": float(metrics["R2"])
        })

    results_df = pd.DataFrame(results).sort_values(
        "RMSE", ascending=True
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_mae = best_row["MAE"]
    best_mse = best_row["MSE"]
    best_rmse = best_row["RMSE"]
    best_r2 = best_row["R² Score"]

    # ---------- PERFORMANCE OVERVIEW ----------
    st.markdown("### ⚡ Performance Overview")
    st.markdown(
        '<p class="section-note">'
        'The best model is selected using the lowest saved RMSE.'
        '</p>',
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏆 Best Model</div>
            <div class="metric-value"
                 style="font-size:21px;color:#2DD4BF">
                {best_name}
            </div>
            <div class="metric-note">Lowest RMSE</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 MAE</div>
            <div class="metric-value">{best_mae:,.2f}</div>
            <div class="metric-note">Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📉 RMSE</div>
            <div class="metric-value">{best_rmse:,.2f}</div>
            <div class="metric-note">Lower is better</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 R² SCORE</div>
            <div class="metric-value">{best_r2:.1%}</div>
            <div class="metric-note">Coefficient of determination</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- MODEL PERFORMANCE COMPARISON ----------
    st.markdown("---")
    st.markdown("### 📊 Model Performance Comparison")

    tab1, tab2, tab3 = st.tabs([
        "RMSE Comparison",
        "MAE Comparison",
        "R² Score Comparison"
    ])

    with tab1:
        fig_rmse = px.bar(
            results_df,
            x="Model",
            y="RMSE",
            color="Model",
            text_auto=".2f",
            title="Root Mean Squared Error — Lower is Better",
            template="plotly_dark",
            color_discrete_sequence=[
                "#2DD4BF", "#818CF8", "#F472B6"
            ]
        )
        fig_rmse.update_traces(textposition="outside")
        fig_rmse.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with tab2:
        mae_df = results_df.sort_values("MAE")
        fig_mae = px.bar(
            mae_df,
            x="Model",
            y="MAE",
            color="Model",
            text_auto=".2f",
            title="Mean Absolute Error — Lower is Better",
            template="plotly_dark",
            color_discrete_sequence=[
                "#2DD4BF", "#818CF8", "#F472B6"
            ]
        )
        fig_mae.update_traces(textposition="outside")
        fig_mae.update_layout(
            height=420,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with tab3:
        r2_df = results_df.sort_values(
            "R² Score", ascending=False
        )
        fig_r2 = px.bar(
            r2_df,
            x="Model",
            y="R² Score",
            color="Model",
            text=r2_df["R² Score"].map(lambda x: f"{x:.1%}"),
            title="R² Score — Higher is Better",
            template="plotly_dark",
            color_discrete_sequence=[
                "#2DD4BF", "#818CF8", "#F472B6"
            ]
        )
        fig_r2.update_traces(textposition="outside")
        fig_r2.update_layout(
            height=420,
            showlegend=False,
            yaxis_tickformat=".0%",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # ---------- FEATURE IMPORTANCE ----------
    st.markdown("---")
    st.markdown("### 🧠 Feature Importance")
    model_path = (
        Path(__file__).parent
        / "notebooks"
        / "models"
        / "trained_models_compressed.joblib"
    )
    if not model_path.exists():
        st.info(
            "Saved model file is unavailable. "
            "Feature importance cannot be displayed."
        )
    else:
        try:
            trained_models = joblib.load(
            Path(__file__).parent
            / "notebooks"
            / "models"
            / "trained_models_compressed.joblib"
            )
            best_model = trained_models[best_name]

            feature_names = None

            if feature_path.exists():
                feature_info = joblib.load(feature_path)
                feature_names = feature_info.get("feature_columns")

            if hasattr(best_model, "feature_importances_"):
                importances = np.asarray(
                    best_model.feature_importances_
                ).ravel()

                if feature_names and len(feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).nlargest(15, "Importance").sort_values(
                        "Importance", ascending=True
                    )

                    fig_importance = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 15 Most Important Features",
                        template="plotly_dark",
                        color="Importance",
                        color_continuous_scale=[
                            "#164E63", "#2DD4BF", "#818CF8"
                        ]
                    )
                    fig_importance.update_layout(
                        height=550,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(
                        fig_importance, use_container_width=True
                    )
                else:
                    st.info(
                        "Feature names are missing or do not match "
                        "the model's feature importance values."
                    )

            elif hasattr(best_model, "coef_"):
                coefficients = np.asarray(best_model.coef_).ravel()

                if feature_names and len(feature_names) == len(coefficients):
                    coef_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Coefficient": coefficients
                    })
                    coef_df["Absolute Impact"] = np.abs(
                        coef_df["Coefficient"]
                    )
                    coef_df = coef_df.nlargest(
                        15, "Absolute Impact"
                    ).sort_values("Coefficient")

                    fig_coef = px.bar(
                        coef_df,
                        x="Coefficient",
                        y="Feature",
                        orientation="h",
                        title="Top 15 Model Coefficients",
                        template="plotly_dark",
                        color="Coefficient",
                        color_continuous_scale=[
                            "#F472B6", "#111C2F", "#2DD4BF"
                        ],
                        color_continuous_midpoint=0
                    )
                    fig_coef.update_layout(
                        height=550,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(
                        fig_coef, use_container_width=True
                    )
                else:
                    st.info(
                        "Feature names are missing or do not match "
                        "the model's coefficients."
                    )
            else:
                st.info(
                    f"Feature importance is not directly available "
                    f"for {best_name}."
                )

        except Exception as e:
            st.warning(f"Unable to display feature importance: {e}")

    # ---------- EVALUATION SUMMARY ----------
    st.markdown("---")
    st.markdown("### 💡 Evaluation Insights")

    st.write(
        f"🏆 **Best model:** {best_name}, selected because it "
        "has the lowest saved RMSE."
    )
    st.write(
        f"🎯 **MAE:** {best_mae:,.2f} — average absolute "
        "prediction error on the original evaluation test set."
    )
    st.write(
        f"📉 **RMSE:** {best_rmse:,.2f} — penalizes larger "
        "prediction errors more strongly."
    )
    st.write(
        f"📈 **R² Score:** {best_r2:.1%} — coefficient of "
        "determination on the original evaluation test set."
    )

    if best_r2 < 0:
        st.warning(
            "The model performed worse than a mean-based baseline "
            "on the evaluation test set."
        )
    elif best_r2 < 0.5:
        st.info(
            "The model's explanatory performance is limited. "
            "Review features and data quality."
        )
    else:
        st.success(
            "The model explains a useful proportion of target "
            "variation, but prediction errors still matter."
        )

    # ---------- COMPLETE RESULTS TABLE ----------
    st.markdown("### 📋 Complete Model Evaluation")

    display_df = results_df.copy()
    display_df["R² Score"] = display_df["R² Score"].map(
        lambda value: f"{value:.1%}"
    )

    st.dataframe(
        display_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics are loaded from saved evaluation results. "
        "Actual-vs-predicted and residual plots are not shown "
        "because the large test dataset is not loaded on this page."
    )



# ---------- PAGE 7: PROFIT PREDICTION ----------
elif page == "💰 Profit Prediction":


    # ---------- STYLING ----------
    st.markdown("""
    <style>
    .pp-hero {
        background: linear-gradient(120deg,#07172F,#09264A,#064E63);
        border: 1px solid #164E80;
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 22px;
    }
    .pp-hero h1 {
        color: #F8FAFC;
        font-size: 30px;
        font-weight: 800;
        margin: 0;
    }
    .pp-hero p {
        color: #A9C9EE;
        margin-top: 10px;
    }
    .pp-card {
        background: linear-gradient(145deg,#081B34,#0B2743);
        border: 1px solid #164E80;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .pp-result {
        background: linear-gradient(120deg,#063B43,#092C40);
        border: 1px solid #00BFA6;
        border-radius: 15px;
        padding: 22px;
        margin: 10px 0;
    }
    .pp-number {
        color: #20E3C2;
        font-size: 34px;
        font-weight: 800;
    }
    .pp-muted { color: #A9C9EE; }
    div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(90deg,#12CDB7,#00BFA6);
        color: #031525;
        border: none;
        border-radius: 10px;
        font-weight: 800;
        min-height: 46px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pp-hero">
        <h1>🎯 Profit Prediction</h1>
        <p>Enter order details, estimate profit, and explore data-driven business recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------- PATHS ----------
    base_path = Path(__file__).parent
    model_dir = base_path / "notebooks" / "models"
    data_path = base_path / "data" / "Global Superstore.csv"

    required_files = [
    model_dir / "trained_models_compressed.joblib",
    model_dir / "encoder.joblib",
    model_dir / "feature_info.joblib",
    model_dir / "model_results.joblib",
]

    if not all(p.exists() for p in required_files):
        st.error("Model files missing hain. notebooks/models folder check karo.")
        st.stop()

    if not data_path.exists():
        st.error("CSV file nahi mili: data/Global Superstore.csv")
        st.stop()

    # ---------- LOAD FILES ----------
    try:
          models = joblib.load(
              model_dir / "trained_models_compressed.joblib"
          )
          results = joblib.load(model_dir / "model_results.joblib")
          encoder = joblib.load(model_dir / "encoder.joblib")
          feature_info = joblib.load(model_dir / "feature_info.joblib")
          df = pd.read_csv(data_path)
  
    except Exception as e:
          st.error(f"File loading error: {e}")
          st.stop()

    cat_cols = list(feature_info["cat_cols"])
    num_cols = list(feature_info["num_cols"])
    feature_columns = list(feature_info["feature_columns"])
    model_input_cols = cat_cols + num_cols

    if not models:
        st.error("Koi trained model available nahi hai.")
        st.stop()

    if "Profit" not in df.columns or "Sales" not in df.columns:
        st.error("CSV mein Sales ya Profit column missing hai.")
        st.stop()

    # ---------- FEATURE VALIDATION ----------
    derived_features = {
        "Discount_Amount",
        "Revenue_Before_Discount",
        "Profit_Margin%",
        "Order_Size_Category",
        "Shipping_Cost_Category",
    }

    missing_csv_cols = [
        c for c in model_input_cols
        if c not in df.columns and c not in derived_features
    ]

    if missing_csv_cols:
        st.error(
            "Training features CSV se match nahi kar rahe: "
            + ", ".join(missing_csv_cols)
        )
        st.stop()

    # Profit margin depends on actual profit, so it cannot be a valid
    # input for predicting an unknown profit.
    if "Profit_Margin%" in model_input_cols:
        st.error(
            "Model mein Profit_Margin% input feature hai. "
            "Notebook mein ise model inputs se remove karke model, encoder "
            "aur feature_info dobara train/save karo."
        )
        st.stop()

    # ---------- MODEL SELECTION ----------
    # Use the best model according to saved test RMSE.
  # ---------- MODEL SELECTION ----------
    try:
        best_name = max(
            results,
            key=lambda name: results[name]["R2"]
        )
        best_model = models[best_name]

    except Exception as e:
        st.error(f"Model selection error: {e}")
        st.stop()

        st.caption(f"🤖 Active model: {best_name}")

    # ---------- HELPERS ----------
    def numeric_median(column):
        if column not in df.columns:
            return 0.0

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else 0.0

    def most_common_value(column):
        if column not in df.columns or df[column].dropna().empty:
            return None
        return df[column].dropna().mode().iloc[0]

    def profit_category(value):
        if value < 0:
            return "Loss"
        if value < 50:
            return "Low Profit"
        if value < 200:
            return "Medium Profit"
        return "High Profit"

    # ---------- ORDER FORM ----------
    st.markdown("### 🧾 Order Information")

    excluded_cols = {
        "Profit", "Order ID", "Customer ID", "Customer Name",
        "Product ID", "Row ID", "ji_lu-shu", "Order Date",
        "Ship Date", "Profit_Margin%", "Revenue_Before_Discount",
        "Discount_Amount",
    }

    form_cols = [
        c for c in model_input_cols
        if c in df.columns and c not in excluded_cols
    ]

    default_values = {
        "Sales": 250.0,
        "Quantity": 3,
        "Discount": 0.10,
        "Shipping Cost": 15.0,
        "Year": 2014,
        "weeknum": 25,
    }

    with st.form("profit_prediction_form"):
        input_values = {}
        left, right = st.columns(2)

        for i, col in enumerate(form_cols):
            container = left if i % 2 == 0 else right

            with container:
                if col in num_cols:
                    default = default_values.get(
                        col, numeric_median(col)
                    )

                    step = (
                        1.0
                        if col in ["Quantity", "Year", "weeknum"]
                        else 0.01
                    )

                    input_values[col] = st.number_input(
                        col,
                        value=float(default),
                        step=step,
                        key=f"pp_{col}",
                    )

                else:
                    options = sorted(
                        df[col].dropna().astype(str).unique().tolist()
                    )

                    if not options:
                        st.warning(f"{col}: options available nahi.")
                    else:
                        mode_value = most_common_value(col)
                        mode_value = (
                            str(mode_value) if mode_value is not None else ""
                        )
                        selected_index = (
                            options.index(mode_value)
                            if mode_value in options else 0
                        )

                        input_values[col] = st.selectbox(
                            col,
                            options,
                            index=selected_index,
                            key=f"pp_{col}",
                        )

        submitted = st.form_submit_button(
            "✨ Predict Profit",
            type="primary",
            use_container_width=True,
        )

    # ---------- PREDICTION ----------
    if submitted:
        try:
            raw_input = pd.DataFrame([input_values])

            if "Order_Size_Category" in model_input_cols:
                if "Quantity" not in raw_input.columns:
                    raise ValueError(
                        "Order_Size_Category ke liye Quantity required hai."
                    )

                quantity = float(raw_input["Quantity"].iloc[0])

                if quantity <= 2:
                    raw_input["Order_Size_Category"] = "Small"
                elif quantity <= 5:
                    raw_input["Order_Size_Category"] = "Medium"
                else:
                    raw_input["Order_Size_Category"] = "Large"

            if "Shipping_Cost_Category" in model_input_cols:
                if "Shipping Cost" not in raw_input.columns:
                    raise ValueError(
                        "Shipping_Cost_Category ke liye Shipping Cost required hai."
                    )

                shipping_cost = float(
                    raw_input["Shipping Cost"].iloc[0]
                )

                if shipping_cost < 10:
                    raw_input["Shipping_Cost_Category"] = "Low"
                elif shipping_cost < 30:
                    raw_input["Shipping_Cost_Category"] = "Medium"
                else:
                    raw_input["Shipping_Cost_Category"] = "High"

            # Create numeric engineered features if the model requires them.
            if "Discount_Amount" in model_input_cols:
                if not {"Sales", "Discount"}.issubset(raw_input.columns):
                    raise ValueError(
                        "Discount_Amount ke liye Sales aur Discount required hain."
                    )

                sales = float(raw_input["Sales"].iloc[0])
                discount = float(raw_input["Discount"].iloc[0])
                raw_input["Discount_Amount"] = sales * discount

            if "Revenue_Before_Discount" in model_input_cols:
                if not {"Sales", "Discount"}.issubset(raw_input.columns):
                    raise ValueError(
                        "Revenue_Before_Discount ke liye Sales aur Discount required hain."
                    )

                sales = float(raw_input["Sales"].iloc[0])
                discount = float(raw_input["Discount"].iloc[0])

                # Assumes Sales is net of discount.
                raw_input["Revenue_Before_Discount"] = (
                    sales / (1 - discount) if discount < 1 else sales
                )

            # Fill missing features from the dataset where possible.
            for col in model_input_cols:
                if col not in raw_input.columns:
                    if col in cat_cols:
                        if col not in df.columns:
                            raise ValueError(
                                f"Categorical feature '{col}' ki definition missing hai."
                            )

                        mode_value = most_common_value(col)
                        if mode_value is None:
                            raise ValueError(
                                f"Feature '{col}' mein koi valid category nahi."
                            )
                        raw_input[col] = str(mode_value)

                    elif col in num_cols:
                        if col not in df.columns:
                            raise ValueError(
                                f"Numeric feature '{col}' ki definition missing hai."
                            )
                        raw_input[col] = numeric_median(col)

            # Ensure original training column order.
            raw_input = raw_input.reindex(columns=model_input_cols)

            # Convert numeric features.
            for col in num_cols:
                raw_input[col] = pd.to_numeric(
                    raw_input[col], errors="coerce"
                )

                if raw_input[col].isna().any():
                    raw_input[col] = raw_input[col].fillna(
                        numeric_median(col)
                    )

            # Ensure categorical features are strings.
            for col in cat_cols:
                raw_input[col] = raw_input[col].astype(str)

            # Encode using the saved encoder.
            encoded = encoder.transform(raw_input[cat_cols])

            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()

            encoded_names = encoder.get_feature_names_out(cat_cols)

            encoded_df = pd.DataFrame(
                encoded,
                columns=encoded_names,
                index=raw_input.index,
            )

            numeric_df = raw_input[num_cols].copy()
            final_input = pd.concat([numeric_df, encoded_df], axis=1)

            # Reindex to the saved training feature schema.
            final_input = final_input.reindex(
                columns=feature_columns,
                fill_value=0,
            )

            prediction = float(best_model.predict(final_input)[0])
            category = profit_category(prediction)

            # ---------- RESULT CARDS ----------
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            result_col, category_col = st.columns([2, 1])

            with result_col:
                st.markdown(
                    f"""
                    <div class="pp-result">
                        <div class="pp-muted">Predicted Profit</div>
                        <div class="pp-number">{prediction:,.2f}</div>
                        <div class="pp-muted">Estimated profit for this order</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with category_col:
                st.markdown("#### Profit Category")
                st.metric("Category", category)
                st.caption(
                    "Loss: below 0 | Low: 0–49.99 | "
                    "Medium: 50–199.99 | High: 200+"
                )

            if prediction < 0:
                st.error("This order is predicted to make a loss.")
            elif prediction < 50:
                st.warning("This order is predicted to generate low profit.")
            else:
                st.success("This order is predicted to be profitable.")

            # ---------- INPUT SUMMARY ----------
            st.markdown("### 🧾 Your Input Summary")

            summary_df = pd.DataFrame({
                "Input": list(input_values.keys()),
                "Value": [str(v) for v in input_values.values()],
            })

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ---------- FINAL BUSINESS RECOMMENDATIONS ----------
    st.markdown("---")
    st.markdown("## 💡 Final Business Recommendations")
    st.caption(
        "Findings and evidence are calculated from the Global Superstore dataset."
    )

    rec_df = df.copy()
    rec_df["Sales"] = pd.to_numeric(rec_df["Sales"], errors="coerce")
    rec_df["Profit"] = pd.to_numeric(rec_df["Profit"], errors="coerce")

    recommendations = []

    # 1. Category profitability
    if "Category" in rec_df.columns:
        category_summary = (
            rec_df.groupby("Category")
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
            )
            .sort_values("Profit", ascending=False)
        )

        if not category_summary.empty:
            name = str(category_summary.index[0])
            row = category_summary.iloc[0]

            recommendations.append({
                "title": "📦 Focus on Profitable Categories",
                "finding": f"{name} has the highest total profit by category.",
                "evidence": (
                    f"Total profit: {row['Profit']:,.2f}; "
                    f"total sales: {row['Sales']:,.2f}."
                ),
                "recommendation": (
                    "Review inventory and marketing allocation for profitable "
                    "categories while monitoring their margins."
                ),
                "expected": (
                    "Better resource allocation and potential improvement "
                    "in overall profitability."
                ),
            })

    # 2. Regional profitability
    if "Region" in rec_df.columns:
        region_summary = (
            rec_df.groupby("Region")
            .agg(
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
            )
            .sort_values("Profit", ascending=False)
        )

        if not region_summary.empty:
            name = str(region_summary.index[0])
            row = region_summary.iloc[0]

            recommendations.append({
                "title": "🌍 Improve Regional Performance",
                "finding": f"{name} has the highest total regional profit.",
                "evidence": f"Regional total profit: {row['Profit']:,.2f}.",
                "recommendation": (
                    "Study its product mix, customer demand and discount "
                    "practices before applying successful strategies elsewhere."
                ),
                "expected": (
                    "More informed regional planning and potential profit growth."
                ),
            })

    # 3. Discount analysis
    if "Discount" in rec_df.columns:
        rec_df["Discount"] = pd.to_numeric(
            rec_df["Discount"], errors="coerce"
        )

        discount_summary = (
            rec_df.groupby("Discount")["Profit"]
            .mean()
            .dropna()
            .sort_index()
        )

        if len(discount_summary) >= 2:
            low_discount = discount_summary.index[0]
            high_discount = discount_summary.index[-1]

            recommendations.append({
                "title": "🏷️ Review Discount Policies",
                "finding": (
                    "Average profit differs across observed discount levels."
                ),
                "evidence": (
                    f"Mean profit at discount {low_discount:.2f}: "
                    f"{discount_summary.iloc[0]:,.2f}; at "
                    f"{high_discount:.2f}: {discount_summary.iloc[-1]:,.2f}."
                ),
                "recommendation": (
                    "Review high-discount orders and set discount limits "
                    "based on product profitability."
                ),
                "expected": (
                    "Reduced risk of unprofitable orders while retaining "
                    "effective promotions."
                ),
            })

    # 4. Loss-making products
    if "Product Name" in rec_df.columns:
        product_summary = (
            rec_df.groupby("Product Name")["Profit"]
            .sum()
            .sort_values()
        )
        loss_products = product_summary[product_summary < 0]

        if not loss_products.empty:
            name = str(loss_products.index[0])
            loss = float(loss_products.iloc[0])

            recommendations.append({
                "title": "⚠️ Investigate Loss-Making Products",
                "finding": f"'{name}' has the lowest aggregated profit among loss-making products.",
                "evidence": f"Total product profit: {loss:,.2f}.",
                "recommendation": (
                    "Review purchase cost, shipping cost, pricing and discounts "
                    "for products with negative total profit."
                ),
                "expected": (
                    "Identify opportunities to reduce avoidable losses."
                ),
            })
        else:
            recommendations.append({
                "title": "🔎 Monitor Product Profitability",
                "finding": "No product has negative aggregated profit.",
                "evidence": "Calculated total profit by product.",
                "recommendation": (
                    "Continue monitoring products for declining profit and demand."
                ),
                "expected": (
                    "Earlier identification of future profitability problems."
                ),
            })

    # 5. City sales
    if "City" in rec_df.columns:
        city_summary = (
            rec_df.groupby("City")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if not city_summary.empty:
            city = str(city_summary.index[0])
            city_sales = float(city_summary.iloc[0])

            recommendations.append({
                "title": "🏙️ Learn from High-Sales Cities",
                "finding": f"{city} has the highest total sales.",
                "evidence": f"City-level total sales: {city_sales:,.2f}.",
                "recommendation": (
                    "Examine demand and profitability in leading cities "
                    "before adjusting stock and marketing budgets."
                ),
                "expected": (
                    "Improved demand planning and more effective resource use."
                ),
            })

    # 6. Shipping cost association
    if "Shipping Cost" in rec_df.columns:
        rec_df["Shipping Cost"] = pd.to_numeric(
            rec_df["Shipping Cost"], errors="coerce"
        )
        shipping_data = rec_df[
            ["Shipping Cost", "Profit"]
        ].dropna()

        if (
            len(shipping_data) >= 2
            and shipping_data["Shipping Cost"].nunique() > 1
            and shipping_data["Profit"].nunique() > 1
        ):
            corr = float(
                shipping_data["Shipping Cost"].corr(
                    shipping_data["Profit"]
                )
            )

            recommendations.append({
                "title": "🚚 Monitor Shipping Costs",
                "finding": (
                    "Shipping cost and profit may be associated; "
                    "correlation alone does not prove causation."
                ),
                "evidence": f"Pearson correlation: {corr:.3f}.",
                "recommendation": (
                    "Compare shipping modes and investigate expensive "
                    "shipments with low profit."
                ),
                "expected": (
                    "Potential reduction in avoidable delivery costs "
                    "and improved order margins."
                ),
            })

    # ---------- RENDER RECOMMENDATION CARDS ----------
    if recommendations:
        for i in range(0, len(recommendations), 2):
            columns = st.columns(2)

            for j, container in enumerate(columns):
                index = i + j

                if index >= len(recommendations):
                    break

                rec = recommendations[index]

                with container:
                    st.markdown(
                        f"""
                        <div class="pp-card">
                            <h3 style="color:#20E3C2;font-size:19px;">
                                {html.escape(rec['title'])}
                            </h3>
                            <p style="color:#A9C9EE;">
                                <b>Finding:</b><br>
                                {html.escape(rec['finding'])}
                            </p>
                            <p style="color:#E2E8F0;">
                                <b>Evidence:</b><br>
                                {html.escape(rec['evidence'])}
                            </p>
                            <p style="color:#E2E8F0;">
                                <b>Recommendation:</b><br>
                                {html.escape(rec['recommendation'])}
                            </p>
                            <p style="color:#20E3C2;">
                                <b>Expected Result:</b><br>
                                {html.escape(rec['expected'])}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Recommendations generate karne ke liye required columns available nahi hain.")

    st.caption(
        "Business recommendations dataset-level analysis par based hain. "
        "Expected results potential improvements hain, guarantees nahi."
    )
