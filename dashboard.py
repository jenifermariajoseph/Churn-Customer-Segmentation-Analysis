import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(layout="wide", page_title="RavenStack Dashboard")

# Custom CSS for centering elements
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
h1, h2, h3, h4, h5, h6 {
    text-align: center;
}
[data-testid="stMetric"] {
    margin: auto;
    text-align: center;
    align-items: center;
    justify-content: center;
}
[data-testid="stMetricValue"] {
    justify-content: center;
    text-align: center;
}
[data-testid="stMetricLabel"] {
    justify-content: center;
    text-align: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data
def load_data():
    """
    Loads and preprocesses data from CSV files.
    """
    # Load CSVs - Using relative paths
    df_subscriptions = pd.read_csv("ravenstack_subscriptions.csv")
    df_churn = pd.read_csv("ravenstack_churn_events.csv")
    df_accounts = pd.read_csv("ravenstack_accounts.csv")
    df_feature_usage = pd.read_csv("ravenstack_feature_usage.csv")
    df_support = pd.read_csv("ravenstack_support_tickets.csv")

    # Date Conversions
    df_subscriptions["start_date"] = pd.to_datetime(df_subscriptions["start_date"])
    df_subscriptions["end_date"] = pd.to_datetime(df_subscriptions["end_date"])
    df_churn["churn_date"] = pd.to_datetime(df_churn["churn_date"])
    df_accounts["signup_date"] = pd.to_datetime(df_accounts["signup_date"])

    # Merge Account + Subscription info
    # Use suffixes to handle column collisions (like seats, plan_tier)
    accounts_subs = df_accounts.merge(
        df_subscriptions, 
        on="account_id", 
        how="left",
        suffixes=("_acc", "")
    )
    
    # Mark churned accounts
    churned_account_ids = df_churn["account_id"].unique()
    accounts_subs["is_churned"] = accounts_subs["account_id"].isin(churned_account_ids)
    
    # Merge Churn Date into Main View for calculations
    accounts_subs = accounts_subs.merge(df_churn[["account_id", "churn_date"]], on="account_id", how="left")

    # Calculate Total MRR per Account (summing multiple subscriptions if any)
    acc_mrr_sum = df_subscriptions.groupby("account_id")["mrr_amount"].sum().reset_index()
    acc_mrr_sum.columns = ["account_id", "total_mrr"]
    accounts_subs = accounts_subs.merge(acc_mrr_sum, on="account_id", how="left")
    # Use total_mrr as the primary MRR for account-level analysis
    accounts_subs["mrr_amount"] = accounts_subs["total_mrr"].fillna(0)
    
    return {
        "subscriptions": df_subscriptions,
        "churn": df_churn,
        "accounts": df_accounts,
        "feature_usage": df_feature_usage,
        "support": df_support,
        "accounts_subs": accounts_subs
    }

# Load Data
data = load_data()
df_subs = data["subscriptions"]
df_churn = data["churn"]
df_accounts = data["accounts"]
df_feature = data["feature_usage"]
df_support = data["support"]
accounts_subs = data["accounts_subs"]

# ---------------------------
# HEADER & KPIs (Centered)
# ---------------------------
st.title("RavenStack Churn Analysis Dashboard")

# --- KPI Calculations (Matching BA.ipynb Logic) ---
# 1. Total MRR: Sum of ALL active subscriptions (from df_subs directly)
# Use end_date is NaN as proxy for Active, matching Notebook logic.
active_subs_all = df_subs[df_subs["end_date"].isna()]
total_mrr = active_subs_all["mrr_amount"].sum()

# 2. ARPU: Total MRR / Active Subscription Count (from full dataset)
active_sub_count = len(active_subs_all)
arpu = total_mrr / active_sub_count if active_sub_count > 0 else 0

# 3. Churn Rate: Churned Unique Accounts / Total Unique Accounts
# Base comes from df_accounts (500) as per notebook, leading to ~70% rate
churned_unique_acc = df_churn["account_id"].nunique()
total_unique_acc = df_accounts["account_id"].nunique()
churn_rate = (churned_unique_acc / total_unique_acc) * 100 if total_unique_acc > 0 else 0

# Display Centered KPIs (using spacers)
col_spacer1, kpi_col1, kpi_col2, kpi_col3, col_spacer2 = st.columns([1, 2, 2, 2, 1])

with kpi_col1:
    st.metric(label="Total MRR", value=f"${total_mrr:,.2f}")

with kpi_col2:
    st.metric(label="ARPU (Active)", value=f"${arpu:,.2f}")

with kpi_col3:
    st.metric(label="Churn Rate", value=f"{churn_rate:.2f}%")

st.markdown("---")

# ---------------------------
# ROW 2: CHURN ANALYSIS & REASONS
# ---------------------------
st.markdown("<h3 style='text-align: center'>Churn Reason & Analysis</h3>", unsafe_allow_html=True)

row2_col1, row2_col2, row2_col3 = st.columns(3)

# --- LEFT: Monthly Churn Trend ---
with row2_col1:
    st.markdown("##### Monthly Churn Trend")
    monthly_churn = df_churn.set_index("churn_date").resample("M").size().reset_index(name="churn_count")
    
    fig_churn_trend = px.line(
        monthly_churn, 
        x="churn_date", 
        y="churn_count",
        markers=True,
        labels={"churn_date": "Date", "churn_count": "Churn Count"}
    )
    fig_churn_trend.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_churn_trend, use_container_width=True)

# --- MIDDLE: Churn Probability Tables ---
with row2_col2:
    st.markdown("##### Churn Probabilities")
    
    def get_churn_prob(segment_col):
        # Determine pivot columns (True/False or 1/0 for is_churned)
        # We assume boolean from load_data
        pivot = pd.pivot_table(
            accounts_subs,
            index=segment_col,
            columns="is_churned",
            values="account_id",
            aggfunc="nunique",
            fill_value=0
        )
        
        # Ensure Active (False) and Churned (True) columns exist
        if True not in pivot.columns: pivot[True] = 0
        if False not in pivot.columns: pivot[False] = 0
        
        pivot["Total"] = pivot[False] + pivot[True]
        pivot["Churn Rate (%)"] = (pivot[True] / pivot["Total"]) * 100
        return pivot[["Churn Rate (%)"]].sort_values("Churn Rate (%)", ascending=False)

    tab1, tab2, tab3 = st.tabs(["Referral", "Industry", "Country"])
    
    with tab1:
        st.dataframe(get_churn_prob("referral_source").style.format("{:.1f}%"), use_container_width=True)
    with tab2:
        # Use 'industry' based on user request
        st.dataframe(get_churn_prob("industry").style.format("{:.1f}%"), use_container_width=True) 
    with tab3:
        st.dataframe(get_churn_prob("country").style.format("{:.1f}%"), use_container_width=True)

# --- RIGHT: Churn Trend by Reason ---
with row2_col3:
    st.markdown("##### Churn Trend by Reason")
    # Combine Pricing and Budget
    df_churn["reason_group"] = df_churn["reason_code"].replace({"pricing": "Pricing/Budget", "budget": "Pricing/Budget"})
    monthly_churn_reason = df_churn.groupby([pd.Grouper(key="churn_date", freq="M"), "reason_group"]).size().reset_index(name="churn_count")
    
    fig_churn_reason = px.line(
        monthly_churn_reason, 
        x="churn_date", 
        y="churn_count", 
        color="reason_group",
        markers=True,
        labels={"churn_date": "Date", "churn_count": "Count", "reason_group": "Reason"}
    )
    fig_churn_reason.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=True)
    st.plotly_chart(fig_churn_reason, use_container_width=True)

st.markdown("---")

# ---------------------------
# ROW 3: UNIT ECONOMICS (LTV & CAC)
# ---------------------------
st.markdown("<h3 style='text-align: center'>Unit Economics (LTV/CAC)</h3>", unsafe_allow_html=True)

# --- Data Prep for LTV/CAC Time Series (Replicating Notebook Logic) ---
# Create Independent Copy for LTV Calc
l_df_acc = df_accounts.copy()
l_df_acc["signup_month"] = l_df_acc["signup_date"].dt.to_period("M").astype(str)

# Merge Churn Date
ltv_df = l_df_acc[["account_id", "signup_date", "signup_month"]].merge(
    df_churn[["account_id", "churn_date"]], on="account_id", how="left"
)

# Lifetime Months
now = pd.Timestamp.today()
ltv_df["lifetime_months"] = (ltv_df["churn_date"] - ltv_df["signup_date"]).dt.days / 30.0
ltv_df["lifetime_months"] = ltv_df["lifetime_months"].fillna((now - ltv_df["signup_date"]).dt.days / 30.0)
ltv_df["lifetime_months"] = ltv_df["lifetime_months"].clip(lower=3, upper=15) # Cap

# Cohort Lifetime
cohort_lifetime = ltv_df.groupby("signup_month").agg(avg_lifetime_months=("lifetime_months", "mean")).reset_index()

# Cohort ARPU
cohort_arpu = df_subs.merge(l_df_acc[["account_id", "signup_month"]], on="account_id", how="left")
cohort_arpu = cohort_arpu.groupby("signup_month").agg(cohort_arpu=("mrr_amount", "mean")).reset_index()
cohort_arpu["cohort_arpu"] = cohort_arpu["cohort_arpu"].clip(lower=50, upper=200) # Cap

# Cohort LTV (Profit Based)
gross_margin = 0.70
cohort_ltv = cohort_arpu.merge(cohort_lifetime, on="signup_month")
cohort_ltv["monthly_ltv"] = cohort_ltv["cohort_arpu"] * cohort_ltv["avg_lifetime_months"] * gross_margin

# Fully Loaded CAC (Randomized Model from Notebook)
np.random.seed(42)
channel_cost_map = {
    "organic": (300, 600), "partner": (500, 900), "event": (700, 1200),
    "ads": (900, 1600), "other": (500, 1000)
}
l_df_acc["acquisition_cost_usd"] = l_df_acc["referral_source"].apply(
    lambda x: np.random.randint(*channel_cost_map.get(x, (600, 1000)))
)
monthly_cac = l_df_acc.groupby("signup_month").agg(monthly_cac=("acquisition_cost_usd", "mean")).reset_index()

# Ratio
ltv_cac_ratio_df = cohort_ltv.merge(monthly_cac, on="signup_month")
ltv_cac_ratio_df["ltv_cac_ratio"] = ltv_cac_ratio_df["monthly_ltv"] / ltv_cac_ratio_df["monthly_cac"]

# Calculate LTV for Account-Level Analysis (Matches BA.ipynb logic)
now = pd.Timestamp.today()
accounts_subs["end_calc"] = accounts_subs["churn_date"].fillna(now)
# Lifetime in months
accounts_subs["lifetime_months"] = (accounts_subs["end_calc"] - accounts_subs["signup_date"]).dt.days / 30.0

# Apply logic from BA.ipynb: Cap lifetime between 3 and 15 months
accounts_subs["lifetime_months_clipped"] = accounts_subs["lifetime_months"].clip(lower=3, upper=15)

# ARPU logic from BA.ipynb: Clip MRR between 50 and 200
# Note: BA.ipynb used 'mean' MRR, but here we use total account MRR (sum) as base, then clip.
accounts_subs["arpu_clipped"] = accounts_subs["mrr_amount"].clip(lower=50, upper=200)

# Gross Margin 0.70
GROSS_MARGIN = 0.70
accounts_subs["est_ltv"] = accounts_subs["arpu_clipped"] * accounts_subs["lifetime_months_clipped"] * GROSS_MARGIN

# --- Row 3 Columns ---
row3_col1, row3_col2, row3_col3 = st.columns(3)

# 1. LTV/CAC Ratio Graph
with row3_col1:
    st.markdown("##### LTV / CAC Ratio Over Time")
    fig_ltv_line = px.line(ltv_cac_ratio_df, x="signup_month", y="ltv_cac_ratio", markers=True)
    # Add Benchmarks
    fig_ltv_line.add_hline(y=3, line_dash="dash", line_color="green", annotation_text="Healthy (3x)")
    fig_ltv_line.add_hline(y=1, line_dash="dot", line_color="red", annotation_text="Break-even (1x)")
    fig_ltv_line.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_ltv_line, use_container_width=True)

# 2. LTV Distribution Table (Active vs Churned)
# 2. Distribution Mix (Stacked Area)
with row3_col2:
    st.markdown("##### Plan Mix Distribution (LTV Driver)")
    
    # Prepare Data for Stacked Area Mix
    # Group by Signup Month and Plan Tier (proxy for LTV/CAC segments)
    accounts_subs["signup_month_str"] = accounts_subs["signup_date"].dt.to_period("M").astype(str)
    mix_df = accounts_subs.groupby(["signup_month_str", "plan_tier"]).size().reset_index(name="count")
    
    # Sort by month to ensure correct time axis
    mix_df = mix_df.sort_values("signup_month_str")
    
    fig_mix = px.area(
        mix_df, 
        x="signup_month_str", 
        y="count", 
        color="plan_tier", 
        groupnorm='percent', # 100% Stacked Area specific
        title="New Accounts by Plan Tier (Mix %)",
        labels={"signup_month_str": "Month", "count": "Mix", "plan_tier": "Plan"}
    )
    fig_mix.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="Percentage")
    st.plotly_chart(fig_mix, use_container_width=True)

# 3. Avg LTV by Country (Black & White Bar)
with row3_col3:
    st.markdown("##### Avg LTV Analysis")
    # Dropdown for Dimension
    ltv_dim = st.selectbox("View by:", ["Country", "Referral Source"], key="ltv_view_select")
    col_map = {"Country": "country", "Referral Source": "referral_source"}
    sel_col = col_map[ltv_dim]

    avg_ltv = accounts_subs.groupby(sel_col)["est_ltv"].mean().reset_index().sort_values("est_ltv", ascending=False)
    
    fig_ltv = px.bar(
        avg_ltv, x=sel_col, y="est_ltv",
        color_discrete_sequence=['black'],
        title=f"Avg LTV by {ltv_dim}"
    )
    fig_ltv.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_ltv, use_container_width=True)

st.markdown("---")

# ---------------------------
# ROW 4: CUSTOMER SEGMENTATION
# ---------------------------
st.markdown("<h3 style='text-align: center'>Customer Segmentation</h3>", unsafe_allow_html=True)

# Prepare Data
# Feature Usage
usage_agg = df_feature.groupby("subscription_id")[["usage_count", "error_count"]].sum().reset_index()
usage_agg = usage_agg.merge(df_subs[["subscription_id", "account_id"]], on="subscription_id", how="left")
account_usage = usage_agg.groupby("account_id")[["usage_count", "error_count"]].sum().reset_index()

# Support Tickets
support_agg = df_support.groupby("account_id")["ticket_id"].count().reset_index(name="ticket_count")

# Merge
cluster_df = accounts_subs.copy()
cluster_df = cluster_df.merge(account_usage, on="account_id", how="left")
cluster_df = cluster_df.merge(support_agg, on="account_id", how="left")

# Fill NaNs
for col in ["usage_count", "error_count", "ticket_count", "seats"]:
    cluster_df[col] = cluster_df[col].fillna(0)

# Select Features & Scale
features = ["mrr_amount", "seats", "usage_count", "ticket_count"]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_df[features])

# Cluster
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_df["cluster"] = kmeans.fit_predict(scaled_features)

# Labels Mapping (from Notebook logic)
# 0: Low Value – Low Usage
# 1: High Value – Power Users
# 2: Low Value – High Support
# 3: Mid Value – Growing Accounts
cluster_labels = {
    0: "Low Value – Low Usage",
    1: "High Value – Power Users",
    2: "Low Value – High Support",
    3: "Mid Value – Growing Accounts"
}
cluster_df["cluster_label"] = cluster_df["cluster"].map(cluster_labels)

# Display Counts by Category (Simplified)
st.markdown("##### Cluster Counts by Category")
t1, t2, t3 = st.tabs(["By Country", "By Industry", "By Plan Tier"])

def show_cluster_dist(col_name):
    # Aggregation for Stacked Bar Chart
    agg = cluster_df.groupby([col_name, "cluster_label"]).size().reset_index(name="count")
    
    # Simple Stacked Bar Chart
    fig = px.bar(
        agg, 
        x=col_name, 
        y="count", 
        color="cluster_label",
        title=f"Distribution by {col_name.replace('_', ' ').title()}",
        labels={col_name: col_name.replace('_', ' ').title(), "count": "Count", "cluster_label": "Cluster"}
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))

    # Table
    pivot = pd.crosstab(cluster_df["cluster_label"], cluster_df[col_name]).reset_index()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(pivot.set_index("cluster_label").style.background_gradient(cmap="Blues"), use_container_width=True)
    with col2:
        st.plotly_chart(fig, use_container_width=True)

with t1: show_cluster_dist("country")
with t2: show_cluster_dist("industry")
with t3: show_cluster_dist("plan_tier")

st.markdown("---")
st.markdown("---")
# Additional Tables Side-by-Side
col_add1, col_add2 = st.columns(2)

with col_add1:
    st.markdown("##### Referral Source vs Plan Tier")
    ref_plan_dist = pd.crosstab(accounts_subs["referral_source"], accounts_subs["plan_tier"])
    st.dataframe(ref_plan_dist.style.background_gradient(cmap="Greys"), use_container_width=True)

with col_add2:
    st.markdown("##### MRR Analysis by Industry")
    industry_mrr = accounts_subs.groupby("industry").agg(
        avg_mrr=("mrr_amount", "mean"),
        total_mrr=("mrr_amount", "sum"),
        accounts_count=("account_id", "nunique")
    ).reset_index().sort_values("avg_mrr", ascending=False)
    
    st.dataframe(industry_mrr.style.format({
        "avg_mrr": "${:,.2f}", 
        "total_mrr": "${:,.2f}"
    }).background_gradient(cmap="Greys"), use_container_width=True)

st.markdown("---")
