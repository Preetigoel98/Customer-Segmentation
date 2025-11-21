# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Customer Segmentation Dashboard")

# -------------------------------
# Load dataset
# -------------------------------
try:
    df = pd.read_csv("marketing_campaign_featured.csv")
    st.success(f"Dataset loaded: {df.shape[0]} customers, {df.shape[1]} columns")
except FileNotFoundError:
    st.error("Dataset not found! Please place 'marketing_campaign_featured.csv' in the same folder as app.py")
    st.stop()

# -------------------------------
# Detect available cluster columns
# -------------------------------
cluster_cols = [col for col in df.columns if "Cluster" in col]
if not cluster_cols:
    st.warning("No cluster columns found! Please run clustering first and save the CSV.")
    st.stop()

cluster_model = st.selectbox("Select Cluster Model", cluster_cols)
cluster_col = cluster_model

# -------------------------------
# Filter numeric features for plotting
# -------------------------------
numeric_cols = df.select_dtypes(include='number').columns.tolist()
plot_features = [f for f in ['Income','Total_Spending','Total_Purchases','Age','Recency'] if f in numeric_cols]

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")

# Age filter
if 'Age' in numeric_cols:
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
    df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

# Cluster filter
clusters_available = df[cluster_col].unique()
selected_clusters = st.sidebar.multiselect("Select Cluster(s)", options=clusters_available, default=clusters_available)
df = df[df[cluster_col].isin(selected_clusters)]

st.write(f"Filtered dataset: {df.shape[0]} customers")

# -------------------------------
# Bar chart: Income & Spending
# -------------------------------
st.subheader("Cluster Summary: Income & Total Spending")
features_to_plot = [f for f in ['Income','Total_Spending'] if f in numeric_cols]
if features_to_plot and cluster_col in df.columns:
    cluster_summary = df.groupby(cluster_col)[features_to_plot].mean().reset_index()
    cluster_melt = cluster_summary.melt(id_vars=cluster_col, value_vars=features_to_plot,
                                        var_name='Feature', value_name='Value')
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=cluster_col, y='Value', hue='Feature', data=cluster_melt, ax=ax)
    ax.set_title(f"{cluster_model} Clusters: Income & Spending")
    st.pyplot(fig)

# -------------------------------
# Pie chart: Cluster distribution
# -------------------------------
st.subheader("Cluster Distribution")
if cluster_col in df.columns:
    cluster_counts = df[cluster_col].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    ax2.set_title(f"{cluster_model} Cluster Sizes")
    st.pyplot(fig2)

# -------------------------------
# Heatmap: Feature means per cluster
# -------------------------------
st.subheader("Cluster Feature Heatmap")
if plot_features and cluster_col in df.columns:
    cluster_heatmap = df.groupby(cluster_col)[plot_features].mean()
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.heatmap(cluster_heatmap, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    ax3.set_title(f"{cluster_model} Cluster Feature Means")
    st.pyplot(fig3)

# -------------------------------
# Scatter plot: Recency vs Total Spending
# -------------------------------
if 'Recency' in numeric_cols and 'Total_Spending' in numeric_cols and cluster_col in df.columns:
    st.subheader("Recency vs Total Spending")
    fig4, ax4 = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='Recency', y='Total_Spending', hue=cluster_col, data=df, palette='Set2', s=60)
    ax4.set_title(f"{cluster_model} Clusters: Recency vs Total Spending")
    st.pyplot(fig4)

# -------------------------------
# Data Table
# -------------------------------
st.subheader("Filtered Customer Data")
st.dataframe(df)

