import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# FILE PATH SETUP (DEPLOYMENT SAFE)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(BASE_DIR, "kmeans_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        data_path = os.path.join(BASE_DIR, "Mall_Customers.csv")
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Data load error: {e}")
        return None

# --------------------------------------------------
# LOAD CLUSTERED DATA
# --------------------------------------------------
@st.cache_data
def load_clustered_data():
    try:
        clustered_path = os.path.join(BASE_DIR, "clustered_mall_customers.csv")
        return pd.read_csv(clustered_path)
    except Exception as e:
        st.error(f"Clustered data load error: {e}")
        return None

# --------------------------------------------------
# LOAD EVERYTHING
# --------------------------------------------------
model = load_model()
df = load_data()
clustered_df = load_clustered_data()

# --------------------------------------------------
# CLUSTER INFO
# --------------------------------------------------
CLUSTER_INFO = {
    0: {"name": "High Value Customers", "color": "#FF6B6B"},
    1: {"name": "Potential Target", "color": "#4ECDC4"},
    2: {"name": "Average Customers", "color": "#45B7D1"},
    3: {"name": "Loyal Customers", "color": "#FFA07A"},
    4: {"name": "Budget Conscious", "color": "#98D8C8"},
}

# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("🛍️ Mall Customer Clustering Prediction")
st.write("Predict customer segments and discover which group your customer belongs to!")

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if model is not None and df is not None and clustered_df is not None:

    col1, col2 = st.columns(2)

    # --------------------------
    # INPUT SECTION
    # --------------------------
    with col1:
        st.subheader("📊 Customer Information")

        age = st.slider(
            "Age",
            int(df["Age"].min()),
            int(df["Age"].max()),
            30
        )

        income = st.slider(
            "Annual Income (k$)",
            int(df["Annual Income (k$)"].min()),
            int(df["Annual Income (k$)"].max()),
            50
        )

        spending = st.slider(
            "Spending Score (1-100)",
            1,
            100,
            50
        )

    # --------------------------
    # DATASET STATS
    # --------------------------
    with col2:
        st.subheader("📈 Dataset Statistics")

        st.metric("Total Customers", len(df))
        st.metric("Average Age", f"{df['Age'].mean():.1f}")
        st.metric("Average Income", f"{df['Annual Income (k$)'].mean():.1f}k")
        st.metric("Average Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

    st.divider()

    # --------------------------
    # PREDICTION
    # --------------------------
    if st.button("🚀 Predict Cluster", use_container_width=True):

        input_df = pd.DataFrame({
            "Age": [age],
            "Annual Income (k$)": [income],
            "Spending Score (1-100)": [spending]
        })

        # Scale using same logic
        scaler = StandardScaler()
        scaler.fit(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])
        scaled_input = scaler.transform(input_df)

        cluster = model.predict(scaled_input)[0]

        st.success(f"Predicted Cluster: {cluster} - {CLUSTER_INFO[cluster]['name']}")

        # --------------------------
        # CLUSTER STATS
        # --------------------------
        cluster_data = clustered_df[clustered_df["Cluster"] == cluster]

        st.info(
            f"""
            Cluster Size: {len(cluster_data)} customers  
            Average Age: {cluster_data['Age'].mean():.1f}  
            Average Income: {cluster_data['Annual Income (k$)'].mean():.1f}k  
            Average Spending: {cluster_data['Spending Score (1-100)'].mean():.1f}
            """
        )

        # --------------------------
        # 3D PLOT
        # --------------------------
        fig = px.scatter_3d(
            clustered_df,
            x="Age",
            y="Annual Income (k$)",
            z="Spending Score (1-100)",
            color=clustered_df["Cluster"].astype(str),
            title="3D Cluster Distribution",
        )

        fig.add_scatter3d(
            x=[age],
            y=[income],
            z=[spending],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Your Input"
        )

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------
        # PIE CHART
        # --------------------------
        cluster_counts = clustered_df["Cluster"].value_counts().sort_index()

        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"Cluster {i}" for i in cluster_counts.index],
            values=cluster_counts.values
        )])

        fig_pie.update_layout(title="Cluster Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.error("Failed to load model or data. Please ensure files are present.")

st.divider()
st.caption("Mall Customer Clustering | Powered by K-Means")
