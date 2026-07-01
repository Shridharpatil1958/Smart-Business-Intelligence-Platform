"""Customer Segmentation Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.clustering import CustomerSegmentation
from src.data_preprocessing import DataPreprocessor


def render_segmentation(data):
    """Render the customer segmentation page."""
    st.markdown("## 🎯 Customer Segmentation")
    st.markdown("Segment customers based on purchasing behavior using clustering algorithms.")
    st.markdown("---")
    
    customers_df = data.get('customers', pd.DataFrame())
    
    if customers_df.empty:
        st.warning("No customer data available.")
        return
    
    preprocessor = DataPreprocessor()
    segmenter = CustomerSegmentation()
    
    X_scaled, feature_names = preprocessor.prepare_clustering_data(customers_df)
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)
    with col2:
        algorithm = st.selectbox("Algorithm", ['KMeans', 'Hierarchical', 'DBSCAN'])
    
    with st.spinner(f"Running {algorithm} clustering..."):
        if algorithm == 'KMeans':
            labels = segmenter.train_kmeans(X_scaled, n_clusters)
        elif algorithm == 'Hierarchical':
            labels = segmenter.train_hierarchical(X_scaled, n_clusters)
        else:
            labels = segmenter.train_dbscan(X_scaled, eps=1.0, min_samples=5)
    
    # Metrics
    st.markdown("### 📊 Clustering Metrics")
    metrics = segmenter.metrics.get(algorithm, {})
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Silhouette Score", f"{metrics.get('Silhouette Score', 0):.4f}")
    with col_m2:
        st.metric("Clusters Found", metrics.get('n_clusters', n_clusters))
    with col_m3:
        if 'Calinski-Harabasz' in metrics:
            st.metric("Calinski-Harabasz", f"{metrics['Calinski-Harabasz']:.2f}")
        elif 'noise_points' in metrics:
            st.metric("Noise Points", metrics['noise_points'])
    
    st.markdown("---")
    
    # Visualization
    st.markdown("### 🗺️ Cluster Visualization")
    
    # PCA for 2D visualization
    pca_df, explained_var = preprocessor.apply_pca(X_scaled, n_components=2)
    pca_df['Cluster'] = labels.astype(str)
    pca_df['Customer'] = customers_df['customer_name'].values[:len(pca_df)]
    
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                    hover_data=['Customer'],
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title=f'Customer Segments ({algorithm})')
    fig.update_layout(height=500, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"PCA Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}")
    
    # Cluster Profiles
    st.markdown("---")
    st.markdown("### 👥 Segment Profiles")
    
    profiles = segmenter.generate_cluster_profiles(customers_df, labels, feature_names)
    personas = segmenter.get_persona_names(profiles)
    
    for cluster_id, profile in profiles.items():
        persona_name = personas.get(cluster_id, f"Segment {cluster_id}")
        with st.expander(f"{persona_name} — {profile['size']} customers ({profile['percentage']}%)"):
            cols = st.columns(len(feature_names))
            for i, feat in enumerate(feature_names):
                with cols[i]:
                    mean_val = profile.get(f'{feat}_mean', 0)
                    st.metric(feat.replace('_', ' ').title(), f"{mean_val:.1f}")
    
    # Elbow Method
    if algorithm == 'KMeans':
        st.markdown("---")
        st.markdown("### 📐 Elbow Method Analysis")
        with st.spinner("Computing optimal K..."):
            elbow_data = segmenter.find_optimal_k(X_scaled, max_k=8)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elbow_data['K_range'], y=elbow_data['inertias'],
            mode='lines+markers', name='Inertia',
            line=dict(color='#667eea')
        ))
        fig.update_layout(height=300, template='plotly_white',
                         xaxis_title='Number of Clusters (K)', yaxis_title='Inertia')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 Optimal K based on silhouette score: **{elbow_data['optimal_k']}**")