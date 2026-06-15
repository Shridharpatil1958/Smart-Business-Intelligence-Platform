"""Anomaly Detection Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.anomaly_detection import AnomalyDetector
from src.data_preprocessing import DataPreprocessor


def render_anomaly(data):
    """Render the anomaly detection page."""
    st.markdown("## 🚨 Anomaly & Fraud Detection")
    st.markdown("Detect suspicious transactions using machine learning algorithms.")
    st.markdown("---")
    
    transactions_df = data.get('transactions', pd.DataFrame())
    
    if transactions_df.empty:
        st.warning("No transaction data available.")
        return
    
    preprocessor = DataPreprocessor()
    detector = AnomalyDetector()
    
    X, y_true, feature_cols = preprocessor.prepare_anomaly_data(transactions_df)
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider("Contamination Rate", 0.01, 0.15, 0.05, 0.01)
    with col2:
        model_choice = st.selectbox("Detection Algorithm", ['Isolation Forest', 'Local Outlier Factor', 'Both'])
    
    if st.button("🔍 Run Anomaly Detection", type="primary"):
        results = {}
        
        if model_choice in ['Isolation Forest', 'Both']:
            with st.spinner("Running Isolation Forest..."):
                labels_if, scores_if = detector.train_isolation_forest(X, contamination)
                if y_true is not None:
                    results['Isolation Forest'] = detector.evaluate(y_true, 'Isolation Forest')
        
        if model_choice in ['Local Outlier Factor', 'Both']:
            with st.spinner("Running Local Outlier Factor..."):
                labels_lof, scores_lof = detector.train_lof(X, contamination)
                if y_true is not None:
                    results['LOF'] = detector.evaluate(y_true, 'LOF')
        
        # Display Results
        st.markdown("### 📊 Detection Results")
        
        if results:
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df, use_container_width=True)
        
        # Summary Cards
        col_s1, col_s2, col_s3 = st.columns(3)
        
        total_transactions = len(transactions_df)
        model_key = 'Isolation Forest' if 'Isolation Forest' in detector.predictions else 'LOF'
        n_anomalies = detector.predictions.get(model_key, {}).get('n_anomalies', 0)
        true_frauds = int(y_true.sum()) if y_true is not None else 'N/A'
        
        with col_s1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col_s2:
            st.metric("🚨 Anomalies Detected", f"{n_anomalies}")
        with col_s3:
            st.metric("Actual Frauds", f"{true_frauds}")
        
        st.markdown("---")
        
        # Visualization
        st.markdown("### 📈 Transaction Amount Distribution")
        
        anomaly_labels = detector.predictions.get(model_key, {}).get('labels', np.zeros(len(transactions_df)))
        
        viz_df = transactions_df.copy()
        viz_df['is_anomaly'] = anomaly_labels[:len(viz_df)]
        viz_df['Status'] = viz_df['is_anomaly'].map({0: 'Normal', 1: 'Anomaly'})
        
        fig = px.scatter(viz_df, x=viz_df.index, y='amount', color='Status',
                        color_discrete_map={'Normal': '#27ae60', 'Anomaly': '#e74c3c'},
                        hover_data=['customer_name', 'merchant_category', 'location'] if 'customer_name' in viz_df.columns else None)
        fig.update_layout(height=400, template='plotly_white',
                         xaxis_title='Transaction Index', yaxis_title='Amount ($)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Amount distribution
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Normal Transactions")
            normal_df = viz_df[viz_df['is_anomaly'] == 0]
            fig = px.histogram(normal_df, x='amount', nbins=30,
                             color_discrete_sequence=['#27ae60'])
            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col_v2:
            st.markdown("#### Anomalous Transactions")
            anomaly_df = viz_df[viz_df['is_anomaly'] == 1]
            fig = px.histogram(anomaly_df, x='amount', nbins=20,
                             color_discrete_sequence=['#e74c3c'])
            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        # Suspicious Transactions Table
        st.markdown("---")
        st.markdown("### 🔴 Suspicious Transactions")
        
        anomalous = detector.get_anomalous_transactions(transactions_df, model_key)
        if not anomalous.empty:
            display_cols = ['transaction_id', 'amount', 'transaction_type', 'merchant_category', 'location']
            if 'customer_name' in anomalous.columns:
                display_cols.insert(1, 'customer_name')
            available_cols = [c for c in display_cols if c in anomalous.columns]
            st.dataframe(anomalous[available_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.success("No suspicious transactions detected!")