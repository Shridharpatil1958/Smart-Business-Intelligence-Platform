"""Sales Prediction Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.regression import SalesPredictor
from src.data_preprocessing import DataPreprocessor


def render_sales_prediction(data):
    """Render the sales prediction page."""
    st.markdown("## 📈 Sales Revenue Prediction")
    st.markdown("Predict future sales revenue using machine learning regression models.")
    st.markdown("---")
    
    sales_df = data.get('sales', pd.DataFrame())
    
    if sales_df.empty:
        st.warning("No sales data available.")
        return
    
    preprocessor = DataPreprocessor()
    predictor = SalesPredictor()
    
    X, y, feature_cols = preprocessor.prepare_sales_data(sales_df)
    
    with st.spinner("Training regression models..."):
        metrics = predictor.train(X, y)
    
    # Display metrics
    st.markdown("### 📊 Model Performance Comparison")
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = 'Model'
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    st.success(f"🏆 Best Model: **{predictor.best_model_name}** (R² = {metrics[predictor.best_model_name]['R2']:.4f})")
    
    st.markdown("---")
    
    # Prediction Form
    st.markdown("### 🔮 Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2)
        discount = st.slider("Discount (%)", 0.0, 50.0, 10.0, 0.5)
        month = st.selectbox("Month", range(1, 13), index=5)
    
    with col2:
        day_of_week = st.selectbox("Day of Week",
                                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                   index=2)
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)
        model_choice = st.selectbox("Select Model", list(metrics.keys()))
    
    dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    
    if st.button("🚀 Predict Revenue", type="primary"):
        input_data = {
            'quantity': quantity,
            'discount_percent': discount,
            'month': month,
            'day_of_week': dow_map[day_of_week],
            'quarter': quarter,
            'is_weekend': 1 if dow_map[day_of_week] >= 5 else 0
        }
        
        for col in feature_cols:
            if col not in input_data:
                input_data[col] = 0
        
        input_df = pd.DataFrame([input_data])[feature_cols]
        prediction = predictor.predict(input_df, model_choice)[0]
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;">
            <h3 style="margin:0;">Predicted Revenue</h3>
            <h1 style="margin:10px 0; font-size: 48px;">${prediction:,.2f}</h1>
            <p style="margin:0;">Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 🎯 Feature Importance")
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted
    st.markdown("### 📉 Actual vs Predicted (Sample)")
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = predictor.predict(X_test, model_choice)
    
    comparison_df = pd.DataFrame({'Actual': y_test.values[:50], 'Predicted': y_pred[:50]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=comparison_df['Actual'], mode='lines+markers', name='Actual', line=dict(color='#667eea')))
    fig.add_trace(go.Scatter(y=comparison_df['Predicted'], mode='lines+markers', name='Predicted', line=dict(color='#f5576c')))
    fig.update_layout(height=350, template='plotly_white', xaxis_title='Sample', yaxis_title='Revenue')
    st.plotly_chart(fig, use_container_width=True)