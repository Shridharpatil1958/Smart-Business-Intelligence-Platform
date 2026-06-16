"""Customer Churn Prediction Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classification import ChurnPredictor
from src.data_preprocessing import DataPreprocessor


def render_churn_prediction(data):
    """Render the churn prediction page."""
    st.markdown("## 👥 Customer Churn Prediction")
    st.markdown("Predict which customers are likely to churn using classification models.")
    st.markdown("---")
    
    customers_df = data.get('customers', pd.DataFrame())
    
    if customers_df.empty:
        st.warning("No customer data available.")
        return
    
    preprocessor = DataPreprocessor()
    predictor = ChurnPredictor()
    
    X, y, feature_cols = preprocessor.prepare_churn_data(customers_df)
    
    with st.spinner("Training classification models..."):
        metrics = predictor.train(X, y)
    
    # Display metrics
    st.markdown("### 📊 Model Performance Comparison")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    st.success(f"🏆 Best Model: **{predictor.best_model_name}** (F1 = {metrics[predictor.best_model_name]['F1 Score']:.4f})")
    
    st.markdown("---")
    
    # ROC Curves
    st.markdown("### 📈 ROC Curves")
    fig = go.Figure()
    for name, roc in predictor.roc_data.items():
        fig.add_trace(go.Scatter(
            x=roc['fpr'], y=roc['tpr'],
            mode='lines', name=f"{name} (AUC={roc['auc']:.3f})"
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            line=dict(dash='dash', color='gray'), name='Random'))
    fig.update_layout(height=400, template='plotly_white',
                     xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction Form
    st.markdown("### 🔮 Predict Customer Churn")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        total_spending = st.number_input("Total Spending ($)", min_value=0.0, max_value=50000.0, value=10000.0)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=80.0)
    
    with col2:
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=24)
        support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=2)
        gender = st.selectbox("Gender", ['Male', 'Female'])
    
    with col3:
        contract = st.selectbox("Contract Type", ['Monthly', 'Yearly', 'Two-Year'])
        payment = st.selectbox("Payment Method", ['Credit Card', 'Bank Transfer', 'Digital Wallet', 'Cash'])
        model_choice = st.selectbox("Select Model", list(metrics.keys()))
    
    if st.button("🔍 Predict Churn Risk", type="primary"):
        # Build input features
        spending_per_month = total_spending / (tenure + 1)
        tickets_per_month = support_tickets / (tenure + 1)
        charge_ratio = monthly_charges / (total_spending + 1)
        
        # Encode categoricals (simple mapping)
        gender_enc = 1 if gender == 'Male' else 0
        contract_enc = {'Monthly': 0, 'Yearly': 1, 'Two-Year': 2}[contract]
        payment_enc = {'Credit Card': 0, 'Bank Transfer': 1, 'Digital Wallet': 2, 'Cash': 3}[payment]
        
        input_data = {
            'age': age,
            'total_spending': total_spending,
            'monthly_charges': monthly_charges,
            'tenure_months': tenure,
            'support_tickets': support_tickets,
            'gender': gender_enc,
            'contract_type': contract_enc,
            'payment_method': payment_enc,
            'spending_per_month': spending_per_month,
            'tickets_per_month': tickets_per_month,
            'charge_to_spending_ratio': charge_ratio
        }
        
        input_df = pd.DataFrame([input_data])[feature_cols]
        prediction, probability = predictor.predict(input_df, model_choice)
        
        churn_prob = probability[0]
        risk_category, risk_icon = ChurnPredictor.get_churn_risk_category(churn_prob)
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            color = '#e74c3c' if churn_prob >= 0.7 else ('#f39c12' if churn_prob >= 0.4 else '#27ae60')
            st.markdown(f"""
            <div style="background: {color}; padding: 30px; border-radius: 15px; text-align: center; color: white;">
                <h3 style="margin:0;">{risk_icon} Churn Risk</h3>
                <h1 style="margin:10px 0; font-size: 48px;">{churn_prob*100:.1f}%</h1>
                <p style="margin:0; font-size: 18px;">{risk_category}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown("#### Risk Factors:")
            if support_tickets > 4:
                st.markdown("- ⚠️ High number of support tickets")
            if tenure < 12:
                st.markdown("- ⚠️ Short tenure (< 12 months)")
            if contract == 'Monthly':
                st.markdown("- ⚠️ Monthly contract (no commitment)")
            if monthly_charges > 100:
                st.markdown("- ⚠️ High monthly charges")
            if churn_prob < 0.4:
                st.markdown("- ✅ Customer appears stable")
    
    # Feature Importance
    st.markdown("---")
    st.markdown("### 🎯 Feature Importance")
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)