"""Model Performance Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classification import ChurnPredictor
from src.regression import SalesPredictor
from src.ensemble import EnsembleLearner
from src.explainability import ModelExplainer
from src.data_preprocessing import DataPreprocessor


def render_model_performance(data):
    """Render the model performance page."""
    st.markdown("## 🤖 Model Performance & Explainability")
    st.markdown("Compare model performances and understand predictions with Explainable AI.")
    st.markdown("---")
    
    customers_df = data.get('customers', pd.DataFrame())
    sales_df = data.get('sales', pd.DataFrame())
    
    if customers_df.empty:
        st.warning("No data available for model training.")
        return
    
    preprocessor = DataPreprocessor()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Classification Models", "🔬 Ensemble Learning", "🧠 Explainable AI (SHAP)"])
    
    with tab1:
        st.markdown("### Classification Model Comparison")
        
        X, y, feature_cols = preprocessor.prepare_churn_data(customers_df)
        predictor = ChurnPredictor()
        
        with st.spinner("Training all classification models..."):
            metrics = predictor.train(X, y)
        
        # Metrics comparison
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'Model'
        
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Bar chart comparison
        fig = go.Figure()
        for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=list(metrics.keys()),
                y=[m[metric_name] for m in metrics.values()]
            ))
        fig.update_layout(barmode='group', height=400, template='plotly_white',
                         title='Model Metrics Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrices
        st.markdown("### Confusion Matrices")
        cols = st.columns(min(len(predictor.confusion_matrices), 4))
        for i, (name, cm) in enumerate(predictor.confusion_matrices.items()):
            with cols[i % len(cols)]:
                fig = px.imshow(cm, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              x=['Not Churned', 'Churned'],
                              y=['Not Churned', 'Churned'],
                              color_continuous_scale='Blues')
                fig.update_layout(height=250, title=name, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Ensemble Learning Comparison")
        
        X_churn, y_churn, _ = preprocessor.prepare_churn_data(customers_df)
        ensemble = EnsembleLearner()
        
        with st.spinner("Training ensemble models..."):
            ensemble_metrics = ensemble.train_all(X_churn, y_churn)
        
        if ensemble_metrics:
            comparison_df = ensemble.get_comparison_df()
            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1 Score'], color='lightgreen'),
                        use_container_width=True, hide_index=True)
            
            # Radar chart
            fig = go.Figure()
            for _, row in comparison_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['F1 Score'], row['CV Mean']],
                    theta=['Accuracy', 'F1 Score', 'CV Mean'],
                    fill='toself',
                    name=row['Model']
                ))
            fig.update_layout(height=400, polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Ensemble training failed.")
    
    with tab3:
        st.markdown("### SHAP Feature Importance")
        st.markdown("*Understanding which features drive model predictions.*")
        
        X_explain, y_explain, feature_names = preprocessor.prepare_churn_data(customers_df)
        
        # Train a model for explanation
        churn_model = ChurnPredictor()
        churn_model.train(X_explain, y_explain)
        
        best_model = churn_model.trained_models.get(churn_model.best_model_name)
        
        if best_model and hasattr(best_model, 'feature_importances_'):
            explainer = ModelExplainer()
            
            with st.spinner("Generating SHAP explanations..."):
                success = explainer.explain_model(best_model, X_explain, feature_names)
            
            if success:
                importance_df = explainer.get_feature_importance()
                if importance_df is not None:
                    fig = px.bar(importance_df.head(15), x='SHAP Importance', y='Feature',
                               orientation='h', color='SHAP Importance',
                               color_continuous_scale='RdYlGn_r')
                    fig.update_layout(height=500, template='plotly_white',
                                    title=f'SHAP Feature Importance ({churn_model.best_model_name})')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("#### 📖 Interpretation Guide")
                    st.markdown("""
                    - **Higher SHAP value** = Feature has more influence on predictions
                    - **Positive SHAP** = Feature pushes prediction toward churn
                    - **Negative SHAP** = Feature pushes prediction away from churn
                    """)
            else:
                st.warning("SHAP explanation generation failed. Showing standard feature importance instead.")
                importance_df = churn_model.get_feature_importance()
                if importance_df is not None:
                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                               color='Importance', color_continuous_scale='Viridis')
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Using standard feature importance (SHAP requires tree-based models).")
            importance_df = churn_model.get_feature_importance()
            if importance_df is not None:
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                           color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)