"""Sentiment Analysis Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nlp_sentiment import SentimentAnalyzer


def render_sentiment(data):
    """Render the sentiment analysis page."""
    st.markdown("## 💬 Customer Review Sentiment Analysis")
    st.markdown("Analyze customer reviews using NLP and machine learning.")
    st.markdown("---")
    
    reviews_df = data.get('reviews', pd.DataFrame())
    
    if reviews_df.empty:
        st.warning("No review data available.")
        return
    
    analyzer = SentimentAnalyzer()
    
    with st.spinner("Training sentiment models..."):
        metrics = analyzer.train(reviews_df)
    
    # Model Performance
    if metrics:
        st.markdown("### 📊 Model Performance")
        col1, col2 = st.columns(2)
        for i, (name, m) in enumerate(metrics.items()):
            with [col1, col2][i]:
                st.metric(f"{name} Accuracy", f"{m['Accuracy']*100:.1f}%")
    
    st.markdown("---")
    
    # Sentiment Distribution
    st.markdown("### 📊 Sentiment Distribution")
    distribution = analyzer.get_sentiment_distribution(reviews_df)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        dist_df = pd.DataFrame(list(distribution.items()), columns=['Sentiment', 'Count'])
        colors = {'Positive': '#27ae60', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
        dist_df['Color'] = dist_df['Sentiment'].map(colors)
        
        fig = px.pie(dist_df, values='Count', names='Sentiment',
                    color='Sentiment', color_discrete_map=colors)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        fig = px.bar(dist_df, x='Sentiment', y='Count', color='Sentiment',
                    color_discrete_map=colors)
        fig.update_layout(height=350, template='plotly_white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Keywords
    st.markdown("### 🔑 Top Keywords")
    keywords = analyzer.get_top_keywords(reviews_df, n_keywords=15)
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("#### ✅ Positive Keywords")
        if keywords['positive']:
            st.markdown(', '.join([f"`{kw}`" for kw in keywords['positive'][:10]]))
    
    with col_k2:
        st.markdown("#### ❌ Negative Keywords")
        if keywords['negative']:
            st.markdown(', '.join([f"`{kw}`" for kw in keywords['negative'][:10]]))
    
    st.markdown("---")
    
    # Live Sentiment Prediction
    st.markdown("### 🔮 Analyze New Review")
    
    review_input = st.text_area(
        "Enter a review to analyze:",
        placeholder="Type or paste a customer review here...",
        height=100
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if review_input.strip():
            result = analyzer.predict_sentiment(review_input)
            
            sentiment = result['sentiment']
            confidence = result['confidence']
            
            emoji = '😊' if sentiment == 'Positive' else ('😐' if sentiment == 'Neutral' else '😞')
            color = '#27ae60' if sentiment == 'Positive' else ('#f39c12' if sentiment == 'Neutral' else '#e74c3c')
            
            st.markdown(f"""
            <div style="background: {color}; padding: 25px; border-radius: 15px; text-align: center; color: white;">
                <h2 style="margin:0;">{emoji} {sentiment}</h2>
                <p style="margin:10px 0; font-size: 18px;">Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability bar
            st.markdown("<br>", unsafe_allow_html=True)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Positive Probability", f"{result['positive_prob']*100:.1f}%")
            with col_p2:
                st.metric("Negative Probability", f"{result['negative_prob']*100:.1f}%")
        else:
            st.warning("Please enter a review to analyze.")
    
    # Recent Reviews
    st.markdown("---")
    st.markdown("### 📝 Recent Reviews")
    
    prepared = analyzer.prepare_data(reviews_df)
    display_cols = ['review_text', 'rating', 'sentiment']
    if 'product_name' in prepared.columns:
        display_cols.insert(0, 'product_name')
    if 'customer_name' in prepared.columns:
        display_cols.insert(0, 'customer_name')
    
    st.dataframe(
        prepared[display_cols].head(20).rename(columns={
            'customer_name': 'Customer', 'product_name': 'Product',
            'review_text': 'Review', 'rating': 'Rating', 'sentiment': 'Sentiment'
        }),
        use_container_width=True, hide_index=True
    )