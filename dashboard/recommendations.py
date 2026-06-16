"""Product Recommendation Dashboard Page."""

import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommendation import RecommendationEngine


def render_recommendations(data):
    """Render the product recommendation page."""
    st.markdown("## 🛍️ Product Recommendation Engine")
    st.markdown("Get personalized product recommendations using AI-powered algorithms.")
    st.markdown("---")
    
    products_df = data.get('products', pd.DataFrame())
    sales_df = data.get('sales', pd.DataFrame())
    customers_df = data.get('customers', pd.DataFrame())
    
    if products_df.empty or sales_df.empty:
        st.warning("Insufficient data for recommendations.")
        return
    
    engine = RecommendationEngine()
    
    with st.spinner("Building recommendation models..."):
        engine.build_content_based(products_df)
        engine.build_collaborative(sales_df)
    
    st.success("✅ Recommendation models ready!")
    
    st.markdown("---")
    
    # Tabs for different recommendation types
    tab1, tab2, tab3 = st.tabs(["🎯 Content-Based", "👥 Collaborative", "🔀 Hybrid"])
    
    with tab1:
        st.markdown("### Content-Based Filtering")
        st.markdown("*Recommends similar products based on product features.*")
        
        product_options = products_df[['product_id', 'product_name', 'category']].copy()
        product_options['display'] = product_options['product_name'] + ' (' + product_options['category'] + ')'
        
        selected_product = st.selectbox(
            "Select a product to find similar items:",
            options=product_options['product_id'].tolist(),
            format_func=lambda x: product_options[product_options['product_id'] == x]['display'].values[0]
        )
        
        n_recs = st.slider("Number of recommendations", 3, 10, 5, key='content_n')
        
        if st.button("🔍 Find Similar Products", key='content_btn'):
            recs = engine.recommend_content_based(selected_product, n_recs)
            if not recs.empty:
                st.markdown("#### 📋 Recommended Products:")
                for _, row in recs.iterrows():
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    with col1:
                        st.markdown(f"**{row['product_name']}**")
                    with col2:
                        st.markdown(f"📁 {row['category']}")
                    with col3:
                        st.markdown(f"💲{row['price']:.2f}")
                    with col4:
                        st.markdown(f"⭐ {row['rating']:.1f}")
                    st.markdown("---")
            else:
                st.info("No recommendations available for this product.")
    
    with tab2:
        st.markdown("### Collaborative Filtering")
        st.markdown("*Recommends products based on similar customers' purchases.*")
        
        customer_options = customers_df[['customer_id', 'customer_name']].copy()
        
        selected_customer = st.selectbox(
            "Select a customer:",
            options=customer_options['customer_id'].tolist(),
            format_func=lambda x: customer_options[customer_options['customer_id'] == x]['customer_name'].values[0]
        )
        
        n_recs_collab = st.slider("Number of recommendations", 3, 10, 5, key='collab_n')
        
        if st.button("🔍 Get Recommendations", key='collab_btn'):
            recs = engine.recommend_collaborative(selected_customer, n_recs_collab)
            if not recs.empty:
                st.markdown("#### 📋 Recommended for You:")
                st.dataframe(
                    recs[['product_name', 'category', 'price', 'rating']].rename(
                        columns={'product_name': 'Product', 'category': 'Category',
                                'price': 'Price ($)', 'rating': 'Rating'}
                    ),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("Not enough purchase history for collaborative recommendations. Showing top-rated products instead.")
                top_rated = products_df.nlargest(n_recs_collab, 'rating')[['product_name', 'category', 'price', 'rating']]
                st.dataframe(top_rated, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Hybrid Recommendations")
        st.markdown("*Combines both content-based and collaborative filtering for best results.*")
        
        col1, col2 = st.columns(2)
        with col1:
            hybrid_customer = st.selectbox(
                "Select Customer:",
                options=customer_options['customer_id'].tolist(),
                format_func=lambda x: customer_options[customer_options['customer_id'] == x]['customer_name'].values[0],
                key='hybrid_customer'
            )
        with col2:
            hybrid_product = st.selectbox(
                "Based on Product (optional):",
                options=[None] + product_options['product_id'].tolist(),
                format_func=lambda x: 'None' if x is None else product_options[product_options['product_id'] == x]['display'].values[0],
                key='hybrid_product'
            )
        
        if st.button("🚀 Get Hybrid Recommendations", key='hybrid_btn', type='primary'):
            recs = engine.get_hybrid_recommendations(hybrid_customer, hybrid_product, 5)
            if not recs.empty:
                st.markdown("#### 🎁 Top 5 Recommended Products:")
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;
                                border-left: 4px solid #667eea;">
                        <strong>#{i} {row['product_name']}</strong><br>
                        📁 {row['category']} | 💲{row['price']:.2f} | ⭐ {row['rating']:.1f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recommendations available.")