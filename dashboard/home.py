"""Home Dashboard Page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render_home(data):
    """Render the home dashboard page."""
    st.markdown("## 🏠 Business Overview Dashboard")
    st.markdown("---")
    
    sales_df = data.get('sales', pd.DataFrame())
    customers_df = data.get('customers', pd.DataFrame())
    products_df = data.get('products', pd.DataFrame())
    
    total_revenue = sales_df['revenue'].sum() if 'revenue' in sales_df.columns else 0
    total_customers = len(customers_df)
    total_products = len(products_df)
    total_orders = len(sales_df)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h4 style="margin:0; font-size: 14px;">💰 Total Revenue</h4>
            <h2 style="margin:5px 0; font-size: 22px;">${total_revenue:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h4 style="margin:0; font-size: 14px;">👥 Total Customers</h4>
            <h2 style="margin:5px 0; font-size: 22px;">{total_customers:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h4 style="margin:0; font-size: 14px;">🛍️ Total Products</h4>
            <h2 style="margin:5px 0; font-size: 22px;">{total_products:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h4 style="margin:0; font-size: 14px;">📦 Total Orders</h4>
            <h2 style="margin:5px 0; font-size: 22px;">{total_orders:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional KPIs
    col5, col6, col7, col8 = st.columns(4)
    
    avg_order = total_revenue / total_orders if total_orders > 0 else 0
    churn_rate = (customers_df['is_churned'].mean() * 100) if 'is_churned' in customers_df.columns else 0
    avg_rating = products_df['rating'].mean() if 'rating' in products_df.columns else 0
    monthly_rev = total_revenue / 24 if total_revenue > 0 else 0
    
    with col5:
        st.metric("📊 Avg Order Value", f"${avg_order:.2f}")
    with col6:
        st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
    with col7:
        st.metric("⭐ Avg Product Rating", f"{avg_rating:.2f}")
    with col8:
        st.metric("📅 Monthly Revenue", f"${monthly_rev:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📈 Revenue Over Time")
        if not sales_df.empty and 'sale_date' in sales_df.columns:
            daily_rev = sales_df.groupby('sale_date')['revenue'].sum().reset_index()
            daily_rev['sale_date'] = pd.to_datetime(daily_rev['sale_date'])
            daily_rev = daily_rev.sort_values('sale_date')
            daily_rev['rolling_avg'] = daily_rev['revenue'].rolling(7, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_rev['sale_date'], y=daily_rev['revenue'],
                mode='lines', name='Daily Revenue',
                line=dict(color='rgba(102, 126, 234, 0.3)', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=daily_rev['sale_date'], y=daily_rev['rolling_avg'],
                mode='lines', name='7-Day Average',
                line=dict(color='#667eea', width=3)
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### 🏷️ Revenue by Category")
        if not sales_df.empty and 'category' in sales_df.columns:
            cat_rev = sales_df.groupby('category')['revenue'].sum().reset_index()
            fig = px.pie(cat_rev, values='revenue', names='category',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col_left2, col_right2 = st.columns(2)
    
    with col_left2:
        st.markdown("### 🌍 Revenue by Region")
        if not sales_df.empty and 'region' in sales_df.columns:
            region_rev = sales_df.groupby('region')['revenue'].sum().reset_index()
            fig = px.bar(region_rev, x='region', y='revenue',
                        color='revenue', color_continuous_scale='Viridis')
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right2:
        st.markdown("### 👤 Customer Age Distribution")
        if not customers_df.empty and 'age' in customers_df.columns:
            fig = px.histogram(customers_df, x='age', nbins=20,
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Top Products
    st.markdown("### 🏆 Top 10 Products by Revenue")
    if not sales_df.empty and 'product_name' in sales_df.columns:
        top_products = sales_df.groupby('product_name')['revenue'].sum().nlargest(10).reset_index()
        top_products.columns = ['Product', 'Total Revenue']
        top_products['Total Revenue'] = top_products['Total Revenue'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(top_products, use_container_width=True, hide_index=True)