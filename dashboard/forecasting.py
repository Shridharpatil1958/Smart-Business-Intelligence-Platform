"""Time Series Forecasting Dashboard Page."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.time_series import SalesForecaster


def render_forecasting(data):
    """Render the time series forecasting page."""
    st.markdown("## 📊 Sales Forecasting")
    st.markdown("Forecast future sales revenue using time series analysis.")
    st.markdown("---")
    
    sales_df = data.get('sales', pd.DataFrame())
    
    if sales_df.empty:
        st.warning("No sales data available.")
        return
    
    forecaster = SalesForecaster()
    
    with st.spinner("Preparing time series data..."):
        daily_sales = forecaster.prepare_data(sales_df)
    
    # Historical Data
    st.markdown("### 📈 Historical Sales Data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sales.index, y=daily_sales['daily_revenue'],
        mode='lines', name='Daily Revenue',
        line=dict(color='rgba(102, 126, 234, 0.3)', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=daily_sales.index, y=daily_sales['revenue_smooth'],
        mode='lines', name='7-Day Moving Average',
        line=dict(color='#667eea', width=3)
    ))
    fig.update_layout(height=400, template='plotly_white',
                     xaxis_title='Date', yaxis_title='Revenue ($)')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Forecasting Settings
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
    with col2:
        model_choice = st.selectbox("Forecasting Model", ['ARIMA', 'Prophet-Style', 'Both'])
    
    if st.button("🔮 Generate Forecast", type="primary"):
        forecasts = {}
        
        if model_choice in ['ARIMA', 'Both']:
            with st.spinner("Running ARIMA forecast..."):
                arima_forecast = forecaster.forecast_arima(steps=forecast_days)
                if arima_forecast is not None:
                    forecasts['ARIMA'] = arima_forecast
        
        if model_choice in ['Prophet-Style', 'Both']:
            with st.spinner("Running Prophet-style forecast..."):
                prophet_forecast = forecaster.forecast_prophet_style(steps=forecast_days)
                if prophet_forecast is not None:
                    forecasts['Prophet'] = prophet_forecast
        
        if forecasts:
            st.markdown("### 🔮 Forecast Results")
            
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=daily_sales.index[-90:], y=daily_sales['revenue_smooth'].iloc[-90:],
                mode='lines', name='Historical (Smoothed)',
                line=dict(color='#667eea', width=2)
            ))
            
            colors = {'ARIMA': '#f5576c', 'Prophet': '#43e97b'}
            
            for model_name, forecast_df in forecasts.items():
                color = colors.get(model_name, '#ffa726')
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['forecast'],
                    mode='lines', name=f'{model_name} Forecast',
                    line=dict(color=color, width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                    y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                    fill='toself', fillcolor=f'rgba({",".join(str(int(color.lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))}, 0.1)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{model_name} Confidence',
                    showlegend=False
                ))
            
            fig.update_layout(height=500, template='plotly_white',
                            xaxis_title='Date', yaxis_title='Revenue ($)',
                            title='Sales Forecast')
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            st.markdown("### 📋 Forecast Summary")
            summary = forecaster.get_forecast_summary()
            
            for model_name, stats in summary.items():
                with st.expander(f"📊 {model_name} Summary"):
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.metric("Total Revenue", f"${stats['total_forecasted_revenue']:,.2f}")
                    with col_s2:
                        st.metric("Avg Daily", f"${stats['avg_daily_revenue']:,.2f}")
                    with col_s3:
                        st.metric("Max Daily", f"${stats['max_daily_revenue']:,.2f}")
                    with col_s4:
                        st.metric("Min Daily", f"${stats['min_daily_revenue']:,.2f}")
        else:
            st.error("Forecasting failed. Please try with different parameters.")
    
    # Decomposition
    st.markdown("---")
    st.markdown("### 🔬 Time Series Decomposition")
    
    decomposition = forecaster.decompose(period=30)
    if decomposition is not None:
        fig = go.Figure()
        
        components = [
            ('Observed', decomposition.observed),
            ('Trend', decomposition.trend),
            ('Seasonal', decomposition.seasonal),
            ('Residual', decomposition.resid)
        ]
        
        for i, (name, component) in enumerate(components):
            fig.add_trace(go.Scatter(
                x=component.index, y=component.values,
                mode='lines', name=name,
                visible=True if i < 2 else 'legendonly'
            ))
        
        fig.update_layout(height=400, template='plotly_white', title='Decomposition Components')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for decomposition analysis.")