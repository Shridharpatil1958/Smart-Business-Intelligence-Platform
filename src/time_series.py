"""Time Series Forecasting for Sales."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class SalesForecaster:
    """Forecasts future sales using time series models."""
    
    def __init__(self):
        self.arima_model = None
        self.daily_sales = None
        self.forecast_results = {}
    
    def prepare_data(self, sales_df):
        """Prepare daily sales data for forecasting."""
        df = sales_df.copy()
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        
        self.daily_sales = df.groupby('sale_date').agg(
            daily_revenue=('revenue', 'sum'),
            order_count=('revenue', 'count')
        ).reset_index()
        
        self.daily_sales = self.daily_sales.sort_values('sale_date')
        self.daily_sales.set_index('sale_date', inplace=True)
        
        date_range = pd.date_range(
            start=self.daily_sales.index.min(),
            end=self.daily_sales.index.max(),
            freq='D'
        )
        self.daily_sales = self.daily_sales.reindex(date_range, fill_value=0)
        self.daily_sales.index.name = 'sale_date'
        
        self.daily_sales['revenue_smooth'] = self.daily_sales['daily_revenue'].rolling(
            window=7, min_periods=1
        ).mean()
        
        return self.daily_sales
    
    def decompose(self, period=30):
        """Decompose time series into trend, seasonal, and residual."""
        if self.daily_sales is None or len(self.daily_sales) < period * 2:
            return None
        
        try:
            decomposition = seasonal_decompose(
                self.daily_sales['daily_revenue'],
                model='additive',
                period=period
            )
            return decomposition
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return None
    
    def forecast_arima(self, steps=30, order=(5, 1, 2)):
        """Forecast using ARIMA model."""
        if self.daily_sales is None:
            return None
        
        try:
            series = self.daily_sales['revenue_smooth'].dropna()
            
            if len(series) < 30:
                return None
            
            model = ARIMA(series, order=order)
            self.arima_model = model.fit()
            
            forecast = self.arima_model.forecast(steps=steps)
            conf_int = self.arima_model.get_forecast(steps=steps).conf_int()
            
            last_date = self.daily_sales.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast.values,
                'lower_bound': conf_int.iloc[:, 0].values,
                'upper_bound': conf_int.iloc[:, 1].values
            })
            
            forecast_df['forecast'] = forecast_df['forecast'].clip(lower=0)
            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
            
            self.forecast_results['ARIMA'] = forecast_df
            return forecast_df
        
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            return None
    
    def forecast_prophet_style(self, steps=30):
        """Forecast using Prophet-like approach (simplified moving average + seasonality)."""
        if self.daily_sales is None:
            return None
        
        try:
            series = self.daily_sales['daily_revenue'].copy()
            
            trend = series.rolling(window=30, min_periods=1).mean()
            
            series_with_dow = pd.DataFrame({'revenue': series})
            series_with_dow['dow'] = series_with_dow.index.dayofweek
            weekly_pattern = series_with_dow.groupby('dow')['revenue'].mean()
            
            last_date = self.daily_sales.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            
            base_value = trend.iloc[-1] if not pd.isna(trend.iloc[-1]) else series.mean()
            
            forecasts = []
            for date in forecast_dates:
                dow = date.dayofweek
                seasonal_factor = weekly_pattern[dow] / weekly_pattern.mean() if weekly_pattern.mean() > 0 else 1
                forecast_value = base_value * seasonal_factor
                forecasts.append(max(0, forecast_value))
            
            std = series.std()
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecasts,
                'lower_bound': [max(0, f - 1.96 * std) for f in forecasts],
                'upper_bound': [f + 1.96 * std for f in forecasts]
            })
            
            self.forecast_results['Prophet'] = forecast_df
            return forecast_df
        
        except Exception as e:
            logger.error(f"Prophet-style forecasting failed: {e}")
            return None
    
    def get_forecast_summary(self):
        """Get summary of forecasts."""
        summary = {}
        for model_name, forecast_df in self.forecast_results.items():
            if forecast_df is not None:
                summary[model_name] = {
                    'total_forecasted_revenue': round(forecast_df['forecast'].sum(), 2),
                    'avg_daily_revenue': round(forecast_df['forecast'].mean(), 2),
                    'max_daily_revenue': round(forecast_df['forecast'].max(), 2),
                    'min_daily_revenue': round(forecast_df['forecast'].min(), 2)
                }
        return summary