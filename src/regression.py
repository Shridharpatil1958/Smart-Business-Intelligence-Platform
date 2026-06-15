"""Sales Revenue Prediction using Regression Models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)


class SalesPredictor:
    """Predicts future sales revenue using multiple regression models."""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, learning_rate=0.1,
                max_depth=6, random_state=42
            )
        }
        self.trained_models = {}
        self.metrics = {}
        self.best_model_name = None
        self.feature_names = None
    
    def train(self, X, y, test_size=0.2):
        """Train all regression models and evaluate."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        best_r2 = -np.inf
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                
                self.metrics[name] = {
                    'MAE': round(mae, 2),
                    'RMSE': round(rmse, 2),
                    'R2': round(r2, 4),
                    'CV_Mean': round(cv_scores.mean(), 4),
                    'CV_Std': round(cv_scores.std(), 4)
                }
                
                self.trained_models[name] = model
                
                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model_name = name
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.metrics
    
    def predict(self, X, model_name=None):
        """Make predictions using specified or best model."""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X)
        return predictions
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance from tree-based models."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.trained_models.get(model_name)
        if model is None:
            return None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if self.feature_names:
                return pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
        return None
    
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.trained_models.items():
            filepath = os.path.join(Config.MODELS_DIR, f"regression_{name.replace(' ', '_').lower()}.joblib")
            joblib.dump(model, filepath)
    
    def load_model(self, model_name):
        """Load a saved model from disk."""
        filepath = os.path.join(Config.MODELS_DIR, f"regression_{model_name.replace(' ', '_').lower()}.joblib")
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            self.trained_models[model_name] = model
            return model
        return None