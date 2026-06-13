"""Customer Churn Prediction using Classification Models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import os
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Predicts customer churn using multiple classification models."""
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
        }
        self.trained_models = {}
        self.metrics = {}
        self.best_model_name = None
        self.feature_names = None
        self.confusion_matrices = {}
        self.roc_data = {}
    
    def train(self, X, y, test_size=0.2):
        """Train all classification models and evaluate."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        best_f1 = -np.inf
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_prob)
                
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
                
                self.metrics[name] = {
                    'Accuracy': round(accuracy, 4),
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1 Score': round(f1, 4),
                    'ROC AUC': round(roc_auc, 4),
                    'CV_Mean': round(cv_scores.mean(), 4),
                    'CV_Std': round(cv_scores.std(), 4)
                }
                
                self.trained_models[name] = model
                self.confusion_matrices[name] = confusion_matrix(y_test, y_pred)
                
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                self.roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_model_name = name
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.metrics
    
    def train_ensemble(self, X, y, test_size=0.2):
        """Train ensemble models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=1000, random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'))
            ],
            voting='soft'
        )
        
        bagging_clf = BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('xgb', XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss'))
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        
        ensemble_models = {
            'Voting Classifier': voting_clf,
            'Bagging': bagging_clf,
            'Stacking': stacking_clf
        }
        
        ensemble_metrics = {}
        for name, model in ensemble_models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                ensemble_metrics[name] = {
                    'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                    'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
                    'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
                    'F1 Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
                    'ROC AUC': round(roc_auc_score(y_test, y_prob), 4)
                }
                
                self.trained_models[name] = model
            except Exception as e:
                logger.error(f"Error training ensemble {name}: {e}")
        
        return ensemble_metrics
    
    def predict(self, X, model_name=None):
        """Predict churn probability."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found.")
        
        prediction = model.predict(X)
        probability = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else prediction
        
        return prediction, probability
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance."""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.trained_models.get(model_name)
        if model and hasattr(model, 'feature_importances_') and self.feature_names:
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        return None
    
    def save_models(self):
        """Save trained models."""
        for name, model in self.trained_models.items():
            filepath = os.path.join(Config.MODELS_DIR, f"classification_{name.replace(' ', '_').lower()}.joblib")
            joblib.dump(model, filepath)
    
    @staticmethod
    def get_churn_risk_category(probability):
        """Categorize churn risk."""
        if probability >= 0.7:
            return "High Risk", "🔴"
        elif probability >= 0.4:
            return "Medium Risk", "🟡"
        else:
            return "Low Risk", "🟢"