"""Data preprocessing and feature engineering module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing and feature engineering tasks."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.pca = None
    
    def handle_missing_values(self, df, strategy='mean', columns=None):
        """Handle missing values in the dataset."""
        df = df.copy()
        if columns is None:
            columns = df.columns
        
        numeric_cols = df[columns].select_dtypes(include=[np.number]).columns
        categorical_cols = df[columns].select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            self.imputers['numeric'] = num_imputer
        
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.imputers['categorical'] = cat_imputer
        
        return df
    
    def label_encode(self, df, columns):
        """Apply label encoding to specified columns."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[f'label_{col}'] = le
        return df
    
    def one_hot_encode(self, df, columns):
        """Apply one-hot encoding to specified columns."""
        df = pd.get_dummies(df, columns=columns, drop_first=True)
        return df
    
    def standard_scale(self, df, columns):
        """Apply standard scaling to specified columns."""
        df = df.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.scalers['standard'] = scaler
        return df
    
    def apply_pca(self, df, n_components=2):
        """Apply PCA for dimensionality reduction."""
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(df)
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        explained_var = self.pca.explained_variance_ratio_
        return pca_df, explained_var
    
    def feature_selection(self, X, y, k=10, method='f_classif'):
        """Select top k features."""
        if method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        scores = selector.scores_
        
        return X_selected, selected_features, scores
    
    def prepare_churn_data(self, customers_df):
        """Prepare customer data for churn prediction."""
        df = customers_df.copy()
        
        # Feature engineering
        df['spending_per_month'] = df['total_spending'] / (df['tenure_months'] + 1)
        df['tickets_per_month'] = df['support_tickets'] / (df['tenure_months'] + 1)
        df['charge_to_spending_ratio'] = df['monthly_charges'] / (df['total_spending'] + 1)
        
        # Encode categorical variables
        df = self.label_encode(df, ['gender', 'contract_type', 'payment_method', 'city'])
        
        # Select features
        feature_cols = ['age', 'total_spending', 'monthly_charges', 'tenure_months',
                       'support_tickets', 'gender', 'contract_type', 'payment_method',
                       'spending_per_month', 'tickets_per_month', 'charge_to_spending_ratio']
        
        available_cols = [c for c in feature_cols if c in df.columns]
        X = df[available_cols]
        y = df['is_churned']
        
        return X, y, available_cols
    
    def prepare_sales_data(self, sales_df):
        """Prepare sales data for revenue prediction."""
        df = sales_df.copy()
        
        # Time-based features
        if 'sale_date' in df.columns:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            df['month'] = df['sale_date'].dt.month
            df['day_of_week'] = df['sale_date'].dt.dayofweek
            df['quarter'] = df['sale_date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Encode categoricals
        if 'region' in df.columns:
            df = self.label_encode(df, ['region'])
        if 'category' in df.columns:
            df = self.label_encode(df, ['category'])
        
        feature_cols = ['quantity', 'discount_percent', 'month', 'day_of_week',
                       'quarter', 'is_weekend']
        if 'region' in df.columns:
            feature_cols.append('region')
        if 'category' in df.columns:
            feature_cols.append('category')
        if 'price' in df.columns:
            feature_cols.append('price')
        
        available_cols = [c for c in feature_cols if c in df.columns]
        X = df[available_cols].fillna(0)
        y = df['revenue']
        
        return X, y, available_cols
    
    def prepare_clustering_data(self, customers_df):
        """Prepare data for customer segmentation."""
        df = customers_df.copy()
        
        features = ['age', 'total_spending', 'monthly_charges', 'tenure_months', 'support_tickets']
        available = [f for f in features if f in df.columns]
        
        X = df[available].copy()
        X = self.handle_missing_values(X)
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available)
        self.scalers['clustering'] = scaler
        
        return X_scaled, available
    
    def prepare_anomaly_data(self, transactions_df):
        """Prepare transaction data for anomaly detection."""
        df = transactions_df.copy()
        
        # Feature engineering
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['hour'] = df['transaction_date'].dt.hour
            df['day_of_week'] = df['transaction_date'].dt.dayofweek
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Encode categoricals
        encode_cols = []
        if 'transaction_type' in df.columns:
            encode_cols.append('transaction_type')
        if 'merchant_category' in df.columns:
            encode_cols.append('merchant_category')
        if 'location' in df.columns:
            encode_cols.append('location')
        
        if encode_cols:
            df = self.label_encode(df, encode_cols)
        
        feature_cols = ['amount', 'hour', 'day_of_week', 'is_night']
        feature_cols.extend(encode_cols)
        available = [c for c in feature_cols if c in df.columns]
        
        X = df[available].fillna(0)
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        return X, y, available