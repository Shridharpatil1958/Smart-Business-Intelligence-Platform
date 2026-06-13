"""Customer Segmentation using Clustering Models."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """Segments customers using clustering algorithms."""
    
    def __init__(self):
        self.models = {}
        self.labels = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        self.cluster_profiles = {}
    
    def find_optimal_k(self, X, max_k=10):
        """Find optimal number of clusters using elbow method."""
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        optimal_k = list(K_range)[np.argmax(silhouette_scores)]
        
        return {
            'K_range': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
    
    def train_kmeans(self, X, n_clusters=4):
        """Train KMeans clustering."""
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        self.models['KMeans'] = model
        self.labels['KMeans'] = labels
        self.metrics['KMeans'] = {
            'Silhouette Score': round(sil_score, 4),
            'Calinski-Harabasz': round(ch_score, 2),
            'n_clusters': n_clusters
        }
        
        return labels
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        """Train DBSCAN clustering."""
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            mask = labels != -1
            sil_score = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else 0
        else:
            sil_score = 0
        
        self.models['DBSCAN'] = model
        self.labels['DBSCAN'] = labels
        self.metrics['DBSCAN'] = {
            'Silhouette Score': round(sil_score, 4),
            'n_clusters': n_clusters,
            'noise_points': int((labels == -1).sum())
        }
        
        return labels
    
    def train_hierarchical(self, X, n_clusters=4):
        """Train Hierarchical (Agglomerative) clustering."""
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        self.models['Hierarchical'] = model
        self.labels['Hierarchical'] = labels
        self.metrics['Hierarchical'] = {
            'Silhouette Score': round(sil_score, 4),
            'Calinski-Harabasz': round(ch_score, 2),
            'n_clusters': n_clusters
        }
        
        return labels
    
    def generate_cluster_profiles(self, original_df, labels, feature_names):
        """Generate descriptive profiles for each cluster."""
        df = original_df.copy()
        df['Cluster'] = labels
        
        profiles = {}
        for cluster_id in sorted(df['Cluster'].unique()):
            if cluster_id == -1:
                continue
            cluster_data = df[df['Cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(df) * 100, 1)
            }
            
            for feat in feature_names:
                if feat in cluster_data.columns:
                    profile[f'{feat}_mean'] = round(cluster_data[feat].mean(), 2)
                    profile[f'{feat}_std'] = round(cluster_data[feat].std(), 2)
            
            profiles[cluster_id] = profile
        
        self.cluster_profiles = profiles
        return profiles
    
    def get_persona_names(self, profiles):
        """Assign persona names based on cluster characteristics."""
        personas = {}
        for cluster_id, profile in profiles.items():
            spending = profile.get('total_spending_mean', 0)
            tenure = profile.get('tenure_months_mean', 0)
            
            if spending > 20000 and tenure > 36:
                personas[cluster_id] = "💎 Premium Loyalists"
            elif spending > 15000:
                personas[cluster_id] = "🌟 High-Value Customers"
            elif spending > 8000 and tenure > 20:
                personas[cluster_id] = "📈 Growing Regulars"
            elif tenure < 12:
                personas[cluster_id] = "🆕 New Customers"
            else:
                personas[cluster_id] = "💰 Budget Shoppers"
        
        return personas