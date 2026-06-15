"""NLP Sentiment Analysis for Customer Reviews."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes customer review sentiment."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.models = {}
        self.trained_models = {}
        self.metrics = {}
    
    def clean_text(self, text):
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def prepare_data(self, reviews_df):
        """Prepare review data for sentiment analysis."""
        df = reviews_df.copy()
        df['cleaned_text'] = df['review_text'].apply(self.clean_text)
        df['sentiment'] = df['rating'].apply(
            lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
        )
        df['sentiment_binary'] = (df['rating'] >= 4).astype(int)
        return df
    
    def train(self, reviews_df):
        """Train sentiment analysis models."""
        df = self.prepare_data(reviews_df)
        df_binary = df[df['sentiment'] != 'Neutral'].copy()
        
        if len(df_binary) < 10:
            return {}
        
        X_text = df_binary['cleaned_text']
        y = df_binary['sentiment_binary']
        
        X_tfidf = self.tfidf.fit_transform(X_text)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Naive Bayes': MultinomialNB(alpha=1.0)
        }
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.trained_models[name] = model
                self.metrics[name] = {'Accuracy': round(accuracy, 4)}
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.metrics
    
    def predict_sentiment(self, text, model_name='Logistic Regression'):
        """Predict sentiment of a single text."""
        if model_name not in self.trained_models:
            return self._keyword_sentiment(text)
        
        cleaned = self.clean_text(text)
        X = self.tfidf.transform([cleaned])
        
        model = self.trained_models[model_name]
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        confidence = max(probability)
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'positive_prob': round(probability[1] if len(probability) > 1 else 0.5, 3),
            'negative_prob': round(probability[0] if len(probability) > 1 else 0.5, 3)
        }
    
    def _keyword_sentiment(self, text):
        """Simple keyword-based sentiment analysis fallback."""
        positive_words = {'excellent', 'amazing', 'great', 'love', 'best', 'perfect',
                         'wonderful', 'fantastic', 'outstanding', 'recommend', 'satisfied'}
        negative_words = {'terrible', 'horrible', 'worst', 'hate', 'awful', 'poor',
                         'disappointed', 'waste', 'broke', 'bad', 'refund'}
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            return {'sentiment': 'Positive', 'confidence': 0.7, 'positive_prob': 0.7, 'negative_prob': 0.3}
        elif neg_count > pos_count:
            return {'sentiment': 'Negative', 'confidence': 0.7, 'positive_prob': 0.3, 'negative_prob': 0.7}
        else:
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'positive_prob': 0.5, 'negative_prob': 0.5}
    
    def get_sentiment_distribution(self, reviews_df):
        """Get overall sentiment distribution."""
        df = self.prepare_data(reviews_df)
        distribution = df['sentiment'].value_counts().to_dict()
        return distribution
    
    def get_top_keywords(self, reviews_df, n_keywords=20):
        """Get top keywords from reviews."""
        df = self.prepare_data(reviews_df)
        
        pos_texts = ' '.join(df[df['sentiment'] == 'Positive']['cleaned_text'])
        neg_texts = ' '.join(df[df['sentiment'] == 'Negative']['cleaned_text'])
        
        tfidf_temp = TfidfVectorizer(max_features=n_keywords, stop_words='english')
        
        pos_keywords = []
        if pos_texts:
            tfidf_temp.fit_transform([pos_texts])
            pos_keywords = tfidf_temp.get_feature_names_out().tolist()
        
        neg_keywords = []
        if neg_texts:
            tfidf_temp = TfidfVectorizer(max_features=n_keywords, stop_words='english')
            tfidf_temp.fit_transform([neg_texts])
            neg_keywords = tfidf_temp.get_feature_names_out().tolist()
        
        return {'positive': pos_keywords, 'negative': neg_keywords}