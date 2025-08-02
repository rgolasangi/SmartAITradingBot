"""
Sentiment Analysis Agent for AI Trading System
"""
import os
import logging
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Import models
from src.models.market_data import NewsData, SocialSentiment

load_dotenv()

class SentimentAgent:
    """Advanced sentiment analysis agent for financial text"""
    
    def __init__(self):
        # Database setup
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_agent.db')
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Redis for caching
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        self._initialize_nltk()
        
        # Financial sentiment lexicon
        self.financial_lexicon = self._build_financial_lexicon()
        
        # Models
        self.vectorizer = None
        self.sentiment_model = None
        self.is_trained = False
        
        # Model paths
        self.model_dir = "models/sentiment"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load pre-trained models if available
        self._load_models()
    
    def _initialize_nltk(self):
        """Initialize NLTK components"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
        except Exception as e:
            self.logger.error(f"Error initializing NLTK: {e}")
    
    def _build_financial_lexicon(self) -> Dict[str, float]:
        """Build financial sentiment lexicon with domain-specific terms"""
        lexicon = {
            # Positive terms
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'boom': 0.9,
            'breakout': 0.7, 'uptrend': 0.8, 'gains': 0.6, 'profit': 0.7,
            'strong': 0.6, 'positive': 0.5, 'buy': 0.6, 'long': 0.5,
            'support': 0.4, 'resistance': 0.3, 'momentum': 0.5,
            'outperform': 0.7, 'upgrade': 0.6, 'beat': 0.6,
            
            # Negative terms
            'bearish': -0.8, 'crash': -0.9, 'fall': -0.6, 'decline': -0.6,
            'breakdown': -0.7, 'downtrend': -0.8, 'losses': -0.6, 'loss': -0.5,
            'weak': -0.6, 'negative': -0.5, 'sell': -0.6, 'short': -0.5,
            'correction': -0.4, 'volatility': -0.3, 'risk': -0.4,
            'underperform': -0.7, 'downgrade': -0.6, 'miss': -0.6,
            
            # Neutral but important
            'options': 0.0, 'derivatives': 0.0, 'futures': 0.0,
            'strike': 0.0, 'expiry': 0.0, 'premium': 0.0,
            'nifty': 0.0, 'sensex': 0.0, 'index': 0.0,
            
            # Market conditions
            'volatile': -0.3, 'stable': 0.2, 'uncertain': -0.4,
            'confident': 0.5, 'optimistic': 0.6, 'pessimistic': -0.6,
            'cautious': -0.2, 'aggressive': 0.3
        }
        
        return lexicon
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def calculate_lexicon_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment using financial lexicon"""
        words = text.lower().split()
        
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        word_count = 0
        
        for word in words:
            if word in self.financial_lexicon:
                score = self.financial_lexicon[word]
                word_count += 1
                
                if score > 0:
                    positive_score += score
                elif score < 0:
                    negative_score += abs(score)
                else:
                    neutral_score += 1
        
        # Normalize scores
        if word_count > 0:
            positive_score /= word_count
            negative_score /= word_count
            neutral_score /= word_count
        
        # Calculate overall sentiment
        overall_sentiment = positive_score - negative_score
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'neutral_score': neutral_score,
            'overall_sentiment': overall_sentiment,
            'confidence': min(positive_score + negative_score, 1.0)
        }
    
    def calculate_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment_label': self._polarity_to_label(polarity)
            }
        except Exception as e:
            self.logger.error(f"Error in TextBlob sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment_label': 'neutral'}
    
    def _polarity_to_label(self, polarity: float) -> str:
        """Convert polarity score to sentiment label"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def prepare_training_data(self) -> Tuple[List[str], List[str]]:
        """Prepare training data from collected news and social media"""
        session = self.SessionLocal()
        
        try:
            # Get news data
            news_query = session.query(NewsData).filter(
                NewsData.relevance_score >= 0.3,
                NewsData.published_at >= datetime.now() - timedelta(days=30)
            ).all()
            
            # Get social sentiment data
            social_query = session.query(SocialSentiment).filter(
                SocialSentiment.relevance_score >= 0.3,
                SocialSentiment.posted_at >= datetime.now() - timedelta(days=30)
            ).all()
            
            texts = []
            labels = []
            
            # Process news data
            for news in news_query:
                text = f"{news.title} {news.content}"
                processed_text = self.preprocess_text(text)
                
                if processed_text:
                    texts.append(processed_text)
                    
                    # Use TextBlob for initial labeling
                    tb_sentiment = self.calculate_textblob_sentiment(processed_text)
                    labels.append(tb_sentiment['sentiment_label'])
            
            # Process social media data
            for social in social_query:
                processed_text = self.preprocess_text(social.content)
                
                if processed_text:
                    texts.append(processed_text)
                    
                    # Use TextBlob for initial labeling
                    tb_sentiment = self.calculate_textblob_sentiment(processed_text)
                    labels.append(tb_sentiment['sentiment_label'])
            
            self.logger.info(f"Prepared {len(texts)} training samples")
            return texts, labels
            
        finally:
            session.close()
    
    def train_model(self, retrain: bool = False):
        """Train the sentiment analysis model"""
        try:
            if self.is_trained and not retrain:
                self.logger.info("Model already trained. Use retrain=True to retrain.")
                return
            
            self.logger.info("Training sentiment analysis model...")
            
            # Prepare training data
            texts, labels = self.prepare_training_data()
            
            if len(texts) < 100:
                self.logger.warning("Insufficient training data. Using pre-trained components only.")
                return
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.sentiment_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            self.sentiment_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.sentiment_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            self.logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Save models
            self._save_models()
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training sentiment model: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.vectorizer:
                with open(f"{self.model_dir}/vectorizer.pkl", 'wb') as f:
                    pickle.dump(self.vectorizer, f)
            
            if self.sentiment_model:
                with open(f"{self.model_dir}/sentiment_model.pkl", 'wb') as f:
                    pickle.dump(self.sentiment_model, f)
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            vectorizer_path = f"{self.model_dir}/vectorizer.pkl"
            model_path = f"{self.model_dir}/sentiment_model.pkl"
            
            if os.path.exists(vectorizer_path) and os.path.exists(model_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                with open(model_path, 'rb') as f:
                    self.sentiment_model = pickle.load(f)
                
                self.is_trained = True
                self.logger.info("Pre-trained models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Comprehensive sentiment analysis of text"""
        if not text:
            return self._empty_sentiment_result()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Calculate different sentiment scores
        lexicon_sentiment = self.calculate_lexicon_sentiment(processed_text)
        textblob_sentiment = self.calculate_textblob_sentiment(text)
        
        # ML model prediction (if trained)
        ml_sentiment = {'prediction': 'neutral', 'confidence': 0.0}
        if self.is_trained and self.vectorizer and self.sentiment_model:
            try:
                X = self.vectorizer.transform([processed_text])
                prediction = self.sentiment_model.predict(X)[0]
                probabilities = self.sentiment_model.predict_proba(X)[0]
                confidence = max(probabilities)
                
                ml_sentiment = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': {
                        label: prob for label, prob in 
                        zip(self.sentiment_model.classes_, probabilities)
                    }
                }
            except Exception as e:
                self.logger.error(f"Error in ML sentiment prediction: {e}")
        
        # Ensemble sentiment (combine all methods)
        ensemble_sentiment = self._calculate_ensemble_sentiment(
            lexicon_sentiment, textblob_sentiment, ml_sentiment
        )
        
        return {
            'text': text,
            'processed_text': processed_text,
            'lexicon_sentiment': lexicon_sentiment,
            'textblob_sentiment': textblob_sentiment,
            'ml_sentiment': ml_sentiment,
            'ensemble_sentiment': ensemble_sentiment,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_ensemble_sentiment(self, 
                                    lexicon: Dict[str, float],
                                    textblob: Dict[str, float],
                                    ml: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble sentiment from multiple methods"""
        # Convert all to same scale (-1 to 1)
        lexicon_score = lexicon['overall_sentiment']
        textblob_score = textblob['polarity']
        
        # Convert ML prediction to score
        ml_score = 0.0
        if ml['prediction'] == 'positive':
            ml_score = ml['confidence']
        elif ml['prediction'] == 'negative':
            ml_score = -ml['confidence']
        
        # Weighted ensemble
        weights = {
            'lexicon': 0.4,
            'textblob': 0.3,
            'ml': 0.3 if self.is_trained else 0.0
        }
        
        # Adjust weights if ML model not available
        if not self.is_trained:
            weights['lexicon'] = 0.6
            weights['textblob'] = 0.4
        
        ensemble_score = (
            weights['lexicon'] * lexicon_score +
            weights['textblob'] * textblob_score +
            weights['ml'] * ml_score
        )
        
        # Calculate confidence
        confidence_scores = [
            lexicon.get('confidence', 0.0),
            textblob.get('subjectivity', 0.0),
            ml.get('confidence', 0.0)
        ]
        
        ensemble_confidence = np.mean([c for c in confidence_scores if c > 0])
        
        # Determine final label
        if ensemble_score > 0.1:
            ensemble_label = 'positive'
        elif ensemble_score < -0.1:
            ensemble_label = 'negative'
        else:
            ensemble_label = 'neutral'
        
        return {
            'score': ensemble_score,
            'label': ensemble_label,
            'confidence': ensemble_confidence,
            'method_scores': {
                'lexicon': lexicon_score,
                'textblob': textblob_score,
                'ml': ml_score
            },
            'weights': weights
        }
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment result"""
        return {
            'text': '',
            'processed_text': '',
            'lexicon_sentiment': {'overall_sentiment': 0.0, 'confidence': 0.0},
            'textblob_sentiment': {'polarity': 0.0, 'sentiment_label': 'neutral'},
            'ml_sentiment': {'prediction': 'neutral', 'confidence': 0.0},
            'ensemble_sentiment': {'score': 0.0, 'label': 'neutral', 'confidence': 0.0},
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_batch_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts"""
        results = []
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        
        return results
    
    def get_market_sentiment_summary(self, hours_back: int = 6) -> Dict[str, Any]:
        """Get aggregated market sentiment for the specified period"""
        session = self.SessionLocal()
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Get recent news
            news_query = session.query(NewsData).filter(
                NewsData.published_at >= cutoff_time,
                NewsData.relevance_score >= 0.3
            ).all()
            
            # Get recent social media
            social_query = session.query(SocialSentiment).filter(
                SocialSentiment.posted_at >= cutoff_time,
                SocialSentiment.relevance_score >= 0.3
            ).all()
            
            # Analyze sentiment for all content
            all_sentiments = []
            
            # Process news
            for news in news_query:
                text = f"{news.title} {news.content}"
                sentiment = self.analyze_sentiment(text)
                sentiment['source'] = 'news'
                sentiment['relevance_score'] = news.relevance_score
                all_sentiments.append(sentiment)
            
            # Process social media
            for social in social_query:
                sentiment = self.analyze_sentiment(social.content)
                sentiment['source'] = 'social'
                sentiment['relevance_score'] = social.relevance_score
                sentiment['influence_score'] = social.influence_score
                all_sentiments.append(sentiment)
            
            # Calculate aggregated metrics
            if all_sentiments:
                sentiment_scores = [s['ensemble_sentiment']['score'] for s in all_sentiments]
                confidence_scores = [s['ensemble_sentiment']['confidence'] for s in all_sentiments]
                
                # Weight by relevance and influence
                weighted_scores = []
                weights = []
                
                for sentiment in all_sentiments:
                    weight = sentiment['relevance_score']
                    if 'influence_score' in sentiment:
                        weight *= sentiment['influence_score']
                    
                    weighted_scores.append(sentiment['ensemble_sentiment']['score'] * weight)
                    weights.append(weight)
                
                # Calculate weighted average
                if sum(weights) > 0:
                    weighted_avg_sentiment = sum(weighted_scores) / sum(weights)
                else:
                    weighted_avg_sentiment = np.mean(sentiment_scores)
                
                # Count sentiment labels
                labels = [s['ensemble_sentiment']['label'] for s in all_sentiments]
                label_counts = {
                    'positive': labels.count('positive'),
                    'negative': labels.count('negative'),
                    'neutral': labels.count('neutral')
                }
                
                summary = {
                    'total_items': len(all_sentiments),
                    'news_items': len(news_query),
                    'social_items': len(social_query),
                    'average_sentiment': np.mean(sentiment_scores),
                    'weighted_sentiment': weighted_avg_sentiment,
                    'average_confidence': np.mean(confidence_scores),
                    'sentiment_distribution': label_counts,
                    'sentiment_trend': self._calculate_sentiment_trend(all_sentiments),
                    'top_positive': [s for s in all_sentiments if s['ensemble_sentiment']['label'] == 'positive'][:5],
                    'top_negative': [s for s in all_sentiments if s['ensemble_sentiment']['label'] == 'negative'][:5],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache summary
                self.redis_client.setex(
                    f'market_sentiment_summary_{hours_back}h',
                    300,  # 5 minutes
                    json.dumps(summary, default=str)
                )
                
                return summary
            
            else:
                return {
                    'total_items': 0,
                    'average_sentiment': 0.0,
                    'message': 'No sentiment data available for the specified period'
                }
                
        finally:
            session.close()
    
    def _calculate_sentiment_trend(self, sentiments: List[Dict[str, Any]]) -> str:
        """Calculate sentiment trend over time"""
        if len(sentiments) < 2:
            return 'insufficient_data'
        
        # Sort by timestamp
        sorted_sentiments = sorted(sentiments, key=lambda x: x['timestamp'])
        
        # Split into two halves
        mid_point = len(sorted_sentiments) // 2
        first_half = sorted_sentiments[:mid_point]
        second_half = sorted_sentiments[mid_point:]
        
        # Calculate average sentiment for each half
        first_avg = np.mean([s['ensemble_sentiment']['score'] for s in first_half])
        second_avg = np.mean([s['ensemble_sentiment']['score'] for s in second_half])
        
        # Determine trend
        diff = second_avg - first_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'deteriorating'
        else:
            return 'stable'
    
    def update_sentiment_scores(self):
        """Update sentiment scores for unprocessed news and social media data"""
        session = self.SessionLocal()
        
        try:
            # Update news sentiment scores
            unprocessed_news = session.query(NewsData).filter(
                NewsData.processed == False
            ).limit(100).all()
            
            for news in unprocessed_news:
                text = f"{news.title} {news.content}"
                sentiment_result = self.analyze_sentiment(text)
                
                # Update database
                news.sentiment_score = sentiment_result['ensemble_sentiment']['score']
                news.sentiment_label = sentiment_result['ensemble_sentiment']['label']
                news.confidence_score = sentiment_result['ensemble_sentiment']['confidence']
                news.processed = True
                news.processing_timestamp = datetime.now()
            
            # Update social sentiment scores
            unprocessed_social = session.query(SocialSentiment).filter(
                SocialSentiment.sentiment_score.is_(None)
            ).limit(100).all()
            
            for social in unprocessed_social:
                sentiment_result = self.analyze_sentiment(social.content)
                
                # Update database
                social.sentiment_score = sentiment_result['ensemble_sentiment']['score']
                social.sentiment_label = sentiment_result['ensemble_sentiment']['label']
                social.confidence_score = sentiment_result['ensemble_sentiment']['confidence']
            
            session.commit()
            
            self.logger.info(f"Updated sentiment scores for {len(unprocessed_news)} news and {len(unprocessed_social)} social posts")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error updating sentiment scores: {e}")
        
        finally:
            session.close()

