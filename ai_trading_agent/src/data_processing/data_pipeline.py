"""
Data Processing and Ingestion Pipeline for AI Trading Agent
"""
import os
import logging
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Import our data collection modules
from src.data_collection.zerodha_client import ZerodhaClient
from src.data_collection.news_collector import NewsCollector
from src.data_collection.sentiment_collector import SentimentCollector
from src.models.market_data import (
    MarketData, OptionsData, NewsData, SocialSentiment,
    TradingSignals, Portfolio, RiskMetrics, db
)

load_dotenv()

class DataPipeline:
    """Main data processing and ingestion pipeline"""
    
    def __init__(self):
        # Database setup
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_agent.db')
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Redis for caching and coordination
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
        
        # Data collectors
        self.zerodha_client = ZerodhaClient()
        self.news_collector = NewsCollector()
        self.sentiment_collector = SentimentCollector()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.market_hours = {
            'start': '09:15',
            'end': '15:30'
        }
        
        # Data collection intervals (in seconds)
        self.intervals = {
            'market_data': 30,      # 30 seconds during market hours
            'options_data': 60,     # 1 minute
            'news_data': 300,       # 5 minutes
            'sentiment_data': 600,  # 10 minutes
            'risk_metrics': 300     # 5 minutes
        }
        
        # Initialize instruments
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the data pipeline system"""
        try:
            self.logger.info("Initializing data pipeline system...")
            
            # Initialize Zerodha instruments
            self.zerodha_client.initialize_instruments()
            
            # Create database tables
            db.metadata.create_all(bind=self.engine)
            
            self.logger.info("Data pipeline system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours
        market_start = now.replace(
            hour=int(self.market_hours['start'].split(':')[0]),
            minute=int(self.market_hours['start'].split(':')[1]),
            second=0,
            microsecond=0
        )
        
        market_end = now.replace(
            hour=int(self.market_hours['end'].split(':')[0]),
            minute=int(self.market_hours['end'].split(':')[1]),
            second=0,
            microsecond=0
        )
        
        return market_start <= now <= market_end
    
    def collect_market_data(self):
        """Collect real-time market data"""
        try:
            self.logger.info("Collecting market data...")
            
            # Get Nifty and Bank Nifty underlying prices
            nifty_price = self.zerodha_client.get_underlying_price('NIFTY')
            banknifty_price = self.zerodha_client.get_underlying_price('BANKNIFTY')
            
            session = self.SessionLocal()
            
            try:
                # Store underlying prices
                if nifty_price:
                    nifty_data = MarketData(
                        instrument_token=256265,  # Nifty 50 token
                        symbol='NIFTY 50',
                        exchange='NSE',
                        last_price=nifty_price,
                        timestamp=datetime.now()
                    )
                    session.add(nifty_data)
                
                if banknifty_price:
                    banknifty_data = MarketData(
                        instrument_token=260105,  # Bank Nifty token
                        symbol='NIFTY BANK',
                        exchange='NSE',
                        last_price=banknifty_price,
                        timestamp=datetime.now()
                    )
                    session.add(banknifty_data)
                
                session.commit()
                
                # Cache current prices for other modules
                self.redis_client.setex('nifty_price', 300, str(nifty_price))
                self.redis_client.setex('banknifty_price', 300, str(banknifty_price))
                
                self.logger.info(f"Market data collected - Nifty: {nifty_price}, Bank Nifty: {banknifty_price}")
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
    
    def collect_options_data(self):
        """Collect options chain data"""
        try:
            self.logger.info("Collecting options data...")
            
            session = self.SessionLocal()
            
            try:
                # Collect Nifty options
                nifty_options = self.zerodha_client.get_nifty_options_chain()
                
                for option in nifty_options:
                    # Calculate additional metrics
                    underlying_price = float(self.redis_client.get('nifty_price') or 0)
                    
                    if underlying_price > 0:
                        # Calculate moneyness
                        moneyness = underlying_price / option['strike_price']
                        
                        # Calculate intrinsic value
                        if option['option_type'] == 'CE':
                            intrinsic_value = max(0, underlying_price - option['strike_price'])
                        else:  # PE
                            intrinsic_value = max(0, option['strike_price'] - underlying_price)
                        
                        # Time value
                        time_value = option['last_price'] - intrinsic_value
                        
                        option_data = OptionsData(
                            instrument_token=option['instrument_token'],
                            symbol=option['symbol'],
                            underlying_symbol='NIFTY',
                            strike_price=option['strike_price'],
                            expiry_date=option['expiry_date'],
                            option_type=option['option_type'],
                            last_price=option['last_price'],
                            volume=option['volume'],
                            open_interest=option['open_interest'],
                            bid_price=option['bid_price'],
                            ask_price=option['ask_price'],
                            intrinsic_value=intrinsic_value,
                            time_value=time_value,
                            moneyness=moneyness,
                            timestamp=datetime.now()
                        )
                        
                        session.add(option_data)
                
                # Collect Bank Nifty options
                banknifty_options = self.zerodha_client.get_banknifty_options_chain()
                
                for option in banknifty_options:
                    underlying_price = float(self.redis_client.get('banknifty_price') or 0)
                    
                    if underlying_price > 0:
                        moneyness = underlying_price / option['strike_price']
                        
                        if option['option_type'] == 'CE':
                            intrinsic_value = max(0, underlying_price - option['strike_price'])
                        else:
                            intrinsic_value = max(0, option['strike_price'] - underlying_price)
                        
                        time_value = option['last_price'] - intrinsic_value
                        
                        option_data = OptionsData(
                            instrument_token=option['instrument_token'],
                            symbol=option['symbol'],
                            underlying_symbol='BANKNIFTY',
                            strike_price=option['strike_price'],
                            expiry_date=option['expiry_date'],
                            option_type=option['option_type'],
                            last_price=option['last_price'],
                            volume=option['volume'],
                            open_interest=option['open_interest'],
                            bid_price=option['bid_price'],
                            ask_price=option['ask_price'],
                            intrinsic_value=intrinsic_value,
                            time_value=time_value,
                            moneyness=moneyness,
                            timestamp=datetime.now()
                        )
                        
                        session.add(option_data)
                
                session.commit()
                
                self.logger.info(f"Options data collected - Nifty: {len(nifty_options)}, Bank Nifty: {len(banknifty_options)}")
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error collecting options data: {e}")
    
    def collect_news_data(self):
        """Collect and process news data"""
        try:
            self.logger.info("Collecting news data...")
            
            # Collect news articles
            articles = self.news_collector.collect_all_news(hours_back=6)
            
            session = self.SessionLocal()
            
            try:
                for article in articles:
                    news_data = NewsData(
                        title=article['title'],
                        content=article['content'],
                        source=article['source'],
                        url=article['url'],
                        author=article.get('author'),
                        published_at=article['published_at'],
                        relevance_score=article['relevance_score'],
                        mentioned_symbols=article['mentioned_symbols'],
                        timestamp=datetime.now()
                    )
                    
                    session.add(news_data)
                
                session.commit()
                
                self.logger.info(f"News data collected: {len(articles)} articles")
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error collecting news data: {e}")
    
    def collect_sentiment_data(self):
        """Collect and process social sentiment data"""
        try:
            self.logger.info("Collecting sentiment data...")
            
            # Collect social sentiment
            posts = self.sentiment_collector.collect_all_sentiment(hours_back=6)
            
            session = self.SessionLocal()
            
            try:
                for post in posts:
                    sentiment_data = SocialSentiment(
                        platform=post['platform'],
                        post_id=post['post_id'],
                        content=post['content'],
                        author=post['author'],
                        followers_count=post['followers_count'],
                        likes_count=post['likes_count'],
                        shares_count=post['shares_count'],
                        posted_at=post['posted_at'],
                        relevance_score=post['relevance_score'],
                        influence_score=post['influence_score'],
                        mentioned_symbols=post['mentioned_symbols'],
                        timestamp=datetime.now()
                    )
                    
                    session.add(sentiment_data)
                
                session.commit()
                
                self.logger.info(f"Sentiment data collected: {len(posts)} posts")
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error collecting sentiment data: {e}")
    
    def calculate_risk_metrics(self):
        """Calculate and update portfolio risk metrics"""
        try:
            self.logger.info("Calculating risk metrics...")
            
            session = self.SessionLocal()
            
            try:
                # Get current positions
                positions = self.zerodha_client.get_positions()
                
                if positions and positions.get('net'):
                    total_pnl = sum([pos.get('pnl', 0) for pos in positions['net']])
                    total_value = sum([abs(pos.get('value', 0)) for pos in positions['net']])
                    
                    # Calculate basic risk metrics
                    risk_metrics = RiskMetrics(
                        total_value=total_value,
                        total_pnl=total_pnl,
                        daily_pnl=total_pnl,  # Simplified for now
                        timestamp=datetime.now()
                    )
                    
                    session.add(risk_metrics)
                    session.commit()
                    
                    # Cache metrics for quick access
                    metrics_cache = {
                        'total_value': total_value,
                        'total_pnl': total_pnl,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.redis_client.setex('risk_metrics', 300, json.dumps(metrics_cache))
                    
                    self.logger.info(f"Risk metrics calculated - Total PnL: {total_pnl}, Total Value: {total_value}")
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
    
    def run_data_collection_cycle(self):
        """Run a complete data collection cycle"""
        try:
            self.logger.info("Starting data collection cycle...")
            
            # Use ThreadPoolExecutor for parallel data collection
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                # Market data (always collect)
                futures.append(executor.submit(self.collect_market_data))
                
                # Options data (only during market hours or slightly extended)
                if self.is_market_open() or self._is_extended_hours():
                    futures.append(executor.submit(self.collect_options_data))
                
                # News and sentiment (collect throughout the day)
                futures.append(executor.submit(self.collect_news_data))
                futures.append(executor.submit(self.collect_sentiment_data))
                
                # Risk metrics (only if we have positions)
                if self._has_active_positions():
                    futures.append(executor.submit(self.calculate_risk_metrics))
                
                # Wait for all tasks to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Error in data collection task: {e}")
            
            self.logger.info("Data collection cycle completed")
            
        except Exception as e:
            self.logger.error(f"Error in data collection cycle: {e}")
    
    def _is_extended_hours(self) -> bool:
        """Check if we're in extended trading hours (for options data)"""
        now = datetime.now()
        
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Extended hours: 9:00 AM to 4:00 PM
        extended_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        extended_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return extended_start <= now <= extended_end
    
    def _has_active_positions(self) -> bool:
        """Check if there are active trading positions"""
        try:
            positions = self.zerodha_client.get_positions()
            return bool(positions and positions.get('net'))
        except:
            return False
    
    def schedule_data_collection(self):
        """Schedule regular data collection tasks"""
        self.logger.info("Scheduling data collection tasks...")
        
        # Market data every 30 seconds during market hours
        schedule.every(30).seconds.do(self.collect_market_data)
        
        # Options data every minute during extended hours
        schedule.every().minute.do(self._conditional_options_collection)
        
        # News data every 5 minutes
        schedule.every(5).minutes.do(self.collect_news_data)
        
        # Sentiment data every 10 minutes
        schedule.every(10).minutes.do(self.collect_sentiment_data)
        
        # Risk metrics every 5 minutes (if positions exist)
        schedule.every(5).minutes.do(self._conditional_risk_calculation)
        
        # Full data collection cycle every hour
        schedule.every().hour.do(self.run_data_collection_cycle)
        
        self.logger.info("Data collection tasks scheduled")
    
    def _conditional_options_collection(self):
        """Collect options data only during appropriate hours"""
        if self.is_market_open() or self._is_extended_hours():
            self.collect_options_data()
    
    def _conditional_risk_calculation(self):
        """Calculate risk metrics only if positions exist"""
        if self._has_active_positions():
            self.calculate_risk_metrics()
    
    def start_pipeline(self):
        """Start the data pipeline"""
        self.logger.info("Starting data pipeline...")
        
        # Schedule tasks
        self.schedule_data_collection()
        
        # Run initial data collection
        self.run_data_collection_cycle()
        
        # Keep the pipeline running
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Data pipeline stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in pipeline main loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def stop_pipeline(self):
        """Stop the data pipeline"""
        self.logger.info("Stopping data pipeline...")
        schedule.clear()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        try:
            # Get last collection timestamps from Redis
            status = {
                'pipeline_running': True,
                'market_open': self.is_market_open(),
                'last_market_data': self.redis_client.get('last_market_data_collection'),
                'last_options_data': self.redis_client.get('last_options_data_collection'),
                'last_news_data': self.redis_client.get('last_news_data_collection'),
                'last_sentiment_data': self.redis_client.get('last_sentiment_data_collection'),
                'current_nifty_price': self.redis_client.get('nifty_price'),
                'current_banknifty_price': self.redis_client.get('banknifty_price'),
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start pipeline
    pipeline = DataPipeline()
    pipeline.start_pipeline()

