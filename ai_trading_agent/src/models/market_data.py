"""
Market Data Models for AI Trading Agent
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
Base = declarative_base()

class MarketData(db.Model):
    """Real-time market data for instruments"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    instrument_token = Column(Integer, nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(10), nullable=False)
    last_price = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    bid_price = Column(Float)
    ask_price = Column(Float)
    bid_quantity = Column(Integer)
    ask_quantity = Column(Integer)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    change = Column(Float)
    change_percent = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_instrument_timestamp', 'instrument_token', 'timestamp'),
    )

class OptionsData(db.Model):
    """Options specific data including Greeks"""
    __tablename__ = 'options_data'
    
    id = Column(Integer, primary_key=True)
    instrument_token = Column(Integer, nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    underlying_symbol = Column(String(50), nullable=False, index=True)
    strike_price = Column(Float, nullable=False)
    expiry_date = Column(DateTime, nullable=False, index=True)
    option_type = Column(String(2), nullable=False)  # CE or PE
    
    # Market Data
    last_price = Column(Float, nullable=False)
    volume = Column(Integer, default=0)
    open_interest = Column(Integer, default=0)
    bid_price = Column(Float)
    ask_price = Column(Float)
    
    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)
    implied_volatility = Column(Float)
    
    # Calculated metrics
    intrinsic_value = Column(Float)
    time_value = Column(Float)
    moneyness = Column(Float)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_underlying_expiry', 'underlying_symbol', 'expiry_date'),
        Index('idx_strike_type', 'strike_price', 'option_type'),
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class NewsData(db.Model):
    """News articles and sentiment analysis"""
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    source = Column(String(100), nullable=False)
    url = Column(String(1000), unique=True)
    author = Column(String(200))
    published_at = Column(DateTime, nullable=False, index=True)
    
    # Sentiment Analysis
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    confidence_score = Column(Float)
    
    # Relevance
    relevance_score = Column(Float)
    mentioned_symbols = Column(Text)  # JSON array of symbols
    
    # Processing status
    processed = Column(Boolean, default=False)
    processing_timestamp = Column(DateTime)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_published_processed', 'published_at', 'processed'),
        Index('idx_sentiment_relevance', 'sentiment_score', 'relevance_score'),
    )

class SocialSentiment(db.Model):
    """Social media sentiment data"""
    __tablename__ = 'social_sentiment'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50), nullable=False)  # twitter, reddit, etc.
    post_id = Column(String(100), unique=True)
    content = Column(Text)
    author = Column(String(100))
    followers_count = Column(Integer)
    likes_count = Column(Integer)
    shares_count = Column(Integer)
    posted_at = Column(DateTime, nullable=False, index=True)
    
    # Sentiment Analysis
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    confidence_score = Column(Float)
    
    # Relevance
    mentioned_symbols = Column(Text)  # JSON array
    relevance_score = Column(Float)
    
    # Influence score based on author metrics
    influence_score = Column(Float)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_platform_posted', 'platform', 'posted_at'),
        Index('idx_sentiment_influence', 'sentiment_score', 'influence_score'),
    )

class TradingSignals(db.Model):
    """Generated trading signals from AI models"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    signal_type = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    
    # Signal details
    entry_price = Column(Float)
    target_price = Column(Float)
    stop_loss = Column(Float)
    quantity = Column(Integer)
    
    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    features_used = Column(Text)  # JSON array
    
    # Market conditions
    market_sentiment = Column(Float)
    volatility = Column(Float)
    volume_profile = Column(Float)
    
    # Execution status
    executed = Column(Boolean, default=False)
    execution_price = Column(Float)
    execution_timestamp = Column(DateTime)
    
    # Performance tracking
    pnl = Column(Float)
    max_profit = Column(Float)
    max_loss = Column(Float)
    closed_at = Column(DateTime)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_symbol_signal_type', 'symbol', 'signal_type'),
        Index('idx_confidence_timestamp', 'confidence', 'timestamp'),
        Index('idx_executed_closed', 'executed', 'closed_at'),
    )

class Portfolio(db.Model):
    """Portfolio positions and performance"""
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False, index=True)
    instrument_token = Column(Integer, nullable=False)
    
    # Position details
    quantity = Column(Integer, nullable=False)
    average_price = Column(Float, nullable=False)
    current_price = Column(Float)
    
    # P&L
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    total_pnl = Column(Float)
    
    # Risk metrics
    position_value = Column(Float)
    portfolio_weight = Column(Float)
    var_1d = Column(Float)  # 1-day Value at Risk
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_symbol_opened', 'symbol', 'opened_at'),
        Index('idx_portfolio_weight', 'portfolio_weight'),
    )

class RiskMetrics(db.Model):
    """Portfolio risk metrics and limits"""
    __tablename__ = 'risk_metrics'
    
    id = Column(Integer, primary_key=True)
    
    # Portfolio metrics
    total_value = Column(Float, nullable=False)
    total_pnl = Column(Float)
    daily_pnl = Column(Float)
    max_drawdown = Column(Float)
    
    # Risk measures
    var_1d = Column(Float)  # 1-day VaR
    var_5d = Column(Float)  # 5-day VaR
    expected_shortfall = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    
    # Exposure metrics
    gross_exposure = Column(Float)
    net_exposure = Column(Float)
    leverage = Column(Float)
    
    # Concentration risk
    max_position_weight = Column(Float)
    sector_concentration = Column(Text)  # JSON
    
    # Limit utilization
    daily_loss_limit = Column(Float)
    daily_loss_used = Column(Float)
    position_limit = Column(Float)
    position_limit_used = Column(Float)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_timestamp_desc', 'timestamp'),
    )

class ModelPerformance(db.Model):
    """AI model performance tracking"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Trading performance
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    
    # Financial metrics
    total_pnl = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Evaluation period
    evaluation_start = Column(DateTime, nullable=False)
    evaluation_end = Column(DateTime, nullable=False)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_model_version', 'model_name', 'model_version'),
        Index('idx_evaluation_period', 'evaluation_start', 'evaluation_end'),
    )

