-- AI Trading Agent Database Initialization Script



-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create market_data table
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(10,2),
    high_price DECIMAL(10,2),
    low_price DECIMAL(10,2),
    close_price DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_market_data_symbol_timestamp (symbol, timestamp)
);

-- Create options_data table
CREATE TABLE IF NOT EXISTS options_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(100) NOT NULL,
    underlying_symbol VARCHAR(50) NOT NULL,
    expiry_date DATE NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    option_type VARCHAR(10) NOT NULL CHECK (option_type IN ('CE', 'PE')),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    ltp DECIMAL(10,2),
    bid DECIMAL(10,2),
    ask DECIMAL(10,2),
    volume BIGINT,
    open_interest BIGINT,
    implied_volatility DECIMAL(8,4),
    delta_value DECIMAL(8,4),
    gamma_value DECIMAL(8,4),
    theta_value DECIMAL(8,4),
    vega_value DECIMAL(8,4),
    rho_value DECIMAL(8,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_options_data_symbol_timestamp (symbol, timestamp),
    INDEX idx_options_data_underlying_expiry (underlying_symbol, expiry_date)
);

-- Create news_data table
CREATE TABLE IF NOT EXISTS news_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    sentiment_score DECIMAL(5,4),
    sentiment_label VARCHAR(20),
    relevance_score DECIMAL(5,4),
    symbols TEXT[], -- Array of related symbols
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_news_data_published_at (published_at),
    INDEX idx_news_data_sentiment (sentiment_score)
);

-- Create social_sentiment table
CREATE TABLE IF NOT EXISTS social_sentiment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    author VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    sentiment_score DECIMAL(5,4),
    sentiment_label VARCHAR(20),
    symbols TEXT[], -- Array of related symbols
    engagement_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_social_sentiment_timestamp (timestamp),
    INDEX idx_social_sentiment_platform (platform)
);

-- Create portfolio_positions table
CREATE TABLE IF NOT EXISTS portfolio_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    average_price DECIMAL(10,2) NOT NULL,
    current_price DECIMAL(10,2),
    unrealized_pnl DECIMAL(12,2),
    realized_pnl DECIMAL(12,2),
    position_type VARCHAR(10) NOT NULL CHECK (position_type IN ('LONG', 'SHORT')),
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    INDEX idx_portfolio_positions_symbol (symbol),
    INDEX idx_portfolio_positions_active (is_active)
);

-- Create trading_orders table
CREATE TABLE IF NOT EXISTS trading_orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(100) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL CHECK (transaction_type IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'SL', 'SL-M')),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2),
    trigger_price DECIMAL(10,2),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    exchange VARCHAR(20),
    product VARCHAR(20),
    validity VARCHAR(20),
    placed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    filled_quantity INTEGER DEFAULT 0,
    average_price DECIMAL(10,2),
    order_tag VARCHAR(100),
    INDEX idx_trading_orders_order_id (order_id),
    INDEX idx_trading_orders_symbol_status (symbol, status),
    INDEX idx_trading_orders_placed_at (placed_at)
);

-- Create trading_signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(100) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    confidence_score DECIMAL(5,4) NOT NULL,
    reasoning TEXT,
    model_version VARCHAR(50),
    features JSONB,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_executed BOOLEAN DEFAULT FALSE,
    execution_details JSONB,
    INDEX idx_trading_signals_symbol_generated (symbol, generated_at),
    INDEX idx_trading_signals_confidence (confidence_score)
);

-- Create risk_metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    portfolio_value DECIMAL(15,2),
    daily_pnl DECIMAL(12,2),
    unrealized_pnl DECIMAL(12,2),
    realized_pnl DECIMAL(12,2),
    max_drawdown DECIMAL(8,4),
    var_95 DECIMAL(12,2), -- Value at Risk 95%
    delta_exposure DECIMAL(12,2),
    gamma_exposure DECIMAL(12,2),
    theta_exposure DECIMAL(12,2),
    vega_exposure DECIMAL(12,2),
    risk_level VARCHAR(20),
    risk_score DECIMAL(5,4),
    INDEX idx_risk_metrics_timestamp (timestamp)
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_win DECIMAL(10,2),
    avg_loss DECIMAL(10,2),
    metrics_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_model_performance_name_date (model_name, evaluation_date)
);

-- Create system_logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) NOT NULL,
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    INDEX idx_system_logs_timestamp (timestamp),
    INDEX idx_system_logs_level (level),
    INDEX idx_system_logs_component (component)
);

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    ip_address INET,
    user_agent TEXT,
    INDEX idx_user_sessions_token (session_token),
    INDEX idx_user_sessions_user_id (user_id)
);

-- Insert sample data for testing
INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume) VALUES
('NIFTY', NOW() - INTERVAL '1 hour', 19800.00, 19850.00, 19780.00, 19847.50, 1000000),
('BANKNIFTY', NOW() - INTERVAL '1 hour', 45200.00, 45250.00, 45100.00, 45123.80, 500000);

INSERT INTO options_data (symbol, underlying_symbol, expiry_date, strike_price, option_type, timestamp, ltp, volume, open_interest) VALUES
('NIFTY24JAN20000CE', 'NIFTY', '2024-01-25', 20000.00, 'CE', NOW(), 125.50, 10000, 50000),
('NIFTY24JAN19800PE', 'NIFTY', '2024-01-25', 19800.00, 'PE', NOW(), 67.80, 8000, 40000),
('BANKNIFTY24JAN45000CE', 'BANKNIFTY', '2024-01-25', 45000.00, 'CE', NOW(), 95.20, 5000, 25000);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp_desc ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_options_data_symbol_timestamp_desc ON options_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_data_published_desc ON news_data (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_orders_placed_desc ON trading_orders (placed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_generated_desc ON trading_signals (generated_at DESC);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;

-- Create a function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete market data older than 30 days
    DELETE FROM market_data WHERE created_at < NOW() - INTERVAL '30 days';
    
    -- Delete news data older than 7 days
    DELETE FROM news_data WHERE created_at < NOW() - INTERVAL '7 days';
    
    -- Delete social sentiment data older than 3 days
    DELETE FROM social_sentiment WHERE created_at < NOW() - INTERVAL '3 days';
    
    -- Delete system logs older than 7 days
    DELETE FROM system_logs WHERE timestamp < NOW() - INTERVAL '7 days';
    
    -- Delete expired user sessions
    DELETE FROM user_sessions WHERE expires_at < NOW() OR is_active = FALSE;
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to run cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

COMMIT;

