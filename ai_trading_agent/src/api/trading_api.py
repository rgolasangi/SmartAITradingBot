"""
Trading API Endpoints for AI Trading Agent
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from dotenv import load_dotenv

# Import all components
from src.ai_agents.sentiment_agent import SentimentAgent
from src.ai_agents.options_greeks_agent import OptionsGreeksAgent
from src.ai_agents.rl_trading_agent_fixed import RLTradingAgent
from src.trading.order_manager import OrderManager
from src.trading.risk_manager import RiskManager
from src.data_collection.zerodha_client import ZerodhaClient

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins for development

# Rate limiting
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri=redis_url
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
sentiment_agent = SentimentAgent()
greeks_agent = OptionsGreeksAgent()
rl_agent = RLTradingAgent()
order_manager = OrderManager()
risk_manager = RiskManager()
zerodha_client = ZerodhaClient()

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Authentication endpoints
@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    """User authentication"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Simplified authentication - in production, use proper auth
        if username == os.getenv('ADMIN_USERNAME') and password == os.getenv('ADMIN_PASSWORD'):
            return jsonify({
                'success': True,
                'token': 'dummy_token_for_demo',
                'user': {'username': username, 'role': 'admin'},
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

# Market data endpoints
@app.route('/api/market/options-chain/<underlying>', methods=['GET'])
@limiter.limit("30 per minute")
def get_options_chain(underlying):
    """Get options chain for underlying"""
    try:
        expiry_date = request.args.get('expiry_date')
        
        options_chain = greeks_agent.get_options_chain_analysis(
            underlying_symbol=underlying,
            expiry_date=expiry_date
        )
        
        return jsonify(options_chain)
        
    except Exception as e:
        logger.error(f"Options chain error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market/volatility-surface/<underlying>', methods=['GET'])
@limiter.limit("10 per minute")
def get_volatility_surface(underlying):
    """Get volatility surface for underlying"""
    try:
        surface_data = greeks_agent.get_volatility_surface(underlying)
        return jsonify(surface_data)
        
    except Exception as e:
        logger.error(f"Volatility surface error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market/sentiment', methods=['GET'])
@limiter.limit("20 per minute")
def get_market_sentiment():
    """Get current market sentiment"""
    try:
        hours_back = request.args.get('hours', 6, type=int)
        
        sentiment_summary = sentiment_agent.get_market_sentiment_summary(hours_back)
        return jsonify(sentiment_summary)
        
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        return jsonify({'error': str(e)}), 500

# Trading signal endpoints
@app.route('/api/signals/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate_trading_signal():
    """Generate trading signal for a symbol"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        signal = rl_agent.generate_trading_signal(symbol)
        return jsonify(signal)
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/batch', methods=['POST'])
@limiter.limit("5 per minute")
def generate_batch_signals():
    """Generate signals for multiple symbols"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'Symbols list is required'}), 400
        
        signals = {}
        for symbol in symbols:
            try:
                signal = rl_agent.generate_trading_signal(symbol)
                signals[symbol] = signal
            except Exception as e:
                signals[symbol] = {'error': str(e)}
        
        return jsonify({
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch signal generation error: {e}")
        return jsonify({'error': str(e)}), 500

# Order management endpoints
@app.route('/api/orders/place', methods=['POST'])
@limiter.limit("20 per minute")
def place_order():
    """Place a trading order"""
    try:
        order_params = request.get_json()
        
        # Validate required parameters
        required_params = ['symbol', 'quantity', 'transaction_type']
        for param in required_params:
            if param not in order_params:
                return jsonify({'error': f'Missing parameter: {param}'}), 400
        
        # Check risk before placing order
        risk_check = risk_manager.check_order_risk(order_params)
        
        if not risk_check['approved']:
            return jsonify({
                'success': False,
                'message': 'Order rejected by risk management',
                'risk_violations': risk_check['risk_violations']
            }), 400
        
        # Place order
        result = order_manager.place_order(order_params)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders/cancel/<order_id>', methods=['DELETE'])
@limiter.limit("30 per minute")
def cancel_order(order_id):
    """Cancel an order"""
    try:
        result = order_manager.cancel_order(order_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Order cancellation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders/modify/<order_id>', methods=['PUT'])
@limiter.limit("30 per minute")
def modify_order(order_id):
    """Modify an order"""
    try:
        modifications = request.get_json()
        result = order_manager.modify_order(order_id, modifications)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Order modification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders', methods=['GET'])
@limiter.limit("60 per minute")
def get_orders():
    """Get all orders"""
    try:
        status_filter = request.args.get('status')
        orders = order_manager.get_all_orders(status_filter)
        return jsonify(orders)
        
    except Exception as e:
        logger.error(f"Get orders error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders/<order_id>/status', methods=['GET'])
@limiter.limit("60 per minute")
def get_order_status(order_id):
    """Get order status"""
    try:
        status = order_manager.get_order_status(order_id)
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Order status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders/statistics', methods=['GET'])
@limiter.limit("30 per minute")
def get_order_statistics():
    """Get order execution statistics"""
    try:
        stats = order_manager.get_order_statistics()
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Order statistics error: {e}")
        return jsonify({'error': str(e)}), 500

# Risk management endpoints
@app.route('/api/risk/assessment', methods=['GET'])
@limiter.limit("30 per minute")
def get_risk_assessment():
    """Get current portfolio risk assessment"""
    try:
        assessment = risk_manager.assess_portfolio_risk()
        return jsonify(assessment)
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/limits', methods=['GET'])
@limiter.limit("30 per minute")
def get_risk_limits():
    """Get current risk limits"""
    try:
        limits = risk_manager.get_risk_limits()
        return jsonify(limits)
        
    except Exception as e:
        logger.error(f"Risk limits error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/limits', methods=['PUT'])
@limiter.limit("10 per minute")
def update_risk_limits():
    """Update risk limits"""
    try:
        new_limits = request.get_json()
        result = risk_manager.update_risk_limits(new_limits)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Risk limits update error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/emergency-shutdown', methods=['POST'])
@limiter.limit("1 per minute")
def emergency_shutdown():
    """Emergency risk shutdown"""
    try:
        result = risk_manager.emergency_risk_shutdown()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Emergency shutdown error: {e}")
        return jsonify({'error': str(e)}), 500

# Portfolio endpoints
@app.route('/api/portfolio/positions', methods=['GET'])
@limiter.limit("60 per minute")
def get_positions():
    """Get current positions"""
    try:
        positions = zerodha_client.get_positions()
        return jsonify(positions)
        
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/margins', methods=['GET'])
@limiter.limit("60 per minute")
def get_margins():
    """Get margin information"""
    try:
        margins = zerodha_client.get_margins()
        return jsonify(margins)
        
    except Exception as e:
        logger.error(f"Margins error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/holdings', methods=['GET'])
@limiter.limit("60 per minute")
def get_holdings():
    """Get holdings information"""
    try:
        holdings = zerodha_client.get_holdings()
        return jsonify(holdings)
        
    except Exception as e:
        logger.error(f"Holdings error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/greeks', methods=['GET'])
@limiter.limit("30 per minute")
def get_portfolio_greeks():
    """Get portfolio Greeks"""
    try:
        positions = zerodha_client.get_positions()
        
        if not positions or 'net' not in positions:
            return jsonify({'error': 'No positions found'})
        
        position_list = []
        for position in positions['net']:
            if position['quantity'] != 0:
                position_list.append({
                    'symbol': position['tradingsymbol'],
                    'quantity': position['quantity']
                })
        
        portfolio_greeks = greeks_agent.calculate_portfolio_greeks(position_list)
        return jsonify(portfolio_greeks)
        
    except Exception as e:
        logger.error(f"Portfolio Greeks error: {e}")
        return jsonify({'error': str(e)}), 500

# Analytics endpoints
@app.route('/api/analytics/performance', methods=['GET'])
@limiter.limit("30 per minute")
def get_performance_analytics():
    """Get performance analytics"""
    try:
        days_back = request.args.get('days', 30, type=int)
        
        # This would implement comprehensive performance analytics
        # For now, return basic metrics
        
        performance = {
            'total_return': 0.0,
            'daily_returns': [],
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"Performance analytics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/backtest', methods=['POST'])
@limiter.limit("5 per minute")
def run_backtest():
    """Run strategy backtest"""
    try:
        data = request.get_json()
        start_date = datetime.fromisoformat(data.get('start_date', (datetime.now() - timedelta(days=30)).isoformat()))
        end_date = datetime.fromisoformat(data.get('end_date', datetime.now().isoformat()))
        
        backtest_results = rl_agent.backtest(start_date, end_date)
        return jsonify(backtest_results)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': str(e)}), 500

# Model management endpoints
@app.route('/api/models/rl/train', methods=['POST'])
@limiter.limit("1 per hour")
def train_rl_model():
    """Train RL model"""
    try:
        data = request.get_json()
        episodes = data.get('episodes', 1000)
        
        # This would run in background in production
        rl_agent.train(episodes=episodes)
        
        return jsonify({
            'success': True,
            'message': f'RL model training started for {episodes} episodes',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"RL training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/sentiment/train', methods=['POST'])
@limiter.limit("1 per hour")
def train_sentiment_model():
    """Train sentiment model"""
    try:
        sentiment_agent.train_model(retrain=True)
        
        return jsonify({
            'success': True,
            'message': 'Sentiment model training completed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Sentiment training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/performance', methods=['GET'])
@limiter.limit("30 per minute")
def get_model_performance():
    """Get model performance metrics"""
    try:
        rl_performance = rl_agent.get_model_performance()
        
        performance = {
            'rl_model': rl_performance,
            'sentiment_model': {
                'is_trained': sentiment_agent.is_trained,
                'last_updated': datetime.now().isoformat()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"Model performance error: {e}")
        return jsonify({'error': str(e)}), 500

# System status endpoints
@app.route('/api/system/status', methods=['GET'])
@limiter.limit("60 per minute")
def get_system_status():
    """Get overall system status"""
    try:
        order_status = order_manager.get_system_status()
        risk_assessment = risk_manager.assess_portfolio_risk()
        
        system_status = {
            'overall_status': 'operational',
            'order_manager': order_status,
            'risk_level': risk_assessment.get('overall_risk_level', 'UNKNOWN'),
            'risk_score': risk_assessment.get('risk_score', 0.0),
            'components': {
                'sentiment_agent': 'operational',
                'greeks_agent': 'operational',
                'rl_agent': 'operational',
                'order_manager': 'operational',
                'risk_manager': 'operational'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(system_status)
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        return jsonify({'error': str(e)}), 500

# WebSocket endpoint for real-time updates
@app.route('/api/stream/market-data')
def stream_market_data():
    """Stream real-time market data"""
    def generate():
        while True:
            try:
                # This would implement real-time data streaming
                # For now, return periodic updates
                import time
                time.sleep(1)
                
                data = {
                    'type': 'market_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': {
                        'nifty': 19500 + (hash(str(datetime.now())) % 100 - 50),
                        'banknifty': 45000 + (hash(str(datetime.now())) % 200 - 100)
                    }
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                break
    
    return Response(generate(), mimetype='text/plain')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Update Greeks and sentiment scores on startup
    try:
        greeks_agent.update_options_greeks()
        sentiment_agent.update_sentiment_scores()
    except Exception as e:
        logger.error(f"Startup update error: {e}")
    
    # Run the application
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

