#!/usr/bin/env python3
"""
Simplified API runner for testing
"""
import os
import sys
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import Zerodha configuration blueprint
try:
    from api.zerodha_config import zerodha_bp
    ZERODHA_CONFIG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Zerodha config module not available: {e}")
    ZERODHA_CONFIG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Register Zerodha configuration blueprint if available
if ZERODHA_CONFIG_AVAILABLE:
    app.register_blueprint(zerodha_bp)
    logger.info("Zerodha configuration endpoints registered")

# Mock data for testing
mock_market_data = {
    'nifty': {'price': 19847.5, 'change': 125.3, 'changePercent': 0.63},
    'banknifty': {'price': 45123.8, 'change': -89.2, 'changePercent': -0.20}
}

mock_signals = [
    {'symbol': 'NIFTY24JAN20000CE', 'signal': 'BUY', 'confidence': 0.87, 'reason': 'Strong bullish momentum'},
    {'symbol': 'BANKNIFTY24JAN45500PE', 'signal': 'SELL', 'confidence': 0.92, 'reason': 'Overbought conditions'}
]

# Basic endpoints for testing
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    return jsonify({
        'success': True,
        'token': 'test_token',
        'user': {'username': 'admin', 'role': 'admin'}
    })

@app.route('/api/market/sentiment', methods=['GET'])
def market_sentiment():
    return jsonify({
        'positive': 45,
        'neutral': 35,
        'negative': 20,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market/options-chain/<underlying>', methods=['GET'])
def options_chain(underlying):
    return jsonify({
        'underlying': underlying,
        'options': [
            {'symbol': f'{underlying}24JAN20000CE', 'ltp': 125.5, 'delta': 0.65},
            {'symbol': f'{underlying}24JAN19800PE', 'ltp': 67.8, 'delta': -0.35}
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/market/volatility-surface/<underlying>', methods=['GET'])
def volatility_surface(underlying):
    return jsonify({
        'underlying': underlying,
        'surface_data': [[0.15, 0.18, 0.22], [0.16, 0.19, 0.23]],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/signals/generate', methods=['POST'])
def generate_signal():
    return jsonify({
        'signal': 'BUY',
        'confidence': 0.87,
        'reason': 'Strong bullish momentum with high IV rank',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/signals/batch', methods=['POST'])
def batch_signals():
    return jsonify({
        'signals': {
            'NIFTY24JAN20000CE': {'signal': 'BUY', 'confidence': 0.87},
            'BANKNIFTY24JAN45500PE': {'signal': 'SELL', 'confidence': 0.92}
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/risk/assessment', methods=['GET'])
def risk_assessment():
    return jsonify({
        'overall_risk_level': 'MEDIUM',
        'risk_score': 0.65,
        'portfolio_value': 125000,
        'daily_pnl': 2190,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/risk/limits', methods=['GET'])
def risk_limits():
    return jsonify({
        'max_daily_loss': 10000,
        'max_position_size': 100000,
        'current_exposure': 65000,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/portfolio/positions', methods=['GET'])
def positions():
    return jsonify({
        'net': [
            {'tradingsymbol': 'NIFTY24JAN20000CE', 'quantity': 50, 'ltp': 125.5},
            {'tradingsymbol': 'BANKNIFTY24JAN45500PE', 'quantity': -25, 'ltp': 89.3}
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/portfolio/margins', methods=['GET'])
def margins():
    return jsonify({
        'equity': {'available': 50000, 'utilised': 25000},
        'commodity': {'available': 10000, 'utilised': 5000},
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/portfolio/holdings', methods=['GET'])
def holdings():
    return jsonify({
        'holdings': [],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/portfolio/greeks', methods=['GET'])
def portfolio_greeks():
    return jsonify({
        'delta': 0.15,
        'gamma': 0.08,
        'theta': -450,
        'vega': 1200,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/orders', methods=['GET'])
def get_orders():
    return jsonify({
        'orders': [
            {'order_id': 'ORD001', 'symbol': 'NIFTY24JAN20000CE', 'status': 'COMPLETE'},
            {'order_id': 'ORD002', 'symbol': 'BANKNIFTY24JAN45000CE', 'status': 'PENDING'}
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/orders/statistics', methods=['GET'])
def order_statistics():
    return jsonify({
        'total_orders': 150,
        'completed_orders': 120,
        'pending_orders': 20,
        'cancelled_orders': 10,
        'success_rate': 0.8,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analytics/performance', methods=['GET'])
def performance_analytics():
    return jsonify({
        'total_return': 15.5,
        'sharpe_ratio': 2.45,
        'max_drawdown': -3.2,
        'win_rate': 0.687,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models/performance', methods=['GET'])
def model_performance():
    return jsonify({
        'rl_model': {
            'accuracy': 0.923,
            'sharpe_ratio': 2.45,
            'win_rate': 0.687
        },
        'sentiment_model': {
            'is_trained': True,
            'accuracy': 0.85
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system/status', methods=['GET'])
def system_status():
    return jsonify({
        'overall_status': 'operational',
        'components': {
            'sentiment_agent': 'operational',
            'greeks_agent': 'operational',
            'rl_agent': 'operational',
            'order_manager': 'operational',
            'risk_manager': 'operational'
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting AI Trading Agent API for testing...")
    app.run(host='0.0.0.0', port=5000, debug=False)

