"""
Zerodha Configuration API endpoints
"""
import os
import json
import logging
from flask import Blueprint, request, jsonify
from datetime import datetime
import redis
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

logger = logging.getLogger(__name__)

# Create blueprint
zerodha_bp = Blueprint('zerodha', __name__, url_prefix='/api/zerodha')

# Redis client for storing credentials
redis_client = None
try:
    redis_url = os.getenv("REDIS_URL", "redis://ai-trading-redis-master.ai-trading.svc.cluster.local:6379/0")
    redis_client = redis.from_url(redis_url)
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")

class ZerodhaCredentialManager:
    """Manages Zerodha API credentials securely"""
    
    CREDENTIALS_KEY = "zerodha_credentials"
    
    @staticmethod
    def save_credentials(api_key, api_secret, access_token):
        """Save credentials to Redis or environment variables"""
        credentials = {
            'api_key': api_key,
            'api_secret': api_secret,
            'access_token': access_token,
            'updated_at': datetime.now().isoformat()
        }
        
        if redis_client:
            try:
                redis_client.setex(
                    ZerodhaCredentialManager.CREDENTIALS_KEY,
                    86400,  # 24 hours
                    json.dumps(credentials)
                )
                logger.info("Credentials saved to Redis")
                return True
            except Exception as e:
                logger.error(f"Failed to save credentials to Redis: {e}")
        
        # Fallback to environment variables (not recommended for production)
        os.environ['ZERODHA_API_KEY'] = api_key
        os.environ['ZERODHA_API_SECRET'] = api_secret
        os.environ['ZERODHA_ACCESS_TOKEN'] = access_token
        logger.info("Credentials saved to environment variables")
        return True
    
    @staticmethod
    def get_credentials():
        """Retrieve credentials from Redis or environment variables"""
        if redis_client:
            try:
                cached_creds = redis_client.get(ZerodhaCredentialManager.CREDENTIALS_KEY)
                if cached_creds:
                    credentials = json.loads(cached_creds)
                    return {
                        'api_key': credentials.get('api_key'),
                        'api_secret': credentials.get('api_secret'),
                        'access_token': credentials.get('access_token')
                    }
            except Exception as e:
                logger.error(f"Failed to retrieve credentials from Redis: {e}")
        
        # Fallback to environment variables
        return {
            'api_key': os.getenv('ZERODHA_API_KEY'),
            'api_secret': os.getenv('ZERODHA_API_SECRET'),
            'access_token': os.getenv('ZERODHA_ACCESS_TOKEN')
        }
    
    @staticmethod
    def test_connection():
        """Test connection with current credentials"""
        credentials = ZerodhaCredentialManager.get_credentials()
        
        if not all([credentials['api_key'], credentials['api_secret'], credentials['access_token']]):
            return False, "Missing credentials"
        
        try:
            kite = KiteConnect(api_key=credentials['api_key'])
            kite.set_access_token(credentials['access_token'])
            
            # Test with a simple API call
            profile = kite.profile()
            return True, f"Connected successfully! User: {profile.get('user_name', 'Unknown')}"
            
        except KiteException as e:
            logger.error(f"Kite API error: {e}")
            return False, f"Kite API error: {str(e)}"
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False, f"Connection failed: {str(e)}"

@zerodha_bp.route('/credentials', methods=['GET'])
def get_credentials():
    """Get current Zerodha credentials (masked for security)"""
    try:
        credentials = ZerodhaCredentialManager.get_credentials()
        
        # Mask sensitive data
        masked_credentials = {
            'api_key': credentials.get('api_key', ''),
            'api_secret': '••••••••••••••••' if credentials.get('api_secret') else '',
            'access_token': '••••••••••••••••••••••••••••••••' if credentials.get('access_token') else ''
        }
        
        return jsonify({
            'success': True,
            'credentials': masked_credentials,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error retrieving credentials: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to retrieve credentials',
            'timestamp': datetime.now().isoformat()
        }), 500

@zerodha_bp.route('/credentials', methods=['POST'])
def save_credentials():
    """Save Zerodha API credentials"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        api_key = data.get('api_key', '').strip()
        api_secret = data.get('api_secret', '').strip()
        access_token = data.get('access_token', '').strip()
        
        if not all([api_key, api_secret, access_token]):
            return jsonify({
                'success': False,
                'message': 'All credentials (api_key, api_secret, access_token) are required'
            }), 400
        
        # Validate format (basic validation)
        if len(api_key) < 10 or len(api_secret) < 10 or len(access_token) < 20:
            return jsonify({
                'success': False,
                'message': 'Invalid credential format'
            }), 400
        
        # Save credentials
        success = ZerodhaCredentialManager.save_credentials(api_key, api_secret, access_token)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Credentials saved successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to save credentials'
            }), 500
            
    except Exception as e:
        logger.error(f"Error saving credentials: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@zerodha_bp.route('/status', methods=['GET'])
def get_connection_status():
    """Get current Zerodha connection status"""
    try:
        credentials = ZerodhaCredentialManager.get_credentials()
        
        if not all([credentials['api_key'], credentials['api_secret'], credentials['access_token']]):
            return jsonify({
                'status': 'disconnected',
                'message': 'Credentials not configured',
                'timestamp': datetime.now().isoformat()
            })
        
        # Test connection without making actual API call for status check
        return jsonify({
            'status': 'configured',
            'message': 'Credentials are configured. Use test-connection to verify.',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to check status',
            'timestamp': datetime.now().isoformat()
        }), 500

@zerodha_bp.route('/test-connection', methods=['POST'])
def test_connection():
    """Test Zerodha API connection"""
    try:
        success, message = ZerodhaCredentialManager.test_connection()
        
        if success:
            return jsonify({
                'success': True,
                'status': 'connected',
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'status': 'error',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        logger.error(f"Error testing connection: {e}")
        return jsonify({
            'success': False,
            'status': 'error',
            'message': 'Connection test failed',
            'timestamp': datetime.now().isoformat()
        }), 500

@zerodha_bp.route('/clear-credentials', methods=['DELETE'])
def clear_credentials():
    """Clear stored Zerodha credentials"""
    try:
        if redis_client:
            try:
                redis_client.delete(ZerodhaCredentialManager.CREDENTIALS_KEY)
                logger.info("Credentials cleared from Redis")
            except Exception as e:
                logger.error(f"Failed to clear credentials from Redis: {e}")
        
        # Clear from environment variables
        for key in ['ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 'ZERODHA_ACCESS_TOKEN']:
            if key in os.environ:
                del os.environ[key]
        
        return jsonify({
            'success': True,
            'message': 'Credentials cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error clearing credentials: {e}")
        return jsonify({
            'success': False,
            'message': 'Failed to clear credentials',
            'timestamp': datetime.now().isoformat()
        }), 500

# Export the blueprint
__all__ = ['zerodha_bp']

