"""
Order Management System for AI Trading Agent
"""
import os
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, and_
from dotenv import load_dotenv

# Import data collection and models
from src.data_collection.zerodha_client import ZerodhaClient
from src.models.market_data import Portfolio, TradingSignals, RiskMetrics

load_dotenv()

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TransactionType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderManager:
    """Comprehensive order management system"""
    
    def __init__(self):
        # Database setup
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_agent.db')
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Redis for caching and order tracking
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
        
        # Zerodha client for order execution
        self.zerodha_client = ZerodhaClient()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.pending_orders = {}  # {order_id: order_details}
        self.executed_orders = {}
        
        # Configuration
        self.max_orders_per_minute = 10
        self.max_position_value = float(os.getenv('MAX_POSITION_SIZE', '0.2'))  # 20% of portfolio
        self.min_order_value = 100  # Minimum order value in rupees
        self.max_slippage = 0.02  # 2% maximum slippage
        
        # Order rate limiting
        self.order_timestamps = []
        
        # Initialize system
        self._initialize_order_manager()
    
    def _initialize_order_manager(self):
        """Initialize the order management system"""
        try:
            self.logger.info("Initializing Order Management System...")
            
            # Load pending orders from database/cache
            self._load_pending_orders()
            
            # Start order monitoring
            self._start_order_monitoring()
            
            self.logger.info("Order Management System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Order Manager: {e}")
            raise
    
    def _load_pending_orders(self):
        """Load pending orders from cache"""
        try:
            cached_orders = self.redis_client.get('pending_orders')
            if cached_orders:
                self.pending_orders = json.loads(cached_orders)
                self.logger.info(f"Loaded {len(self.pending_orders)} pending orders")
        except Exception as e:
            self.logger.error(f"Error loading pending orders: {e}")
    
    def _save_pending_orders(self):
        """Save pending orders to cache"""
        try:
            self.redis_client.setex(
                'pending_orders',
                3600,  # 1 hour TTL
                json.dumps(self.pending_orders, default=str)
            )
        except Exception as e:
            self.logger.error(f"Error saving pending orders: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Check if order rate limit is exceeded"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.order_timestamps = [
            ts for ts in self.order_timestamps 
            if current_time - ts < 60
        ]
        
        return len(self.order_timestamps) < self.max_orders_per_minute
    
    def _add_order_timestamp(self):
        """Add current timestamp to order tracking"""
        self.order_timestamps.append(time.time())
    
    def validate_order(self, order_params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate order parameters before execution"""
        try:
            # Check required parameters
            required_params = ['symbol', 'quantity', 'transaction_type']
            for param in required_params:
                if param not in order_params:
                    return False, f"Missing required parameter: {param}"
            
            # Validate symbol
            symbol = order_params['symbol']
            if not symbol or len(symbol) < 3:
                return False, "Invalid symbol"
            
            # Validate quantity
            quantity = order_params['quantity']
            if not isinstance(quantity, int) or quantity <= 0:
                return False, "Invalid quantity"
            
            # Validate transaction type
            transaction_type = order_params['transaction_type']
            if transaction_type not in ['BUY', 'SELL']:
                return False, "Invalid transaction type"
            
            # Check rate limiting
            if not self._check_rate_limit():
                return False, "Order rate limit exceeded"
            
            # Validate order value
            price = order_params.get('price', 0)
            if price > 0:
                order_value = price * quantity
                if order_value < self.min_order_value:
                    return False, f"Order value below minimum: {self.min_order_value}"
            
            # Check position limits for BUY orders
            if transaction_type == 'BUY':
                if not self._check_position_limits(symbol, quantity, price):
                    return False, "Position limits exceeded"
            
            # Check available balance for BUY orders
            if transaction_type == 'BUY':
                if not self._check_available_balance(quantity, price):
                    return False, "Insufficient balance"
            
            # Check existing positions for SELL orders
            if transaction_type == 'SELL':
                if not self._check_sell_position(symbol, quantity):
                    return False, "Insufficient position to sell"
            
            return True, "Order validation successful"
            
        except Exception as e:
            self.logger.error(f"Error validating order: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _check_position_limits(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if order would exceed position limits"""
        try:
            # Get current portfolio value
            margins = self.zerodha_client.get_margins()
            if not margins:
                return False
            
            available_margin = margins.get('equity', {}).get('available', {}).get('cash', 0)
            
            # Calculate order value
            order_value = price * quantity if price > 0 else 0
            
            # Check against maximum position size
            max_order_value = available_margin * self.max_position_value
            
            return order_value <= max_order_value
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return False
    
    def _check_available_balance(self, quantity: int, price: float) -> bool:
        """Check if sufficient balance is available"""
        try:
            margins = self.zerodha_client.get_margins()
            if not margins:
                return False
            
            available_cash = margins.get('equity', {}).get('available', {}).get('cash', 0)
            
            # Calculate required amount (including buffer for charges)
            required_amount = price * quantity * 1.01  # 1% buffer
            
            return available_cash >= required_amount
            
        except Exception as e:
            self.logger.error(f"Error checking available balance: {e}")
            return False
    
    def _check_sell_position(self, symbol: str, quantity: int) -> bool:
        """Check if sufficient position exists for selling"""
        try:
            positions = self.zerodha_client.get_positions()
            if not positions or 'net' not in positions:
                return False
            
            for position in positions['net']:
                if position['tradingsymbol'] == symbol:
                    available_quantity = position['quantity']
                    return available_quantity >= quantity
            
            return False  # Position not found
            
        except Exception as e:
            self.logger.error(f"Error checking sell position: {e}")
            return False
    
    def place_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Place a trading order"""
        try:
            # Generate unique order ID
            order_id = str(uuid.uuid4())
            
            # Validate order
            is_valid, validation_message = self.validate_order(order_params)
            if not is_valid:
                return {
                    'success': False,
                    'order_id': order_id,
                    'message': validation_message,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Add rate limiting timestamp
            self._add_order_timestamp()
            
            # Prepare order for Zerodha API
            zerodha_params = self._prepare_zerodha_order(order_params)
            
            # Place order through Zerodha
            zerodha_order_id = self.zerodha_client.place_order(**zerodha_params)
            
            if zerodha_order_id:
                # Store order details
                order_details = {
                    'order_id': order_id,
                    'zerodha_order_id': zerodha_order_id,
                    'symbol': order_params['symbol'],
                    'quantity': order_params['quantity'],
                    'transaction_type': order_params['transaction_type'],
                    'order_type': order_params.get('order_type', 'MARKET'),
                    'price': order_params.get('price'),
                    'status': OrderStatus.PENDING.value,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                # Add to pending orders
                self.pending_orders[order_id] = order_details
                self._save_pending_orders()
                
                # Log order placement
                self.logger.info(f"Order placed successfully: {order_id} -> {zerodha_order_id}")
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'zerodha_order_id': zerodha_order_id,
                    'message': 'Order placed successfully',
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return {
                    'success': False,
                    'order_id': order_id,
                    'message': 'Failed to place order with broker',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {
                'success': False,
                'order_id': order_id if 'order_id' in locals() else 'unknown',
                'message': f'Order placement error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_zerodha_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare order parameters for Zerodha API"""
        zerodha_params = {
            'tradingsymbol': order_params['symbol'],
            'exchange': 'NSE',
            'quantity': order_params['quantity'],
            'transaction_type': order_params['transaction_type'],
            'order_type': order_params.get('order_type', 'MARKET'),
            'product': 'MIS',  # Intraday
            'validity': 'DAY'
        }
        
        # Add price for limit orders
        if order_params.get('order_type') == 'LIMIT' and 'price' in order_params:
            zerodha_params['price'] = order_params['price']
        
        # Add trigger price for stop loss orders
        if 'trigger_price' in order_params:
            zerodha_params['trigger_price'] = order_params['trigger_price']
        
        return zerodha_params
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a pending order"""
        try:
            if order_id not in self.pending_orders:
                return {
                    'success': False,
                    'message': 'Order not found in pending orders',
                    'timestamp': datetime.now().isoformat()
                }
            
            order_details = self.pending_orders[order_id]
            zerodha_order_id = order_details['zerodha_order_id']
            
            # Cancel order through Zerodha
            success = self.zerodha_client.cancel_order(zerodha_order_id)
            
            if success:
                # Update order status
                order_details['status'] = OrderStatus.CANCELLED.value
                order_details['updated_at'] = datetime.now().isoformat()
                
                # Move to executed orders
                self.executed_orders[order_id] = order_details
                del self.pending_orders[order_id]
                self._save_pending_orders()
                
                self.logger.info(f"Order cancelled successfully: {order_id}")
                
                return {
                    'success': True,
                    'message': 'Order cancelled successfully',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to cancel order with broker',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                'success': False,
                'message': f'Order cancellation error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify a pending order"""
        try:
            if order_id not in self.pending_orders:
                return {
                    'success': False,
                    'message': 'Order not found in pending orders',
                    'timestamp': datetime.now().isoformat()
                }
            
            order_details = self.pending_orders[order_id]
            zerodha_order_id = order_details['zerodha_order_id']
            
            # Prepare modification parameters
            modify_params = {}
            if 'price' in modifications:
                modify_params['price'] = modifications['price']
            if 'quantity' in modifications:
                modify_params['quantity'] = modifications['quantity']
            if 'order_type' in modifications:
                modify_params['order_type'] = modifications['order_type']
            
            # Modify order through Zerodha
            success = self.zerodha_client.modify_order(zerodha_order_id, **modify_params)
            
            if success:
                # Update order details
                for key, value in modifications.items():
                    if key in order_details:
                        order_details[key] = value
                
                order_details['updated_at'] = datetime.now().isoformat()
                self._save_pending_orders()
                
                self.logger.info(f"Order modified successfully: {order_id}")
                
                return {
                    'success': True,
                    'message': 'Order modified successfully',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to modify order with broker',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return {
                'success': False,
                'message': f'Order modification error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current status of an order"""
        try:
            # Check pending orders first
            if order_id in self.pending_orders:
                return {
                    'order_id': order_id,
                    'status': 'found',
                    'details': self.pending_orders[order_id],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check executed orders
            if order_id in self.executed_orders:
                return {
                    'order_id': order_id,
                    'status': 'found',
                    'details': self.executed_orders[order_id],
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'order_id': order_id,
                'status': 'not_found',
                'message': 'Order not found',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {
                'order_id': order_id,
                'status': 'error',
                'message': f'Error retrieving order status: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_all_orders(self, status_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get all orders with optional status filter"""
        try:
            all_orders = {}
            all_orders.update(self.pending_orders)
            all_orders.update(self.executed_orders)
            
            if status_filter:
                filtered_orders = {
                    order_id: details for order_id, details in all_orders.items()
                    if details.get('status') == status_filter
                }
                return {
                    'orders': filtered_orders,
                    'count': len(filtered_orders),
                    'filter': status_filter,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'orders': all_orders,
                'count': len(all_orders),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all orders: {e}")
            return {
                'orders': {},
                'count': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _start_order_monitoring(self):
        """Start monitoring order status updates"""
        # This would typically run in a separate thread
        # For now, it's a placeholder for the monitoring logic
        self.logger.info("Order monitoring started")
    
    def update_order_statuses(self):
        """Update status of all pending orders"""
        try:
            if not self.pending_orders:
                return
            
            # Get order book from Zerodha
            # Note: This is a simplified implementation
            # In practice, you'd need to implement proper order status tracking
            
            updated_count = 0
            
            for order_id, order_details in list(self.pending_orders.items()):
                try:
                    # This is a placeholder - implement actual status checking
                    # zerodha_order_id = order_details['zerodha_order_id']
                    # status = self.zerodha_client.get_order_status(zerodha_order_id)
                    
                    # For now, simulate status updates
                    # In production, this would query the actual order status
                    
                    updated_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error updating status for order {order_id}: {e}")
                    continue
            
            if updated_count > 0:
                self._save_pending_orders()
                self.logger.info(f"Updated status for {updated_count} orders")
                
        except Exception as e:
            self.logger.error(f"Error updating order statuses: {e}")
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics"""
        try:
            all_orders = {}
            all_orders.update(self.pending_orders)
            all_orders.update(self.executed_orders)
            
            if not all_orders:
                return {
                    'total_orders': 0,
                    'message': 'No orders found'
                }
            
            # Calculate statistics
            total_orders = len(all_orders)
            status_counts = {}
            
            for order_details in all_orders.values():
                status = order_details.get('status', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate success rate
            completed_orders = status_counts.get('COMPLETE', 0)
            success_rate = (completed_orders / total_orders) * 100 if total_orders > 0 else 0
            
            return {
                'total_orders': total_orders,
                'status_breakdown': status_counts,
                'success_rate': success_rate,
                'pending_orders': len(self.pending_orders),
                'executed_orders': len(self.executed_orders),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating order statistics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def emergency_cancel_all_orders(self) -> Dict[str, Any]:
        """Emergency function to cancel all pending orders"""
        try:
            self.logger.warning("Emergency cancellation of all pending orders initiated")
            
            cancelled_count = 0
            failed_count = 0
            
            for order_id in list(self.pending_orders.keys()):
                try:
                    result = self.cancel_order(order_id)
                    if result['success']:
                        cancelled_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order_id}: {e}")
                    failed_count += 1
            
            return {
                'success': True,
                'cancelled_orders': cancelled_count,
                'failed_cancellations': failed_count,
                'message': f'Emergency cancellation completed: {cancelled_count} cancelled, {failed_count} failed',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in emergency cancel all orders: {e}")
            return {
                'success': False,
                'message': f'Emergency cancellation error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get order management system status"""
        try:
            return {
                'system_status': 'operational',
                'pending_orders_count': len(self.pending_orders),
                'executed_orders_count': len(self.executed_orders),
                'rate_limit_status': {
                    'current_minute_orders': len([
                        ts for ts in self.order_timestamps 
                        if time.time() - ts < 60
                    ]),
                    'max_orders_per_minute': self.max_orders_per_minute
                },
                'configuration': {
                    'max_position_value': self.max_position_value,
                    'min_order_value': self.min_order_value,
                    'max_slippage': self.max_slippage
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

