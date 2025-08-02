"""
Zerodha KiteConnect API Client for Market Data Collection
"""
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
import redis
import json
from dotenv import load_dotenv

load_dotenv()

class ZerodhaClient:
    """Zerodha API client for market data and trading operations"""
    
    CREDENTIALS_KEY = "zerodha_credentials"
    
    def __init__(self):
        # Redis for caching and credentials
        redis_url = os.getenv("REDIS_URL", "redis://ai-trading-redis-master.ai-trading.svc.cluster.local:6379/0")
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(redis_url)
            # Ping Redis to verify connection
            self.redis_client.ping()
            self.logger.info(f"Successfully connected to Redis at {redis_url}")
        except Exception as e:
            self.logger.error(f"Redis connection failed at {redis_url}: {e}")
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Load credentials dynamically
        self._load_credentials()
        
        # Initialize KiteConnect if credentials are available
        if all([self.api_key, self.api_secret, self.access_token]):
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
        else:
            self.kite = None
            self.logger.warning("Zerodha credentials not available. Client initialized without connection.")
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 4 requests per second max
        
        # Instrument cache
        self.instruments = {}
        self.nifty_instruments = {}
        self.banknifty_instruments = {}
    
    def _load_credentials(self):
        """Load credentials from Redis or environment variables"""
        self.api_key = None
        self.api_secret = None
        self.access_token = None
        
        # Try to load from Redis first
        if self.redis_client:
            try:
                cached_creds = self.redis_client.get(self.CREDENTIALS_KEY)
                if cached_creds:
                    credentials = json.loads(cached_creds)
                    self.api_key = credentials.get('api_key')
                    self.api_secret = credentials.get('api_secret')
                    self.access_token = credentials.get('access_token')
                    self.logger.info("Loaded credentials from Redis")
                    return
            except Exception as e:
                self.logger.error(f"Failed to load credentials from Redis: {e}")
        
        # Fallback to environment variables
        self.api_key = os.getenv('ZERODHA_API_KEY')
        self.api_secret = os.getenv('ZERODHA_API_SECRET')
        self.access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
        
        if all([self.api_key, self.api_secret, self.access_token]):
            self.logger.info("Loaded credentials from environment variables")
        else:
            self.logger.warning("No valid Zerodha credentials found")
    
    def refresh_credentials(self):
        """Refresh credentials and reinitialize KiteConnect"""
        self._load_credentials()
        
        if all([self.api_key, self.api_secret, self.access_token]):
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            self.logger.info("Credentials refreshed and KiteConnect reinitialized")
            return True
        else:
            self.kite = None
            self.logger.warning("Failed to refresh credentials")
            return False
    
    def is_connected(self):
        """Check if client is properly connected"""
        return self.kite is not None and all([self.api_key, self.api_secret, self.access_token])
    
    def _ensure_connection(self):
        """Ensure connection is available before making API calls"""
        if not self.is_connected():
            # Try to refresh credentials
            if not self.refresh_credentials():
                raise ValueError("Zerodha API credentials not available. Please configure them first.")
        
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, func, *args, **kwargs):
        """Make API request with rate limiting and error handling"""
        self._ensure_connection()  # Ensure connection before making request
        self._rate_limit()
        
        try:
            return func(*args, **kwargs)
        except KiteException as e:
            self.logger.error(f"Kite API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in API request: {e}")
            raise
    
    def initialize_instruments(self):
        """Load and cache instrument data"""
        try:
            # Check cache first
            cached_instruments = self.redis_client.get('instruments_cache')
            if cached_instruments:
                self.instruments = json.loads(cached_instruments)
                self.logger.info("Loaded instruments from cache")
                return
            
            # Fetch from API
            instruments = self._make_request(self.kite.instruments)
            
            # Process instruments
            for instrument in instruments:
                token = instrument['instrument_token']
                self.instruments[token] = instrument
                
                # Filter Nifty and Bank Nifty options
                if 'NIFTY' in instrument['name'] and instrument['instrument_type'] in ['CE', 'PE']:
                    if instrument['name'].startswith('NIFTY') and not instrument['name'].startswith('NIFTYBANK'):
                        self.nifty_instruments[token] = instrument
                    elif instrument['name'].startswith('NIFTYBANK'):
                        self.banknifty_instruments[token] = instrument
            
            # Cache for 24 hours
            self.redis_client.setex(
                'instruments_cache', 
                86400, 
                json.dumps(self.instruments)
            )
            
            self.logger.info(f"Loaded {len(self.instruments)} instruments")
            self.logger.info(f"Nifty options: {len(self.nifty_instruments)}")
            self.logger.info(f"Bank Nifty options: {len(self.banknifty_instruments)}")
            
        except Exception as e:
            self.logger.error(f"Error loading instruments: {e}")
            raise
    
    def get_quote(self, instruments: List[str]) -> Dict[str, Any]:
        """Get real-time quotes for instruments"""
        try:
            quotes = self._make_request(self.kite.quote, instruments)
            return quotes
        except Exception as e:
            self.logger.error(f"Error fetching quotes: {e}")
            return {}
    
    def get_ohlc(self, instruments: List[str]) -> Dict[str, Any]:
        """Get OHLC data for instruments"""
        try:
            ohlc_data = self._make_request(self.kite.ohlc, instruments)
            return ohlc_data
        except Exception as e:
            self.logger.error(f"Error fetching OHLC data: {e}")
            return {}
    
    def get_historical_data(self, 
                          instrument_token: int,
                          from_date: datetime,
                          to_date: datetime,
                          interval: str = "minute") -> pd.DataFrame:
        """Get historical data for an instrument"""
        try:
            data = self._make_request(
                self.kite.historical_data,
                instrument_token,
                from_date,
                to_date,
                interval
            )
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_nifty_options_chain(self, expiry_date: Optional[str] = None) -> List[Dict]:
        """Get Nifty options chain data"""
        try:
            if not self.nifty_instruments:
                self.initialize_instruments()
            
            options_data = []
            
            # Filter by expiry if specified
            instruments_to_fetch = self.nifty_instruments
            if expiry_date:
                instruments_to_fetch = {
                    token: inst for token, inst in self.nifty_instruments.items()
                    if inst['expiry'].strftime('%Y-%m-%d') == expiry_date
                }
            
            # Get quotes in batches (API limit)
            batch_size = 200
            tokens = list(instruments_to_fetch.keys())
            
            for i in range(0, len(tokens), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                batch_instruments = [f"NSE:{instruments_to_fetch[token]['tradingsymbol']}" 
                                   for token in batch_tokens]
                
                quotes = self.get_quote(batch_instruments)
                
                for token in batch_tokens:
                    instrument = instruments_to_fetch[token]
                    symbol = f"NSE:{instrument['tradingsymbol']}"
                    
                    if symbol in quotes:
                        quote_data = quotes[symbol]
                        
                        option_data = {
                            'instrument_token': token,
                            'symbol': instrument['tradingsymbol'],
                            'strike_price': instrument['strike'],
                            'expiry_date': instrument['expiry'],
                            'option_type': instrument['instrument_type'],
                            'last_price': quote_data.get('last_price', 0),
                            'volume': quote_data.get('volume', 0),
                            'open_interest': quote_data.get('oi', 0),
                            'bid_price': quote_data.get('depth', {}).get('buy', [{}])[0].get('price'),
                            'ask_price': quote_data.get('depth', {}).get('sell', [{}])[0].get('price'),
                            'change': quote_data.get('net_change', 0),
                            'change_percent': quote_data.get('change', 0)
                        }
                        
                        options_data.append(option_data)
                
                # Small delay between batches
                time.sleep(0.1)
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Nifty options chain: {e}")
            return []
    
    def get_banknifty_options_chain(self, expiry_date: Optional[str] = None) -> List[Dict]:
        """Get Bank Nifty options chain data"""
        try:
            if not self.banknifty_instruments:
                self.initialize_instruments()
            
            options_data = []
            
            # Filter by expiry if specified
            instruments_to_fetch = self.banknifty_instruments
            if expiry_date:
                instruments_to_fetch = {
                    token: inst for token, inst in self.banknifty_instruments.items()
                    if inst['expiry'].strftime('%Y-%m-%d') == expiry_date
                }
            
            # Get quotes in batches
            batch_size = 200
            tokens = list(instruments_to_fetch.keys())
            
            for i in range(0, len(tokens), batch_size):
                batch_tokens = tokens[i:i + batch_size]
                batch_instruments = [f"NSE:{instruments_to_fetch[token]['tradingsymbol']}" 
                                   for token in batch_tokens]
                
                quotes = self.get_quote(batch_instruments)
                
                for token in batch_tokens:
                    instrument = instruments_to_fetch[token]
                    symbol = f"NSE:{instrument['tradingsymbol']}"
                    
                    if symbol in quotes:
                        quote_data = quotes[symbol]
                        
                        option_data = {
                            'instrument_token': token,
                            'symbol': instrument['tradingsymbol'],
                            'strike_price': instrument['strike'],
                            'expiry_date': instrument['expiry'],
                            'option_type': instrument['instrument_type'],
                            'last_price': quote_data.get('last_price', 0),
                            'volume': quote_data.get('volume', 0),
                            'open_interest': quote_data.get('oi', 0),
                            'bid_price': quote_data.get('depth', {}).get('buy', [{}])[0].get('price'),
                            'ask_price': quote_data.get('depth', {}).get('sell', [{}])[0].get('price'),
                            'change': quote_data.get('net_change', 0),
                            'change_percent': quote_data.get('change', 0)
                        }
                        
                        options_data.append(option_data)
                
                time.sleep(0.1)
            
            return options_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Bank Nifty options chain: {e}")
            return []
    
    def get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get current price of underlying asset"""
        try:
            if symbol.upper() == 'NIFTY':
                instrument = "NSE:NIFTY 50"
            elif symbol.upper() == 'BANKNIFTY':
                instrument = "NSE:NIFTY BANK"
            else:
                instrument = f"NSE:{symbol}"
            
            quotes = self.get_quote([instrument])
            
            if instrument in quotes:
                return quotes[instrument].get('last_price')
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching underlying price for {symbol}: {e}")
            return None
    
    def place_order(self, 
                   symbol: str,
                   quantity: int,
                   order_type: str,
                   transaction_type: str,
                   price: Optional[float] = None) -> Optional[str]:
        """Place a trading order"""
        try:
            order_params = {
                'tradingsymbol': symbol,
                'exchange': 'NSE',
                'quantity': quantity,
                'order_type': order_type,
                'transaction_type': transaction_type,
                'product': 'MIS',  # Intraday
                'validity': 'DAY'
            }
            
            if price and order_type == 'LIMIT':
                order_params['price'] = price
            
            order_id = self._make_request(self.kite.place_order, **order_params)
            
            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        try:
            positions = self._make_request(self.kite.positions)
            return positions
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return {}
    
    def get_holdings(self) -> List[Dict]:
        """Get current holdings"""
        try:
            holdings = self._make_request(self.kite.holdings)
            return holdings
        except Exception as e:
            self.logger.error(f"Error fetching holdings: {e}")
            return []
    
    def get_margins(self) -> Dict[str, Any]:
        """Get account margins"""
        try:
            margins = self._make_request(self.kite.margins)
            return margins
        except Exception as e:
            self.logger.error(f"Error fetching margins: {e}")
            return {}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self._make_request(self.kite.cancel_order, order_id=order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def modify_order(self, order_id: str, **params) -> bool:
        """Modify an existing order"""
        try:
            self._make_request(self.kite.modify_order, order_id=order_id, **params)
            self.logger.info(f"Order modified: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return False

