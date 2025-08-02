"""
Options Greeks Calculation Agent for AI Trading System
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Import models
from src.models.market_data import OptionsData, MarketData

load_dotenv()

class OptionsGreeksAgent:
    """Advanced options Greeks calculation and analysis agent"""
    
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
        
        # Risk-free rate (can be updated from RBI repo rate)
        self.risk_free_rate = float(os.getenv('RISK_FREE_RATE', '0.065'))  # 6.5%
        
        # Dividend yield for indices (typically 0 for index options)
        self.dividend_yield = 0.0
        
        # Volatility cache
        self.volatility_cache = {}
        
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Black-Scholes call option price"""
        if T <= 0:
            return max(0, S - K)
        
        if sigma <= 0:
            return max(0, S - K) if S > K else 0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = (S * np.exp(-q * T) * norm.cdf(d1) - 
                     K * np.exp(-r * T) * norm.cdf(d2))
        
        return max(0, call_price)
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Black-Scholes put option price"""
        if T <= 0:
            return max(0, K - S)
        
        if sigma <= 0:
            return max(0, K - S) if K > S else 0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                    S * np.exp(-q * T) * norm.cdf(-d1))
        
        return max(0, put_price)
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str, q: float = 0) -> float:
        """Calculate Delta (price sensitivity to underlying)"""
        if T <= 0:
            if option_type.upper() == 'CE':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        if sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        if option_type.upper() == 'CE':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:  # PE
            delta = -np.exp(-q * T) * norm.cdf(-d1)
        
        return delta
    
    def calculate_gamma(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Gamma (rate of change of Delta)"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        
        return gamma
    
    def calculate_theta(self, S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str, q: float = 0) -> float:
        """Calculate Theta (time decay)"""
        if T <= 0:
            return 0.0
        
        if sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.upper() == 'CE':
            theta = ((-S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2) +
                    q * S * np.exp(-q * T) * norm.cdf(d1))
        else:  # PE
            theta = ((-S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2) -
                    q * S * np.exp(-q * T) * norm.cdf(-d1))
        
        # Convert to per-day theta
        return theta / 365
    
    def calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Vega (sensitivity to volatility)"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Convert to per 1% change in volatility
        return vega / 100
    
    def calculate_rho(self, S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str, q: float = 0) -> float:
        """Calculate Rho (sensitivity to interest rate)"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.upper() == 'CE':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:  # PE
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        # Convert to per 1% change in interest rate
        return rho / 100
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float, T: float, 
                                   r: float, option_type: str, q: float = 0) -> float:
        """Calculate implied volatility using Brent's method"""
        if T <= 0 or market_price <= 0:
            return 0.0
        
        # Intrinsic value
        if option_type.upper() == 'CE':
            intrinsic = max(0, S - K)
        else:
            intrinsic = max(0, K - S)
        
        if market_price <= intrinsic:
            return 0.0
        
        def objective(sigma):
            if option_type.upper() == 'CE':
                theoretical_price = self.black_scholes_call(S, K, T, r, sigma, q)
            else:
                theoretical_price = self.black_scholes_put(S, K, T, r, sigma, q)
            
            return abs(theoretical_price - market_price)
        
        try:
            # Use scipy's minimize_scalar for optimization
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            
            if result.success:
                return result.x
            else:
                return 0.2  # Default volatility if optimization fails
                
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return 0.2
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, 
                           market_price: float, option_type: str, q: float = 0) -> Dict[str, float]:
        """Calculate all Greeks for an option"""
        try:
            # Calculate implied volatility first
            iv = self.calculate_implied_volatility(market_price, S, K, T, r, option_type, q)
            
            # Use implied volatility for Greeks calculation
            sigma = max(iv, 0.01)  # Minimum volatility to avoid division by zero
            
            # Calculate theoretical price
            if option_type.upper() == 'CE':
                theoretical_price = self.black_scholes_call(S, K, T, r, sigma, q)
            else:
                theoretical_price = self.black_scholes_put(S, K, T, r, sigma, q)
            
            # Calculate all Greeks
            delta = self.calculate_delta(S, K, T, r, sigma, option_type, q)
            gamma = self.calculate_gamma(S, K, T, r, sigma, q)
            theta = self.calculate_theta(S, K, T, r, sigma, option_type, q)
            vega = self.calculate_vega(S, K, T, r, sigma, q)
            rho = self.calculate_rho(S, K, T, r, sigma, option_type, q)
            
            return {
                'implied_volatility': iv,
                'theoretical_price': theoretical_price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'intrinsic_value': max(0, S - K) if option_type.upper() == 'CE' else max(0, K - S),
                'time_value': market_price - (max(0, S - K) if option_type.upper() == 'CE' else max(0, K - S)),
                'moneyness': S / K
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return self._empty_greeks_result()
    
    def _empty_greeks_result(self) -> Dict[str, float]:
        """Return empty Greeks result"""
        return {
            'implied_volatility': 0.0,
            'theoretical_price': 0.0,
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'intrinsic_value': 0.0,
            'time_value': 0.0,
            'moneyness': 1.0
        }
    
    def calculate_time_to_expiry(self, expiry_date: datetime) -> float:
        """Calculate time to expiry in years"""
        now = datetime.now()
        
        if expiry_date <= now:
            return 0.0
        
        # Calculate time difference
        time_diff = expiry_date - now
        
        # Convert to years (assuming 365 days per year)
        time_to_expiry = time_diff.total_seconds() / (365 * 24 * 3600)
        
        return max(0.0, time_to_expiry)
    
    def update_options_greeks(self):
        """Update Greeks for all options in the database"""
        session = self.SessionLocal()
        
        try:
            # Get current underlying prices
            nifty_price = self._get_underlying_price('NIFTY')
            banknifty_price = self._get_underlying_price('BANKNIFTY')
            
            if not nifty_price or not banknifty_price:
                self.logger.warning("Unable to get underlying prices")
                return
            
            # Get recent options data
            recent_options = session.query(OptionsData).filter(
                OptionsData.timestamp >= datetime.now() - timedelta(hours=1)
            ).all()
            
            updated_count = 0
            
            for option in recent_options:
                try:
                    # Get underlying price
                    if option.underlying_symbol == 'NIFTY':
                        underlying_price = nifty_price
                    elif option.underlying_symbol == 'BANKNIFTY':
                        underlying_price = banknifty_price
                    else:
                        continue
                    
                    # Calculate time to expiry
                    time_to_expiry = self.calculate_time_to_expiry(option.expiry_date)
                    
                    if time_to_expiry <= 0:
                        continue
                    
                    # Calculate Greeks
                    greeks = self.calculate_all_greeks(
                        S=underlying_price,
                        K=option.strike_price,
                        T=time_to_expiry,
                        r=self.risk_free_rate,
                        market_price=option.last_price,
                        option_type=option.option_type,
                        q=self.dividend_yield
                    )
                    
                    # Update option record
                    option.delta = greeks['delta']
                    option.gamma = greeks['gamma']
                    option.theta = greeks['theta']
                    option.vega = greeks['vega']
                    option.rho = greeks['rho']
                    option.implied_volatility = greeks['implied_volatility']
                    option.intrinsic_value = greeks['intrinsic_value']
                    option.time_value = greeks['time_value']
                    option.moneyness = greeks['moneyness']
                    
                    updated_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error updating Greeks for option {option.symbol}: {e}")
                    continue
            
            session.commit()
            self.logger.info(f"Updated Greeks for {updated_count} options")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error updating options Greeks: {e}")
        
        finally:
            session.close()
    
    def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get current underlying price from cache or database"""
        try:
            # Try Redis cache first
            cache_key = f"{symbol.lower()}_price"
            cached_price = self.redis_client.get(cache_key)
            
            if cached_price:
                return float(cached_price)
            
            # Fallback to database
            session = self.SessionLocal()
            try:
                if symbol == 'NIFTY':
                    market_data = session.query(MarketData).filter(
                        MarketData.symbol == 'NIFTY 50'
                    ).order_by(MarketData.timestamp.desc()).first()
                elif symbol == 'BANKNIFTY':
                    market_data = session.query(MarketData).filter(
                        MarketData.symbol == 'NIFTY BANK'
                    ).order_by(MarketData.timestamp.desc()).first()
                else:
                    return None
                
                if market_data:
                    return market_data.last_price
                
                return None
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error getting underlying price for {symbol}: {e}")
            return None
    
    def get_options_chain_analysis(self, underlying_symbol: str, expiry_date: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive options chain analysis"""
        session = self.SessionLocal()
        
        try:
            # Build query
            query = session.query(OptionsData).filter(
                OptionsData.underlying_symbol == underlying_symbol.upper(),
                OptionsData.timestamp >= datetime.now() - timedelta(hours=1)
            )
            
            if expiry_date:
                query = query.filter(OptionsData.expiry_date == expiry_date)
            
            options = query.all()
            
            if not options:
                return {'error': 'No options data found'}
            
            # Separate calls and puts
            calls = [opt for opt in options if opt.option_type == 'CE']
            puts = [opt for opt in options if opt.option_type == 'PE']
            
            # Get underlying price
            underlying_price = self._get_underlying_price(underlying_symbol)
            
            # Analyze options chain
            analysis = {
                'underlying_symbol': underlying_symbol,
                'underlying_price': underlying_price,
                'total_options': len(options),
                'calls_count': len(calls),
                'puts_count': len(puts),
                'expiry_dates': list(set([opt.expiry_date.strftime('%Y-%m-%d') for opt in options])),
                'strike_range': {
                    'min': min([opt.strike_price for opt in options]),
                    'max': max([opt.strike_price for opt in options])
                },
                'call_analysis': self._analyze_option_type(calls, underlying_price),
                'put_analysis': self._analyze_option_type(puts, underlying_price),
                'volatility_analysis': self._analyze_volatility(options),
                'greeks_analysis': self._analyze_greeks(options),
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        finally:
            session.close()
    
    def _analyze_option_type(self, options: List[OptionsData], underlying_price: float) -> Dict[str, Any]:
        """Analyze specific option type (calls or puts)"""
        if not options:
            return {}
        
        # Calculate metrics
        total_volume = sum([opt.volume for opt in options])
        total_oi = sum([opt.open_interest for opt in options])
        
        # Find ATM, ITM, OTM options
        atm_options = []
        itm_options = []
        otm_options = []
        
        for opt in options:
            if opt.option_type == 'CE':
                if abs(opt.strike_price - underlying_price) <= underlying_price * 0.02:  # Within 2%
                    atm_options.append(opt)
                elif opt.strike_price < underlying_price:
                    itm_options.append(opt)
                else:
                    otm_options.append(opt)
            else:  # PE
                if abs(opt.strike_price - underlying_price) <= underlying_price * 0.02:
                    atm_options.append(opt)
                elif opt.strike_price > underlying_price:
                    itm_options.append(opt)
                else:
                    otm_options.append(opt)
        
        # Calculate weighted average IV
        iv_values = [opt.implied_volatility for opt in options if opt.implied_volatility]
        volumes = [opt.volume for opt in options if opt.implied_volatility and opt.volume > 0]
        
        if iv_values and volumes:
            weighted_iv = np.average(iv_values, weights=volumes)
        else:
            weighted_iv = np.mean(iv_values) if iv_values else 0.0
        
        return {
            'total_volume': total_volume,
            'total_open_interest': total_oi,
            'atm_count': len(atm_options),
            'itm_count': len(itm_options),
            'otm_count': len(otm_options),
            'weighted_iv': weighted_iv,
            'max_volume_strike': max(options, key=lambda x: x.volume).strike_price if options else 0,
            'max_oi_strike': max(options, key=lambda x: x.open_interest).strike_price if options else 0,
            'price_range': {
                'min': min([opt.last_price for opt in options]),
                'max': max([opt.last_price for opt in options])
            }
        }
    
    def _analyze_volatility(self, options: List[OptionsData]) -> Dict[str, Any]:
        """Analyze implied volatility patterns"""
        iv_data = [(opt.strike_price, opt.implied_volatility, opt.moneyness) 
                  for opt in options if opt.implied_volatility and opt.moneyness]
        
        if not iv_data:
            return {}
        
        strikes, ivs, moneyness = zip(*iv_data)
        
        return {
            'iv_range': {
                'min': min(ivs),
                'max': max(ivs),
                'mean': np.mean(ivs),
                'std': np.std(ivs)
            },
            'volatility_smile': {
                'atm_iv': np.mean([iv for strike, iv, money in iv_data if 0.95 <= money <= 1.05]),
                'otm_call_iv': np.mean([iv for strike, iv, money in iv_data if money > 1.05]),
                'otm_put_iv': np.mean([iv for strike, iv, money in iv_data if money < 0.95])
            },
            'iv_percentiles': {
                '25th': np.percentile(ivs, 25),
                '50th': np.percentile(ivs, 50),
                '75th': np.percentile(ivs, 75)
            }
        }
    
    def _analyze_greeks(self, options: List[OptionsData]) -> Dict[str, Any]:
        """Analyze Greeks patterns"""
        greeks_data = {
            'delta': [opt.delta for opt in options if opt.delta is not None],
            'gamma': [opt.gamma for opt in options if opt.gamma is not None],
            'theta': [opt.theta for opt in options if opt.theta is not None],
            'vega': [opt.vega for opt in options if opt.vega is not None]
        }
        
        analysis = {}
        
        for greek, values in greeks_data.items():
            if values:
                analysis[greek] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'total': sum(values)  # Portfolio Greek
                }
        
        return analysis
    
    def calculate_portfolio_greeks(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        try:
            portfolio_greeks = {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
            
            session = self.SessionLocal()
            
            try:
                for position in positions:
                    symbol = position.get('symbol')
                    quantity = position.get('quantity', 0)
                    
                    if quantity == 0:
                        continue
                    
                    # Get latest option data
                    option_data = session.query(OptionsData).filter(
                        OptionsData.symbol == symbol
                    ).order_by(OptionsData.timestamp.desc()).first()
                    
                    if option_data:
                        # Add to portfolio Greeks (weighted by quantity)
                        portfolio_greeks['delta'] += (option_data.delta or 0) * quantity
                        portfolio_greeks['gamma'] += (option_data.gamma or 0) * quantity
                        portfolio_greeks['theta'] += (option_data.theta or 0) * quantity
                        portfolio_greeks['vega'] += (option_data.vega or 0) * quantity
                        portfolio_greeks['rho'] += (option_data.rho or 0) * quantity
                
                return portfolio_greeks
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {e}")
            return {greek: 0.0 for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']}
    
    def get_volatility_surface(self, underlying_symbol: str) -> Dict[str, Any]:
        """Generate volatility surface data"""
        session = self.SessionLocal()
        
        try:
            # Get options data for multiple expiries
            options = session.query(OptionsData).filter(
                OptionsData.underlying_symbol == underlying_symbol.upper(),
                OptionsData.timestamp >= datetime.now() - timedelta(hours=1),
                OptionsData.implied_volatility.isnot(None)
            ).all()
            
            if not options:
                return {'error': 'No volatility data found'}
            
            # Group by expiry and strike
            surface_data = {}
            
            for option in options:
                expiry_str = option.expiry_date.strftime('%Y-%m-%d')
                
                if expiry_str not in surface_data:
                    surface_data[expiry_str] = {}
                
                strike = option.strike_price
                if strike not in surface_data[expiry_str]:
                    surface_data[expiry_str][strike] = []
                
                surface_data[expiry_str][strike].append({
                    'iv': option.implied_volatility,
                    'option_type': option.option_type,
                    'moneyness': option.moneyness,
                    'time_to_expiry': self.calculate_time_to_expiry(option.expiry_date)
                })
            
            # Calculate average IV for each strike/expiry combination
            processed_surface = {}
            for expiry, strikes in surface_data.items():
                processed_surface[expiry] = {}
                for strike, data_points in strikes.items():
                    avg_iv = np.mean([dp['iv'] for dp in data_points])
                    processed_surface[expiry][strike] = {
                        'implied_volatility': avg_iv,
                        'time_to_expiry': data_points[0]['time_to_expiry'],
                        'moneyness': data_points[0]['moneyness']
                    }
            
            return {
                'underlying_symbol': underlying_symbol,
                'surface_data': processed_surface,
                'timestamp': datetime.now().isoformat()
            }
            
        finally:
            session.close()

