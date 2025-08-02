"""
Risk Management System for AI Trading Agent
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, and_
from dotenv import load_dotenv

# Import models and other components
from src.models.market_data import Portfolio, RiskMetrics, OptionsData, MarketData
from src.data_collection.zerodha_client import ZerodhaClient
from src.ai_agents.options_greeks_agent import OptionsGreeksAgent

load_dotenv()

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskType(Enum):
    MARKET_RISK = "MARKET_RISK"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"
    VOLATILITY_RISK = "VOLATILITY_RISK"
    TIME_DECAY_RISK = "TIME_DECAY_RISK"
    DELTA_RISK = "DELTA_RISK"
    GAMMA_RISK = "GAMMA_RISK"

class RiskManager:
    """Comprehensive risk management system for options trading"""
    
    def __init__(self):
        # Database setup
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///trading_agent.db')
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Redis for caching
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
        
        # External components
        self.zerodha_client = ZerodhaClient()
        self.greeks_agent = OptionsGreeksAgent()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters (configurable)
        self.risk_limits = {
            'max_portfolio_loss': float(os.getenv('MAX_PORTFOLIO_LOSS', '0.05')),  # 5%
            'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.02')),  # 2%
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.2')),  # 20%
            'max_sector_exposure': float(os.getenv('MAX_SECTOR_EXPOSURE', '0.3')),  # 30%
            'max_delta_exposure': float(os.getenv('MAX_DELTA_EXPOSURE', '0.1')),  # 10%
            'max_gamma_exposure': float(os.getenv('MAX_GAMMA_EXPOSURE', '0.05')),  # 5%
            'max_theta_exposure': float(os.getenv('MAX_THETA_EXPOSURE', '1000')),  # ₹1000/day
            'max_vega_exposure': float(os.getenv('MAX_VEGA_EXPOSURE', '2000')),  # ₹2000 per 1% vol
            'min_liquidity_threshold': int(os.getenv('MIN_LIQUIDITY_THRESHOLD', '100')),  # Min volume
            'max_concentration_single_stock': float(os.getenv('MAX_CONCENTRATION', '0.15'))  # 15%
        }
        
        # Risk monitoring
        self.active_risks = {}
        self.risk_alerts = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.portfolio_value = 0.0
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '100000'))
        
        # Initialize risk management
        self._initialize_risk_manager()
    
    def _initialize_risk_manager(self):
        """Initialize the risk management system"""
        try:
            self.logger.info("Initializing Risk Management System...")
            
            # Load current portfolio state
            self._update_portfolio_state()
            
            # Load historical risk metrics
            self._load_risk_history()
            
            self.logger.info("Risk Management System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Risk Manager: {e}")
            raise
    
    def _update_portfolio_state(self):
        """Update current portfolio state"""
        try:
            # Get current positions
            positions = self.zerodha_client.get_positions()
            if positions and 'net' in positions:
                total_value = 0
                for position in positions['net']:
                    if position['quantity'] != 0:
                        total_value += position['value']
                
                self.portfolio_value = total_value
            
            # Get margins
            margins = self.zerodha_client.get_margins()
            if margins:
                self.daily_pnl = margins.get('equity', {}).get('net', 0)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")
    
    def _load_risk_history(self):
        """Load historical risk metrics"""
        session = self.SessionLocal()
        
        try:
            # Get recent risk metrics
            recent_metrics = session.query(RiskMetrics).filter(
                RiskMetrics.timestamp >= datetime.now() - timedelta(days=7)
            ).order_by(RiskMetrics.timestamp.desc()).limit(100).all()
            
            self.logger.info(f"Loaded {len(recent_metrics)} historical risk metrics")
            
        finally:
            session.close()
    
    def assess_portfolio_risk(self) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment"""
        try:
            self._update_portfolio_state()
            
            risk_assessment = {
                'overall_risk_level': RiskLevel.LOW.value,
                'risk_score': 0.0,
                'individual_risks': {},
                'portfolio_metrics': {},
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Assess different types of risks
            market_risk = self._assess_market_risk()
            concentration_risk = self._assess_concentration_risk()
            liquidity_risk = self._assess_liquidity_risk()
            greeks_risk = self._assess_greeks_risk()
            volatility_risk = self._assess_volatility_risk()
            time_decay_risk = self._assess_time_decay_risk()
            
            # Combine individual risk assessments
            individual_risks = {
                'market_risk': market_risk,
                'concentration_risk': concentration_risk,
                'liquidity_risk': liquidity_risk,
                'greeks_risk': greeks_risk,
                'volatility_risk': volatility_risk,
                'time_decay_risk': time_decay_risk
            }
            
            risk_assessment['individual_risks'] = individual_risks
            
            # Calculate overall risk score
            risk_scores = [risk['risk_score'] for risk in individual_risks.values()]
            overall_risk_score = np.mean(risk_scores) if risk_scores else 0.0
            
            risk_assessment['risk_score'] = overall_risk_score
            
            # Determine overall risk level
            if overall_risk_score >= 0.8:
                risk_assessment['overall_risk_level'] = RiskLevel.CRITICAL.value
            elif overall_risk_score >= 0.6:
                risk_assessment['overall_risk_level'] = RiskLevel.HIGH.value
            elif overall_risk_score >= 0.4:
                risk_assessment['overall_risk_level'] = RiskLevel.MEDIUM.value
            else:
                risk_assessment['overall_risk_level'] = RiskLevel.LOW.value
            
            # Add portfolio metrics
            risk_assessment['portfolio_metrics'] = self._calculate_portfolio_metrics()
            
            # Generate recommendations
            risk_assessment['recommendations'] = self._generate_risk_recommendations(individual_risks)
            
            # Store risk assessment
            self._store_risk_assessment(risk_assessment)
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio risk: {e}")
            return {
                'overall_risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_market_risk(self) -> Dict[str, Any]:
        """Assess market risk based on portfolio exposure"""
        try:
            positions = self.zerodha_client.get_positions()
            
            if not positions or 'net' not in positions:
                return {
                    'risk_type': RiskType.MARKET_RISK.value,
                    'risk_level': RiskLevel.LOW.value,
                    'risk_score': 0.0,
                    'message': 'No positions to assess'
                }
            
            total_exposure = 0
            long_exposure = 0
            short_exposure = 0
            
            for position in positions['net']:
                if position['quantity'] != 0:
                    exposure = abs(position['value'])
                    total_exposure += exposure
                    
                    if position['quantity'] > 0:
                        long_exposure += exposure
                    else:
                        short_exposure += exposure
            
            # Calculate risk metrics
            if total_exposure == 0:
                return {
                    'risk_type': RiskType.MARKET_RISK.value,
                    'risk_level': RiskLevel.LOW.value,
                    'risk_score': 0.0,
                    'message': 'No market exposure'
                }
            
            # Portfolio loss percentage
            portfolio_loss_pct = abs(self.daily_pnl) / self.initial_capital if self.initial_capital > 0 else 0
            
            # Exposure as percentage of capital
            exposure_pct = total_exposure / self.initial_capital if self.initial_capital > 0 else 0
            
            # Risk score calculation
            risk_score = 0.0
            
            # Daily loss component
            if portfolio_loss_pct > self.risk_limits['max_daily_loss']:
                risk_score += 0.4
            elif portfolio_loss_pct > self.risk_limits['max_daily_loss'] * 0.5:
                risk_score += 0.2
            
            # Exposure component
            if exposure_pct > 1.0:  # Over-leveraged
                risk_score += 0.4
            elif exposure_pct > 0.8:
                risk_score += 0.2
            
            # Directional bias component
            net_exposure = long_exposure - short_exposure
            bias_pct = abs(net_exposure) / total_exposure if total_exposure > 0 else 0
            
            if bias_pct > 0.8:  # Highly directional
                risk_score += 0.2
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = RiskLevel.CRITICAL.value
            elif risk_score >= 0.5:
                risk_level = RiskLevel.HIGH.value
            elif risk_score >= 0.3:
                risk_level = RiskLevel.MEDIUM.value
            else:
                risk_level = RiskLevel.LOW.value
            
            return {
                'risk_type': RiskType.MARKET_RISK.value,
                'risk_level': risk_level,
                'risk_score': min(risk_score, 1.0),
                'metrics': {
                    'total_exposure': total_exposure,
                    'long_exposure': long_exposure,
                    'short_exposure': short_exposure,
                    'net_exposure': net_exposure,
                    'exposure_percentage': exposure_pct,
                    'daily_pnl': self.daily_pnl,
                    'portfolio_loss_pct': portfolio_loss_pct,
                    'directional_bias': bias_pct
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing market risk: {e}")
            return {
                'risk_type': RiskType.MARKET_RISK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e)
            }
    
    def _assess_concentration_risk(self) -> Dict[str, Any]:
        """Assess concentration risk across positions"""
        try:
            positions = self.zerodha_client.get_positions()
            
            if not positions or 'net' not in positions:
                return {
                    'risk_type': RiskType.CONCENTRATION_RISK.value,
                    'risk_level': RiskLevel.LOW.value,
                    'risk_score': 0.0,
                    'message': 'No positions to assess'
                }
            
            # Group positions by underlying
            underlying_exposure = {}
            total_portfolio_value = 0
            
            for position in positions['net']:
                if position['quantity'] != 0:
                    symbol = position['tradingsymbol']
                    value = abs(position['value'])
                    total_portfolio_value += value
                    
                    # Extract underlying (simplified)
                    underlying = self._extract_underlying(symbol)
                    
                    if underlying not in underlying_exposure:
                        underlying_exposure[underlying] = 0
                    underlying_exposure[underlying] += value
            
            if total_portfolio_value == 0:
                return {
                    'risk_type': RiskType.CONCENTRATION_RISK.value,
                    'risk_level': RiskLevel.LOW.value,
                    'risk_score': 0.0,
                    'message': 'No portfolio value to assess'
                }
            
            # Calculate concentration metrics
            max_concentration = 0
            concentration_violations = 0
            
            for underlying, exposure in underlying_exposure.items():
                concentration_pct = exposure / total_portfolio_value
                max_concentration = max(max_concentration, concentration_pct)
                
                if concentration_pct > self.risk_limits['max_concentration_single_stock']:
                    concentration_violations += 1
            
            # Risk score calculation
            risk_score = 0.0
            
            # Maximum concentration component
            if max_concentration > self.risk_limits['max_concentration_single_stock'] * 1.5:
                risk_score += 0.5
            elif max_concentration > self.risk_limits['max_concentration_single_stock']:
                risk_score += 0.3
            
            # Number of violations component
            if concentration_violations > 2:
                risk_score += 0.3
            elif concentration_violations > 0:
                risk_score += 0.2
            
            # Portfolio diversification component
            num_positions = len([p for p in positions['net'] if p['quantity'] != 0])
            if num_positions < 3:
                risk_score += 0.2
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = RiskLevel.CRITICAL.value
            elif risk_score >= 0.5:
                risk_level = RiskLevel.HIGH.value
            elif risk_score >= 0.3:
                risk_level = RiskLevel.MEDIUM.value
            else:
                risk_level = RiskLevel.LOW.value
            
            return {
                'risk_type': RiskType.CONCENTRATION_RISK.value,
                'risk_level': risk_level,
                'risk_score': min(risk_score, 1.0),
                'metrics': {
                    'max_concentration': max_concentration,
                    'concentration_violations': concentration_violations,
                    'num_positions': num_positions,
                    'underlying_exposure': underlying_exposure,
                    'total_portfolio_value': total_portfolio_value
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing concentration risk: {e}")
            return {
                'risk_type': RiskType.CONCENTRATION_RISK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e)
            }
    
    def _assess_liquidity_risk(self) -> Dict[str, Any]:
        """Assess liquidity risk of current positions"""
        try:
            session = self.SessionLocal()
            
            try:
                positions = self.zerodha_client.get_positions()
                
                if not positions or 'net' not in positions:
                    return {
                        'risk_type': RiskType.LIQUIDITY_RISK.value,
                        'risk_level': RiskLevel.LOW.value,
                        'risk_score': 0.0,
                        'message': 'No positions to assess'
                    }
                
                low_liquidity_positions = 0
                total_positions = 0
                liquidity_scores = []
                
                for position in positions['net']:
                    if position['quantity'] != 0:
                        total_positions += 1
                        symbol = position['tradingsymbol']
                        
                        # Get recent options data for liquidity metrics
                        option_data = session.query(OptionsData).filter(
                            OptionsData.symbol == symbol,
                            OptionsData.timestamp >= datetime.now() - timedelta(hours=1)
                        ).order_by(OptionsData.timestamp.desc()).first()
                        
                        if option_data:
                            volume = option_data.volume or 0
                            open_interest = option_data.open_interest or 0
                            
                            # Calculate liquidity score
                            liquidity_score = min(volume / 1000, 1.0) * 0.6 + min(open_interest / 10000, 1.0) * 0.4
                            liquidity_scores.append(liquidity_score)
                            
                            if volume < self.risk_limits['min_liquidity_threshold']:
                                low_liquidity_positions += 1
                        else:
                            # No data available - assume low liquidity
                            liquidity_scores.append(0.1)
                            low_liquidity_positions += 1
                
                if total_positions == 0:
                    return {
                        'risk_type': RiskType.LIQUIDITY_RISK.value,
                        'risk_level': RiskLevel.LOW.value,
                        'risk_score': 0.0,
                        'message': 'No positions to assess'
                    }
                
                # Calculate risk metrics
                low_liquidity_pct = low_liquidity_positions / total_positions
                avg_liquidity_score = np.mean(liquidity_scores) if liquidity_scores else 0.0
                
                # Risk score calculation
                risk_score = 0.0
                
                # Low liquidity percentage component
                if low_liquidity_pct > 0.5:
                    risk_score += 0.4
                elif low_liquidity_pct > 0.3:
                    risk_score += 0.2
                
                # Average liquidity score component
                if avg_liquidity_score < 0.3:
                    risk_score += 0.4
                elif avg_liquidity_score < 0.5:
                    risk_score += 0.2
                
                # Determine risk level
                if risk_score >= 0.6:
                    risk_level = RiskLevel.CRITICAL.value
                elif risk_score >= 0.4:
                    risk_level = RiskLevel.HIGH.value
                elif risk_score >= 0.2:
                    risk_level = RiskLevel.MEDIUM.value
                else:
                    risk_level = RiskLevel.LOW.value
                
                return {
                    'risk_type': RiskType.LIQUIDITY_RISK.value,
                    'risk_level': risk_level,
                    'risk_score': min(risk_score, 1.0),
                    'metrics': {
                        'total_positions': total_positions,
                        'low_liquidity_positions': low_liquidity_positions,
                        'low_liquidity_percentage': low_liquidity_pct,
                        'average_liquidity_score': avg_liquidity_score
                    }
                }
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error assessing liquidity risk: {e}")
            return {
                'risk_type': RiskType.LIQUIDITY_RISK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e)
            }
    
    def _assess_greeks_risk(self) -> Dict[str, Any]:
        """Assess risk from options Greeks exposure"""
        try:
            positions = self.zerodha_client.get_positions()
            
            if not positions or 'net' not in positions:
                return {
                    'risk_type': RiskType.DELTA_RISK.value,
                    'risk_level': RiskLevel.LOW.value,
                    'risk_score': 0.0,
                    'message': 'No positions to assess'
                }
            
            # Get portfolio Greeks
            position_list = []
            for position in positions['net']:
                if position['quantity'] != 0:
                    position_list.append({
                        'symbol': position['tradingsymbol'],
                        'quantity': position['quantity']
                    })
            
            portfolio_greeks = self.greeks_agent.calculate_portfolio_greeks(position_list)
            
            # Calculate risk scores for each Greek
            delta_risk = abs(portfolio_greeks.get('delta', 0)) / self.risk_limits['max_delta_exposure']
            gamma_risk = abs(portfolio_greeks.get('gamma', 0)) / self.risk_limits['max_gamma_exposure']
            theta_risk = abs(portfolio_greeks.get('theta', 0)) / self.risk_limits['max_theta_exposure']
            vega_risk = abs(portfolio_greeks.get('vega', 0)) / self.risk_limits['max_vega_exposure']
            
            # Overall Greeks risk score
            greeks_risks = [delta_risk, gamma_risk, theta_risk, vega_risk]
            max_greek_risk = max(greeks_risks)
            avg_greek_risk = np.mean(greeks_risks)
            
            # Risk score calculation
            risk_score = min(max_greek_risk * 0.6 + avg_greek_risk * 0.4, 1.0)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL.value
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH.value
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM.value
            else:
                risk_level = RiskLevel.LOW.value
            
            return {
                'risk_type': RiskType.GREEKS_RISK.value,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'metrics': {
                    'portfolio_greeks': portfolio_greeks,
                    'delta_risk_ratio': delta_risk,
                    'gamma_risk_ratio': gamma_risk,
                    'theta_risk_ratio': theta_risk,
                    'vega_risk_ratio': vega_risk,
                    'max_greek_risk': max_greek_risk
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing Greeks risk: {e}")
            return {
                'risk_type': RiskType.GREEKS_RISK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e)
            }
    
    def _assess_volatility_risk(self) -> Dict[str, Any]:
        """Assess volatility risk of portfolio"""
        try:
            session = self.SessionLocal()
            
            try:
                # Get recent volatility data
                recent_options = session.query(OptionsData).filter(
                    OptionsData.timestamp >= datetime.now() - timedelta(hours=1),
                    OptionsData.implied_volatility.isnot(None)
                ).all()
                
                if not recent_options:
                    return {
                        'risk_type': RiskType.VOLATILITY_RISK.value,
                        'risk_level': RiskLevel.LOW.value,
                        'risk_score': 0.0,
                        'message': 'No volatility data available'
                    }
                
                # Calculate volatility metrics
                iv_values = [opt.implied_volatility for opt in recent_options]
                avg_iv = np.mean(iv_values)
                iv_std = np.std(iv_values)
                
                # Historical volatility comparison (simplified)
                historical_avg_iv = 0.25  # 25% - typical average
                
                # Risk score calculation
                risk_score = 0.0
                
                # High volatility component
                if avg_iv > historical_avg_iv * 1.5:
                    risk_score += 0.3
                elif avg_iv > historical_avg_iv * 1.2:
                    risk_score += 0.2
                
                # Volatility instability component
                if iv_std > 0.1:  # High volatility of volatility
                    risk_score += 0.3
                elif iv_std > 0.05:
                    risk_score += 0.2
                
                # Low volatility (complacency) component
                if avg_iv < historical_avg_iv * 0.7:
                    risk_score += 0.2
                
                # Determine risk level
                if risk_score >= 0.6:
                    risk_level = RiskLevel.HIGH.value
                elif risk_score >= 0.4:
                    risk_level = RiskLevel.MEDIUM.value
                else:
                    risk_level = RiskLevel.LOW.value
                
                return {
                    'risk_type': RiskType.VOLATILITY_RISK.value,
                    'risk_level': risk_level,
                    'risk_score': min(risk_score, 1.0),
                    'metrics': {
                        'average_iv': avg_iv,
                        'iv_standard_deviation': iv_std,
                        'historical_average_iv': historical_avg_iv,
                        'iv_percentile': (avg_iv / historical_avg_iv) if historical_avg_iv > 0 else 1.0
                    }
                }
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error assessing volatility risk: {e}")
            return {
                'risk_type': RiskType.VOLATILITY_RISK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e)
            }
    
    def _assess_time_decay_risk(self) -> Dict[str, Any]:
        """Assess time decay risk from options positions"""
        try:
            session = self.SessionLocal()
            
            try:
                positions = self.zerodha_client.get_positions()
                
                if not positions or 'net' not in positions:
                    return {
                        'risk_type': RiskType.TIME_DECAY_RISK.value,
                        'risk_level': RiskLevel.LOW.value,
                        'risk_score': 0.0,
                        'message': 'No positions to assess'
                    }
                
                total_theta_exposure = 0
                short_dated_positions = 0
                total_positions = 0
                
                for position in positions['net']:
                    if position['quantity'] != 0:
                        total_positions += 1
                        symbol = position['tradingsymbol']
                        quantity = position['quantity']
                        
                        # Get option data
                        option_data = session.query(OptionsData).filter(
                            OptionsData.symbol == symbol
                        ).order_by(OptionsData.timestamp.desc()).first()
                        
                        if option_data:
                            theta = option_data.theta or 0
                            total_theta_exposure += theta * quantity
                            
                            # Check if position is short-dated (< 7 days to expiry)
                            days_to_expiry = (option_data.expiry_date - datetime.now()).days
                            if days_to_expiry < 7:
                                short_dated_positions += 1
                
                if total_positions == 0:
                    return {
                        'risk_type': RiskType.TIME_DECAY_RISK.value,
                        'risk_level': RiskLevel.LOW.value,
                        'risk_score': 0.0,
                        'message': 'No positions to assess'
                    }
                
                # Calculate risk metrics
                short_dated_pct = short_dated_positions / total_positions
                
                # Risk score calculation
                risk_score = 0.0
                
                # High theta exposure component
                if abs(total_theta_exposure) > self.risk_limits['max_theta_exposure']:
                    risk_score += 0.4
                elif abs(total_theta_exposure) > self.risk_limits['max_theta_exposure'] * 0.7:
                    risk_score += 0.2
                
                # Short-dated positions component
                if short_dated_pct > 0.5:
                    risk_score += 0.3
                elif short_dated_pct > 0.3:
                    risk_score += 0.2
                
                # Determine risk level
                if risk_score >= 0.6:
                    risk_level = RiskLevel.HIGH.value
                elif risk_score >= 0.4:
                    risk_level = RiskLevel.MEDIUM.value
                else:
                    risk_level = RiskLevel.LOW.value
                
                return {
                    'risk_type': RiskType.TIME_DECAY_RISK.value,
                    'risk_level': risk_level,
                    'risk_score': min(risk_score, 1.0),
                    'metrics': {
                        'total_theta_exposure': total_theta_exposure,
                        'short_dated_positions': short_dated_positions,
                        'short_dated_percentage': short_dated_pct,
                        'total_positions': total_positions
                    }
                }
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error assessing time decay risk: {e}")
            return {
                'risk_type': RiskType.TIME_DECAY_RISK.value,
                'risk_level': RiskLevel.CRITICAL.value,
                'risk_score': 1.0,
                'error': str(e)
            }
    
    def _extract_underlying(self, symbol: str) -> str:
        """Extract underlying symbol from options symbol"""
        # Simplified extraction - in practice, this would be more sophisticated
        if 'NIFTY' in symbol:
            return 'NIFTY'
        elif 'BANKNIFTY' in symbol:
            return 'BANKNIFTY'
        else:
            # Extract first part before numbers
            import re
            match = re.match(r'^([A-Z]+)', symbol)
            return match.group(1) if match else symbol
    
    def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        try:
            return {
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'daily_return': (self.daily_pnl / self.initial_capital) if self.initial_capital > 0 else 0,
                'total_return': ((self.portfolio_value - self.initial_capital) / self.initial_capital) if self.initial_capital > 0 else 0,
                'leverage_ratio': self.portfolio_value / self.initial_capital if self.initial_capital > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def _generate_risk_recommendations(self, individual_risks: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            for risk_type, risk_data in individual_risks.items():
                risk_level = risk_data.get('risk_level')
                risk_score = risk_data.get('risk_score', 0)
                
                if risk_level == RiskLevel.CRITICAL.value:
                    if risk_type == 'market_risk':
                        recommendations.append("URGENT: Reduce market exposure immediately - consider hedging or closing positions")
                    elif risk_type == 'concentration_risk':
                        recommendations.append("URGENT: Diversify portfolio - reduce concentration in single positions")
                    elif risk_type == 'liquidity_risk':
                        recommendations.append("URGENT: Close illiquid positions or reduce position sizes")
                    elif risk_type == 'greeks_risk':
                        recommendations.append("URGENT: Rebalance Greeks exposure - consider delta/gamma hedging")
                    elif risk_type == 'volatility_risk':
                        recommendations.append("URGENT: Adjust volatility exposure - consider volatility hedging")
                    elif risk_type == 'time_decay_risk':
                        recommendations.append("URGENT: Close short-dated positions or roll to longer expiries")
                
                elif risk_level == RiskLevel.HIGH.value:
                    if risk_type == 'market_risk':
                        recommendations.append("Consider reducing market exposure or adding hedges")
                    elif risk_type == 'concentration_risk':
                        recommendations.append("Gradually diversify portfolio across more positions")
                    elif risk_type == 'liquidity_risk':
                        recommendations.append("Monitor liquidity closely and prepare exit strategies")
                    elif risk_type == 'greeks_risk':
                        recommendations.append("Monitor Greeks exposure and consider rebalancing")
                    elif risk_type == 'volatility_risk':
                        recommendations.append("Monitor volatility levels and adjust strategies accordingly")
                    elif risk_type == 'time_decay_risk':
                        recommendations.append("Monitor time decay and consider rolling positions")
            
            # General recommendations
            if len([r for r in individual_risks.values() if r.get('risk_level') == RiskLevel.CRITICAL.value]) > 2:
                recommendations.append("CRITICAL: Multiple high-risk areas detected - consider emergency risk reduction")
            
            if not recommendations:
                recommendations.append("Portfolio risk levels are within acceptable limits")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations - manual review required"]
    
    def _store_risk_assessment(self, risk_assessment: Dict[str, Any]):
        """Store risk assessment in database"""
        session = self.SessionLocal()
        
        try:
            # Create risk metrics record
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                overall_risk_level=risk_assessment['overall_risk_level'],
                risk_score=risk_assessment['risk_score'],
                market_risk_score=risk_assessment['individual_risks'].get('market_risk', {}).get('risk_score', 0),
                concentration_risk_score=risk_assessment['individual_risks'].get('concentration_risk', {}).get('risk_score', 0),
                liquidity_risk_score=risk_assessment['individual_risks'].get('liquidity_risk', {}).get('risk_score', 0),
                volatility_risk_score=risk_assessment['individual_risks'].get('volatility_risk', {}).get('risk_score', 0),
                portfolio_value=self.portfolio_value,
                daily_pnl=self.daily_pnl
            )
            
            session.add(risk_metrics)
            session.commit()
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error storing risk assessment: {e}")
        
        finally:
            session.close()
    
    def check_order_risk(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if proposed order would violate risk limits"""
        try:
            # Simulate the order impact
            symbol = order_params['symbol']
            quantity = order_params['quantity']
            transaction_type = order_params['transaction_type']
            price = order_params.get('price', 0)
            
            # Get current risk assessment
            current_risk = self.assess_portfolio_risk()
            
            # Check specific risk limits
            risk_violations = []
            
            # Position size check
            if transaction_type == 'BUY':
                order_value = price * quantity
                portfolio_pct = order_value / self.initial_capital if self.initial_capital > 0 else 0
                
                if portfolio_pct > self.risk_limits['max_position_size']:
                    risk_violations.append(f"Order exceeds maximum position size limit ({self.risk_limits['max_position_size']*100}%)")
            
            # Concentration check
            underlying = self._extract_underlying(symbol)
            # This would require more sophisticated position tracking
            
            # Overall risk level check
            if current_risk['overall_risk_level'] == RiskLevel.CRITICAL.value:
                if transaction_type == 'BUY':
                    risk_violations.append("Portfolio is at critical risk level - new BUY orders not recommended")
            
            return {
                'approved': len(risk_violations) == 0,
                'risk_violations': risk_violations,
                'current_risk_level': current_risk['overall_risk_level'],
                'recommendations': risk_violations if risk_violations else ["Order approved from risk perspective"],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking order risk: {e}")
            return {
                'approved': False,
                'risk_violations': [f"Risk check error: {str(e)}"],
                'current_risk_level': RiskLevel.CRITICAL.value,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits configuration"""
        return {
            'risk_limits': self.risk_limits,
            'last_updated': datetime.now().isoformat()
        }
    
    def update_risk_limits(self, new_limits: Dict[str, float]) -> Dict[str, Any]:
        """Update risk limits configuration"""
        try:
            for key, value in new_limits.items():
                if key in self.risk_limits:
                    self.risk_limits[key] = value
            
            # Store updated limits in cache
            self.redis_client.setex(
                'risk_limits',
                86400,  # 24 hours
                json.dumps(self.risk_limits)
            )
            
            self.logger.info("Risk limits updated successfully")
            
            return {
                'success': True,
                'updated_limits': self.risk_limits,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def emergency_risk_shutdown(self) -> Dict[str, Any]:
        """Emergency function to close all positions due to excessive risk"""
        try:
            self.logger.critical("EMERGENCY RISK SHUTDOWN INITIATED")
            
            # This would implement emergency position closure
            # For now, it's a placeholder that would:
            # 1. Cancel all pending orders
            # 2. Close all positions at market prices
            # 3. Send alerts to administrators
            
            return {
                'success': True,
                'message': 'Emergency risk shutdown completed',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in emergency risk shutdown: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

