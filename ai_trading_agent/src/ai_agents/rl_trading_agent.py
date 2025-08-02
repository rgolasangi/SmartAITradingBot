"""
Reinforcement Learning Trading Agent for AI Trading System
"""
import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, using simplified RL implementation")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import redis
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Import models and agents
from src.models.market_data import OptionsData, MarketData, NewsData, SocialSentiment, TradingSignals
from src.ai_agents.sentiment_agent import SentimentAgent
from src.ai_agents.options_greeks_agent import OptionsGreeksAgent

load_dotenv()

class TradingEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {}  # {symbol: quantity}
        self.transaction_costs = 0.001  # 0.1% transaction cost
        self.max_position_size = 0.2  # 20% of portfolio per position
        
        # State space dimensions
        self.state_dim = self._calculate_state_dim()
        self.action_dim = 3  # Buy, Sell, Hold
        
        # Performance tracking
        self.portfolio_values = [initial_balance]
        self.trades = []
        
    def _calculate_state_dim(self) -> int:
        """Calculate state space dimensions"""
        # Market features + Greeks + Sentiment + Technical indicators
        return 25  # Adjust based on feature engineering
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {}
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_dim)
        
        row = self.data.iloc[self.current_step]
        
        # Market features
        market_features = [
            row.get('underlying_price', 0),
            row.get('volatility', 0),
            row.get('volume', 0),
            row.get('open_interest', 0)
        ]
        
        # Greeks
        greeks_features = [
            row.get('delta', 0),
            row.get('gamma', 0),
            row.get('theta', 0),
            row.get('vega', 0),
            row.get('rho', 0)
        ]
        
        # Sentiment features
        sentiment_features = [
            row.get('sentiment_score', 0),
            row.get('news_sentiment', 0),
            row.get('social_sentiment', 0)
        ]
        
        # Technical indicators
        technical_features = [
            row.get('rsi', 50),
            row.get('macd', 0),
            row.get('bollinger_upper', 0),
            row.get('bollinger_lower', 0),
            row.get('moving_avg_20', 0),
            row.get('moving_avg_50', 0)
        ]
        
        # Portfolio features
        portfolio_features = [
            self.balance / self.initial_balance,
            len(self.positions),
            sum(self.positions.values()),
            self._calculate_portfolio_value() / self.initial_balance
        ]
        
        # Time features
        time_features = [
            row.get('hour', 12) / 24,
            row.get('day_of_week', 2) / 7,
            row.get('days_to_expiry', 30) / 365
        ]
        
        # Combine all features
        state = np.array(
            market_features + greeks_features + sentiment_features + 
            technical_features + portfolio_features + time_features
        )
        
        # Pad or truncate to match state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return state.astype(np.float32)
    
    def step(self, action: int, symbol: str, quantity: int = 1) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
        
        # Execute action
        reward = self._execute_action(action, symbol, quantity)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Calculate portfolio value
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_values.append(portfolio_value)
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: int, symbol: str, quantity: int) -> float:
        """Execute trading action and return reward"""
        if self.current_step >= len(self.data):
            return 0
        
        row = self.data.iloc[self.current_step]
        current_price = row.get('last_price', 0)
        
        if current_price <= 0:
            return 0
        
        reward = 0
        
        if action == 0:  # Buy
            cost = current_price * quantity * (1 + self.transaction_costs)
            max_quantity = int((self.balance * self.max_position_size) / cost)
            
            if max_quantity > 0 and self.balance >= cost:
                actual_quantity = min(quantity, max_quantity)
                total_cost = current_price * actual_quantity * (1 + self.transaction_costs)
                
                self.balance -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + actual_quantity
                
                # Record trade
                self.trades.append({
                    'action': 'BUY',
                    'symbol': symbol,
                    'quantity': actual_quantity,
                    'price': current_price,
                    'timestamp': self.current_step,
                    'cost': total_cost
                })
                
                # Small positive reward for taking action
                reward = 0.01
        
        elif action == 1:  # Sell
            if symbol in self.positions and self.positions[symbol] > 0:
                sell_quantity = min(quantity, self.positions[symbol])
                revenue = current_price * sell_quantity * (1 - self.transaction_costs)
                
                self.balance += revenue
                self.positions[symbol] -= sell_quantity
                
                if self.positions[symbol] == 0:
                    del self.positions[symbol]
                
                # Record trade
                self.trades.append({
                    'action': 'SELL',
                    'symbol': symbol,
                    'quantity': sell_quantity,
                    'price': current_price,
                    'timestamp': self.current_step,
                    'revenue': revenue
                })
                
                # Calculate profit/loss for reward
                # This is simplified - in practice, you'd track buy prices
                reward = 0.01
        
        # Action 2 (Hold) gets no immediate reward
        
        return reward
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.current_step >= len(self.data):
            return self.balance
        
        row = self.data.iloc[self.current_step]
        portfolio_value = self.balance
        
        # Add value of positions
        for symbol, quantity in self.positions.items():
            # Simplified - assumes all positions use same price
            current_price = row.get('last_price', 0)
            portfolio_value += current_price * quantity
        
        return portfolio_value
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        total_return = (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (self.portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_portfolio_value': self.portfolio_values[-1]
if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for trading decisions"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super(DQNNetwork, self).__init__()
            
            self.network = nn.Sequentia                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)

class RLTradingAgent:" if torch.cuda.is_available() else "cpu")
            self.q_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Fallback to simpler ML model
            self.q_network = RandomForestRegressor(n_estimators=100, random_state=42)
            self.is_trained = False
        
        # Model paths
        self.model_dir = "models/rl_agent"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Performance tracking
        self.training_history = []
        self.episode_rewards = []
        
        # Load pre-trained model if available
        self._load_model()
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        if TORCH_AVAILABLE:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        else:
            # Fallback for non-PyTorch implementation
            if hasattr(self, 'is_trained') and self.is_trained:
                q_values = self.q_network.predict([state])
                return np.argmax(q_values[0])
            else:
                return random.randrange(self.action_dim)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        if TORCH_AVAILABLE:
            self._replay_pytorch(batch)
        else:
            self._replay_sklearn(batch)
    
    def _replay_pytorch(self, batch):
        """PyTorch-based replay training"""
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _replay_sklearn(self, batch):
        """Scikit-learn based replay training (simplified)"""
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Simplified Q-learning update
        if hasattr(self, 'is_trained') and self.is_trained:
            current_q = self.q_network.predict(states)
            next_q = self.q_network.predict(next_states)
        else:
            current_q = np.random.random((len(batch), self.action_dim))
            next_q = np.random.random((len(batch), self.action_dim))
        
        targets = current_q.copy()
        
        for i in range(len(batch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Retrain model
        self.q_network.fit(states, targets)
        self.is_trained = True
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        if TORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def prepare_training_data(self, days_back: int = 30) -> pd.DataFrame:
        """Prepare training data from database"""
        session = self.SessionLocal()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get options data
            options_query = session.query(OptionsData).filter(
                OptionsData.timestamp >= cutoff_date,
                OptionsData.underlying_symbol.in_(['NIFTY', 'BANKNIFTY'])
            ).order_by(OptionsData.timestamp).all()
            
            # Get market data
            market_query = session.query(MarketData).filter(
                MarketData.timestamp >= cutoff_date,
                MarketData.symbol.in_(['NIFTY 50', 'NIFTY BANK'])
            ).order_by(MarketData.timestamp).all()
            
            # Convert to DataFrames
            options_df = pd.DataFrame([{
                'timestamp': opt.timestamp,
                'symbol': opt.symbol,
                'underlying_symbol': opt.underlying_symbol,
                'strike_price': opt.strike_price,
                'option_type': opt.option_type,
                'last_price': opt.last_price,
                'volume': opt.volume,
                'open_interest': opt.open_interest,
                'delta': opt.delta,
                'gamma': opt.gamma,
                'theta': opt.theta,
                'vega': opt.vega,
                'rho': opt.rho,
                'implied_volatility': opt.implied_volatility,
                'moneyness': opt.moneyness
            } for opt in options_query])
            
            market_df = pd.DataFrame([{
                'timestamp': mkt.timestamp,
                'symbol': mkt.symbol,
                'underlying_price': mkt.last_price,
                'volume': mkt.volume
            } for mkt in market_query])
            
            if options_df.empty:
                self.logger.warning("No options data found for training")
                return pd.DataFrame()
            
            # Merge and process data
            combined_df = self._process_training_data(options_df, market_df)
            
            return combined_df
            
        finally:
            session.close()
    
    def _process_training_data(self, options_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """Process and engineer features for training"""
        if options_df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp
        options_df = options_df.sort_values('timestamp')
        
        # Add time-based features
        options_df['hour'] = options_df['timestamp'].dt.hour
        options_df['day_of_week'] = options_df['timestamp'].dt.dayofweek
        
        # Calculate technical indicators (simplified)
        for symbol in options_df['underlying_symbol'].unique():
            symbol_data = options_df[options_df['underlying_symbol'] == symbol].copy()
            
            if len(symbol_data) > 20:
                # Moving averages
                symbol_data['moving_avg_20'] = symbol_data['last_price'].rolling(20).mean()
                symbol_data['moving_avg_50'] = symbol_data['last_price'].rolling(50).mean()
                
                # RSI (simplified)
                delta = symbol_data['last_price'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                symbol_data['rsi'] = 100 - (100 / (1 + rs))
                
                # Update main dataframe
                options_df.loc[options_df['underlying_symbol'] == symbol, 'moving_avg_20'] = symbol_data['moving_avg_20']
                options_df.loc[options_df['underlying_symbol'] == symbol, 'moving_avg_50'] = symbol_data['moving_avg_50']
                options_df.loc[options_df['underlying_symbol'] == symbol, 'rsi'] = symbol_data['rsi']
        
        # Add sentiment scores (simplified - would integrate with sentiment agent)
        options_df['sentiment_score'] = 0.0
        options_df['news_sentiment'] = 0.0
        options_df['social_sentiment'] = 0.0
        
        # Fill missing values
        options_df = options_df.fillna(0)
        
        return options_df
    
    def train(self, episodes: int = 1000, save_interval: int = 100):
        """Train the RL agent"""
        self.logger.info(f"Starting RL training for {episodes} episodes")
        
        # Prepare training data
        training_data = self.prepare_training_data(days_back=60)
        
        if training_data.empty:
            self.logger.error("No training data available")
            return
        
        # Create trading environment
        env = TradingEnvironment(training_data)
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Choose action
                action = self.act(state, training=True)
                
                # Get random symbol and quantity for training
                available_symbols = training_data['symbol'].unique()
                symbol = np.random.choice(available_symbols)
                quantity = np.random.randint(1, 5)
                
                # Execute action
                next_state, reward, done, info = env.step(action, symbol, quantity)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Train model
                if len(self.memory) > self.batch_size:
                    self.replay()
                
                state = next_state
                total_reward += reward
            
            # Update target network periodically
            if TORCH_AVAILABLE and episode % 10 == 0:
                self.update_target_network()
            
            # Track performance
            self.episode_rewards.append(total_reward)
            performance = env.get_performance_metrics()
            
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                self.logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, "
                               f"Epsilon: {self.epsilon:.4f}, "
                               f"Portfolio Return: {performance.get('total_return', 0):.4f}")
            
            # Save model periodically
            if episode % save_interval == 0:
                self._save_model()
        
        # Final save
        self._save_model()
        self.logger.info("RL training completed")
    
    def generate_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal for a specific symbol"""
        try:
            # Get current market state
            state = self._get_current_state(symbol)
            
            if state is None:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Get action from model
            action = self.act(state, training=False)
            
            # Convert action to signal
            signal_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
            signal = signal_map[action]
            
            # Calculate confidence (simplified)
            if TORCH_AVAILABLE:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                confidence = torch.softmax(q_values, dim=1).max().item()
            else:
                confidence = 0.7  # Default confidence for non-PyTorch
            
            # Get additional context
            context = self._get_signal_context(symbol, state)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'model_version': 'v1.0'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _get_current_state(self, symbol: str) -> Optional[np.ndarray]:
        """Get current market state for a symbol"""
        session = self.SessionLocal()
        
        try:
            # Get latest option data
            option_data = session.query(OptionsData).filter(
                OptionsData.symbol == symbol
            ).order_by(OptionsData.timestamp.desc()).first()
            
            if not option_data:
                return None
            
            # Get underlying price
            underlying_symbol = option_data.underlying_symbol
            underlying_price = self.greeks_agent._get_underlying_price(underlying_symbol)
            
            if not underlying_price:
                return None
            
            # Build state vector (simplified)
            state = np.array([
                underlying_price / 20000,  # Normalized
                option_data.last_price / 1000,
                option_data.volume / 10000,
                option_data.open_interest / 100000,
                option_data.delta or 0,
                option_data.gamma or 0,
                option_data.theta or 0,
                option_data.vega or 0,
                option_data.rho or 0,
                option_data.implied_volatility or 0,
                option_data.moneyness or 1,
                0.5,  # Sentiment placeholder
                0.5,  # News sentiment placeholder
                0.5,  # Social sentiment placeholder
                50 / 100,  # RSI placeholder
                0,  # MACD placeholder
                0,  # Bollinger upper placeholder
                0,  # Bollinger lower placeholder
                underlying_price / 20000,  # MA20 placeholder
                underlying_price / 20000,  # MA50 placeholder
                1.0,  # Balance ratio
                0,  # Position count
                0,  # Total positions
                1.0,  # Portfolio value ratio
                datetime.now().hour / 24,  # Hour
            ])
            
            # Ensure correct dimensions
            if len(state) < self.state_dim:
                state = np.pad(state, (0, self.state_dim - len(state)))
            elif len(state) > self.state_dim:
                state = state[:self.state_dim]
            
            return state.astype(np.float32)
            
        finally:
            session.close()
    
    def _get_signal_context(self, symbol: str, state: np.ndarray) -> Dict[str, Any]:
        """Get additional context for trading signal"""
        return {
            'market_conditions': 'normal',  # Simplified
            'volatility_regime': 'medium',
            'sentiment_score': state[11] if len(state) > 11 else 0.5,
            'time_to_expiry': 'medium',
            'moneyness': state[10] if len(state) > 10 else 1.0
        }
    
    def _save_model(self):
        """Save trained model"""
        try:
            if TORCH_AVAILABLE:
                torch.save({
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'episode_rewards': self.episode_rewards
                }, f"{self.model_dir}/rl_model.pth")
            else:
                with open(f"{self.model_dir}/rl_model.pkl", 'wb') as f:
                    pickle.dump({
                        'model': self.q_network,
                        'epsilon': self.epsilon,
                        'is_trained': getattr(self, 'is_trained', False),
                        'episode_rewards': self.episode_rewards
                    }, f)
            
            self.logger.info("RL model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving RL model: {e}")
    
    def _load_model(self):
        """Load pre-trained model"""
        try:
            if TORCH_AVAILABLE:
                model_path = f"{self.model_dir}/rl_model.pth"
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                    self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.epsilon = checkpoint.get('epsilon', self.epsilon)
                    self.episode_rewards = checkpoint.get('episode_rewards', [])
                    self.logger.info("RL model loaded successfully")
            else:
                model_path = f"{self.model_dir}/rl_model.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                        self.q_network = data['model']
                        self.epsilon = data.get('epsilon', self.epsilon)
                        self.is_trained = data.get('is_trained', False)
                        self.episode_rewards = data.get('episode_rewards', [])
                    self.logger.info("RL model loaded successfully")
                    
        except Exception as e:
            self.logger.error(f"Error loading RL model: {e}")
    
    def backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Backtest the trading strategy"""
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get historical data
        session = self.SessionLocal()
        
        try:
            options_query = session.query(OptionsData).filter(
                OptionsData.timestamp >= start_date,
                OptionsData.timestamp <= end_date,
                OptionsData.underlying_symbol.in_(['NIFTY', 'BANKNIFTY'])
            ).order_by(OptionsData.timestamp).all()
            
            if not options_query:
                return {'error': 'No historical data found for backtest period'}
            
            # Convert to DataFrame
            backtest_data = pd.DataFrame([{
                'timestamp': opt.timestamp,
                'symbol': opt.symbol,
                'last_price': opt.last_price,
                'volume': opt.volume,
                'delta': opt.delta,
                'gamma': opt.gamma,
                'theta': opt.theta,
                'vega': opt.vega
            } for opt in options_query])
            
            # Create environment
            env = TradingEnvironment(backtest_data, initial_balance=100000)
            
            # Run backtest
            state = env.reset()
            done = False
            signals_generated = []
            
            while not done:
                # Generate signal
                action = self.act(state, training=False)
                
                # Execute action
                symbol = backtest_data.iloc[env.current_step]['symbol']
                next_state, reward, done, info = env.step(action, symbol, 1)
                
                # Record signal
                signals_generated.append({
                    'timestamp': backtest_data.iloc[env.current_step]['timestamp'],
                    'symbol': symbol,
                    'action': ['BUY', 'SELL', 'HOLD'][action],
                    'portfolio_value': info['portfolio_value']
                })
                
                state = next_state
            
            # Calculate performance
            performance = env.get_performance_metrics()
            
            backtest_results = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'performance': performance,
                'signals_count': len(signals_generated),
                'trades_executed': len(env.trades),
                'final_balance': env.balance,
                'portfolio_values': env.portfolio_values
            }
            
            self.logger.info(f"Backtest completed. Total return: {performance.get('total_return', 0):.4f}")
            
            return backtest_results
            
        finally:
            session.close()
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        return {
            'episodes_trained': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'current_epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'model_type': 'PyTorch DQN' if TORCH_AVAILABLE else 'Scikit-learn RF',
            'last_updated': datetime.now().isoformat()
        }

