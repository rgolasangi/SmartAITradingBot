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
        }

# Define DQNNetwork only if PyTorch is available
if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for trading decisions"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super(DQNNetwork, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)

class RLTradingAgent:
    """Reinforcement Learning Trading Agent"""
    
    def __init__(self, state_dim: int = 25, action_dim: int = 3, 
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Database connection
        self.engine = create_engine(os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/trading_db'))
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize AI agents
        self.sentiment_agent = SentimentAgent()
        self.greeks_agent = OptionsGreeksAgent()
        
        # Initialize models
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    def _load_model(self):
        """Load pre-trained model if available"""
        try:
            if TORCH_AVAILABLE:
                model_path = os.path.join(self.model_dir, "dqn_model.pth")
                if os.path.exists(model_path):
                    self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    logging.info("Loaded pre-trained DQN model")
            else:
                model_path = os.path.join(self.model_dir, "rf_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.q_network = pickle.load(f)
                    self.is_trained = True
                    logging.info("Loaded pre-trained RandomForest model")
        except Exception as e:
            logging.warning(f"Could not load pre-trained model: {e}")
    
    def save_model(self):
        """Save trained model"""
        try:
            if TORCH_AVAILABLE:
                model_path = os.path.join(self.model_dir, "dqn_model.pth")
                torch.save(self.q_network.state_dict(), model_path)
                logging.info("Saved DQN model")
            else:
                model_path = os.path.join(self.model_dir, "rf_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.q_network, f)
                logging.info("Saved RandomForest model")
        except Exception as e:
            logging.error(f"Could not save model: {e}")

