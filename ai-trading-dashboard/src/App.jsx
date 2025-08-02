import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import ZerodhaConfig from '@/components/ZerodhaConfig.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'
import { 
  TrendingUp, TrendingDown, Activity, DollarSign, AlertTriangle, 
  Shield, Target, Brain, BarChart3, Settings, User, LogOut,
  Play, Pause, RefreshCw, Eye, EyeOff, Bell, CheckCircle,
  XCircle, Clock, Zap, PieChart as PieChartIcon
} from 'lucide-react'
import './App.css'

// Mock data for demonstration
const mockMarketData = {
  nifty: { price: 19847.5, change: 125.3, changePercent: 0.63 },
  banknifty: { price: 45123.8, change: -89.2, changePercent: -0.20 }
}

const mockPositions = [
  { symbol: 'NIFTY24JAN20000CE', quantity: 50, ltp: 125.5, pnl: 2500, pnlPercent: 8.5 },
  { symbol: 'BANKNIFTY24JAN45500PE', quantity: -25, ltp: 89.3, pnl: -1200, pnlPercent: -4.2 },
  { symbol: 'NIFTY24JAN19800PE', quantity: 30, ltp: 67.8, pnl: 890, pnlPercent: 3.1 }
]

const mockOrders = [
  { id: 'ORD001', symbol: 'NIFTY24JAN20000CE', type: 'BUY', quantity: 25, price: 120.5, status: 'COMPLETE', time: '10:15:23' },
  { id: 'ORD002', symbol: 'BANKNIFTY24JAN45000CE', type: 'SELL', quantity: 50, price: 95.2, status: 'PENDING', time: '10:18:45' },
  { id: 'ORD003', symbol: 'NIFTY24JAN19500PE', type: 'BUY', quantity: 75, price: 45.8, status: 'CANCELLED', time: '10:22:11' }
]

const mockSignals = [
  { symbol: 'NIFTY24JAN20000CE', signal: 'BUY', confidence: 0.87, reason: 'Strong bullish momentum with high IV rank' },
  { symbol: 'BANKNIFTY24JAN45500PE', signal: 'SELL', confidence: 0.92, reason: 'Overbought conditions with negative sentiment' },
  { symbol: 'NIFTY24JAN19800CE', signal: 'HOLD', confidence: 0.65, reason: 'Mixed signals, await confirmation' }
]

const mockRiskMetrics = {
  portfolioValue: 125000,
  dailyPnL: 2190,
  maxDrawdown: -3.2,
  riskLevel: 'MEDIUM',
  deltaExposure: 0.15,
  gammaExposure: 0.08,
  thetaDecay: -450,
  vegaRisk: 1200
}

const performanceData = [
  { date: '2024-01-15', pnl: 1200, cumulative: 1200 },
  { date: '2024-01-16', pnl: -800, cumulative: 400 },
  { date: '2024-01-17', pnl: 2100, cumulative: 2500 },
  { date: '2024-01-18', pnl: 1500, cumulative: 4000 },
  { date: '2024-01-19', pnl: -600, cumulative: 3400 },
  { date: '2024-01-20', pnl: 890, cumulative: 4290 },
  { date: '2024-01-21', pnl: 1200, cumulative: 5490 }
]

const sentimentData = [
  { name: 'Positive', value: 45, color: '#10b981' },
  { name: 'Neutral', value: 35, color: '#6b7280' },
  { name: 'Negative', value: 20, color: '#ef4444' }
]

// API service
const API_BASE = "." // Use relative path for API calls

class ApiService {
  static async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    }
    
    try {
      const response = await fetch(url, config)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('API request failed:', error)
      throw error
    }
  }

  static async getSystemStatus() {
    return this.request('/system/status')
  }

  static async getPositions() {
    return this.request('/portfolio/positions')
  }

  static async getOrders() {
    return this.request('/orders')
  }

  static async placeOrder(orderData) {
    return this.request('/orders/place', {
      method: 'POST',
      body: JSON.stringify(orderData)
    })
  }

  static async getRiskAssessment() {
    return this.request('/risk/assessment')
  }

  static async generateSignal(symbol) {
    return this.request('/signals/generate', {
      method: 'POST',
      body: JSON.stringify({ symbol })
    })
  }

  static async getMarketSentiment() {
    return this.request('/market/sentiment')
  }
}

// Components
function Header({ user, onLogout }) {
  return (
    <div className="border-b bg-white shadow-sm">
      <div className="flex h-16 items-center px-6 justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Brain className="h-8 w-8 text-blue-600" />
            <h1 className="text-xl font-bold text-gray-900">AI Trading Agent</h1>
          </div>
          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
            <Activity className="h-3 w-3 mr-1" />
            Live
          </Badge>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Clock className="h-4 w-4" />
            <span>{new Date().toLocaleTimeString()}</span>
          </div>
          
          <Button variant="ghost" size="sm">
            <Bell className="h-4 w-4" />
          </Button>
          
          <div className="flex items-center space-x-2">
            <User className="h-4 w-4" />
            <span className="text-sm font-medium">{user?.username || 'Admin'}</span>
          </div>
          
          <Button variant="ghost" size="sm" onClick={onLogout}>
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

function MarketOverview() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">NIFTY 50</p>
              <p className="text-2xl font-bold">{mockMarketData.nifty.price.toLocaleString()}</p>
              <div className="flex items-center mt-1">
                {mockMarketData.nifty.change > 0 ? (
                  <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-500 mr-1" />
                )}
                <span className={`text-sm font-medium ${mockMarketData.nifty.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {mockMarketData.nifty.change > 0 ? '+' : ''}{mockMarketData.nifty.change} ({mockMarketData.nifty.changePercent}%)
                </span>
              </div>
            </div>
            <div className="h-12 w-12 bg-blue-100 rounded-full flex items-center justify-center">
              <BarChart3 className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">BANK NIFTY</p>
              <p className="text-2xl font-bold">{mockMarketData.banknifty.price.toLocaleString()}</p>
              <div className="flex items-center mt-1">
                {mockMarketData.banknifty.change > 0 ? (
                  <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                ) : (
                  <TrendingDown className="h-4 w-4 text-red-500 mr-1" />
                )}
                <span className={`text-sm font-medium ${mockMarketData.banknifty.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {mockMarketData.banknifty.change > 0 ? '+' : ''}{mockMarketData.banknifty.change} ({mockMarketData.banknifty.changePercent}%)
                </span>
              </div>
            </div>
            <div className="h-12 w-12 bg-purple-100 rounded-full flex items-center justify-center">
              <BarChart3 className="h-6 w-6 text-purple-600" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Portfolio P&L</p>
              <p className="text-2xl font-bold">₹{mockRiskMetrics.dailyPnL.toLocaleString()}</p>
              <div className="flex items-center mt-1">
                <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                <span className="text-sm font-medium text-green-600">+1.75%</span>
              </div>
            </div>
            <div className="h-12 w-12 bg-green-100 rounded-full flex items-center justify-center">
              <DollarSign className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Risk Level</p>
              <p className="text-2xl font-bold">{mockRiskMetrics.riskLevel}</p>
              <div className="flex items-center mt-1">
                <Shield className="h-4 w-4 text-yellow-500 mr-1" />
                <span className="text-sm font-medium text-yellow-600">Monitored</span>
              </div>
            </div>
            <div className="h-12 w-12 bg-yellow-100 rounded-full flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-yellow-600" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function TradingSignals() {
  const [selectedSymbol, setSelectedSymbol] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)

  const handleGenerateSignal = async () => {
    if (!selectedSymbol) return
    
    setIsGenerating(true)
    try {
      await ApiService.generateSignal(selectedSymbol)
      // Refresh signals
    } catch (error) {
      console.error('Failed to generate signal:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Target className="h-5 w-5" />
          <span>AI Trading Signals</span>
        </CardTitle>
        <CardDescription>
          Real-time trading recommendations powered by reinforcement learning
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex space-x-2 mb-4">
          <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
            <SelectTrigger className="flex-1">
              <SelectValue placeholder="Select symbol to analyze" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="NIFTY24JAN20000CE">NIFTY 20000 CE</SelectItem>
              <SelectItem value="NIFTY24JAN19800PE">NIFTY 19800 PE</SelectItem>
              <SelectItem value="BANKNIFTY24JAN45000CE">BANKNIFTY 45000 CE</SelectItem>
              <SelectItem value="BANKNIFTY24JAN44500PE">BANKNIFTY 44500 PE</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={handleGenerateSignal} disabled={!selectedSymbol || isGenerating}>
            {isGenerating ? (
              <RefreshCw className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <Zap className="h-4 w-4 mr-2" />
            )}
            Generate Signal
          </Button>
        </div>

        <div className="space-y-3">
          {mockSignals.map((signal, index) => (
            <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex-1">
                <div className="font-medium text-sm">{signal.symbol}</div>
                <div className="text-xs text-gray-500 mt-1">{signal.reason}</div>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="text-right">
                  <div className={`font-bold text-sm ${
                    signal.signal === 'BUY' ? 'text-green-600' : 
                    signal.signal === 'SELL' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {signal.signal}
                  </div>
                  <div className="text-xs text-gray-500">
                    {(signal.confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
                
                <Badge variant={
                  signal.signal === 'BUY' ? 'default' : 
                  signal.signal === 'SELL' ? 'destructive' : 'secondary'
                }>
                  {signal.signal}
                </Badge>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function OrderManagement() {
  const [orderForm, setOrderForm] = useState({
    symbol: '',
    quantity: '',
    price: '',
    orderType: 'MARKET',
    transactionType: 'BUY'
  })

  const handlePlaceOrder = async (e) => {
    e.preventDefault()
    try {
      await ApiService.placeOrder({
        symbol: orderForm.symbol,
        quantity: parseInt(orderForm.quantity),
        price: parseFloat(orderForm.price) || undefined,
        order_type: orderForm.orderType,
        transaction_type: orderForm.transactionType
      })
      // Reset form and refresh orders
      setOrderForm({
        symbol: '',
        quantity: '',
        price: '',
        orderType: 'MARKET',
        transactionType: 'BUY'
      })
    } catch (error) {
      console.error('Failed to place order:', error)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Place Order</CardTitle>
          <CardDescription>Execute trades with AI-powered validation</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePlaceOrder} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="symbol">Symbol</Label>
                <Input
                  id="symbol"
                  value={orderForm.symbol}
                  onChange={(e) => setOrderForm({...orderForm, symbol: e.target.value})}
                  placeholder="e.g., NIFTY24JAN20000CE"
                  required
                />
              </div>
              <div>
                <Label htmlFor="quantity">Quantity</Label>
                <Input
                  id="quantity"
                  type="number"
                  value={orderForm.quantity}
                  onChange={(e) => setOrderForm({...orderForm, quantity: e.target.value})}
                  placeholder="25"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="transactionType">Type</Label>
                <Select value={orderForm.transactionType} onValueChange={(value) => setOrderForm({...orderForm, transactionType: value})}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="BUY">BUY</SelectItem>
                    <SelectItem value="SELL">SELL</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="orderType">Order Type</Label>
                <Select value={orderForm.orderType} onValueChange={(value) => setOrderForm({...orderForm, orderType: value})}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MARKET">MARKET</SelectItem>
                    <SelectItem value="LIMIT">LIMIT</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {orderForm.orderType === 'LIMIT' && (
              <div>
                <Label htmlFor="price">Price</Label>
                <Input
                  id="price"
                  type="number"
                  step="0.05"
                  value={orderForm.price}
                  onChange={(e) => setOrderForm({...orderForm, price: e.target.value})}
                  placeholder="125.50"
                />
              </div>
            )}

            <Button type="submit" className="w-full">
              <Play className="h-4 w-4 mr-2" />
              Place Order
            </Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Orders</CardTitle>
          <CardDescription>Order execution history and status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {mockOrders.map((order) => (
              <div key={order.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex-1">
                  <div className="font-medium text-sm">{order.symbol}</div>
                  <div className="text-xs text-gray-500">
                    {order.type} {order.quantity} @ ₹{order.price} • {order.time}
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  {order.status === 'COMPLETE' && <CheckCircle className="h-4 w-4 text-green-500" />}
                  {order.status === 'PENDING' && <Clock className="h-4 w-4 text-yellow-500" />}
                  {order.status === 'CANCELLED' && <XCircle className="h-4 w-4 text-red-500" />}
                  
                  <Badge variant={
                    order.status === 'COMPLETE' ? 'default' :
                    order.status === 'PENDING' ? 'secondary' : 'destructive'
                  }>
                    {order.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function PortfolioView() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Current Positions</CardTitle>
            <CardDescription>Active options positions and P&L</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {mockPositions.map((position, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{position.symbol}</div>
                    <div className="text-xs text-gray-500">
                      Qty: {position.quantity} • LTP: ₹{position.ltp}
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className={`font-bold text-sm ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {position.pnl >= 0 ? '+' : ''}₹{position.pnl.toLocaleString()}
                    </div>
                    <div className={`text-xs ${position.pnlPercent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Performance Chart</CardTitle>
            <CardDescription>Daily P&L and cumulative performance</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="cumulative" stroke="#2563eb" strokeWidth={2} name="Cumulative P&L" />
                <Bar dataKey="pnl" fill="#10b981" name="Daily P&L" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Risk Metrics</CardTitle>
          <CardDescription>Portfolio risk analysis and Greeks exposure</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{mockRiskMetrics.deltaExposure}</div>
              <div className="text-sm text-gray-600">Delta Exposure</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{mockRiskMetrics.gammaExposure}</div>
              <div className="text-sm text-gray-600">Gamma Exposure</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-red-600">₹{mockRiskMetrics.thetaDecay}</div>
              <div className="text-sm text-gray-600">Theta Decay</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <div className="text-2xl font-bold text-green-600">₹{mockRiskMetrics.vegaRisk}</div>
              <div className="text-sm text-gray-600">Vega Risk</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function AnalyticsView() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Market Sentiment</CardTitle>
            <CardDescription>AI-powered sentiment analysis from news and social media</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={sentimentData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model Performance</CardTitle>
            <CardDescription>AI model accuracy and prediction metrics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Prediction Accuracy</span>
                <span className="text-lg font-bold text-green-600">92.3%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-600 h-2 rounded-full" style={{width: '92.3%'}}></div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Sharpe Ratio</span>
                <span className="text-lg font-bold text-blue-600">2.45</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{width: '80%'}}></div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Win Rate</span>
                <span className="text-lg font-bold text-purple-600">68.7%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-purple-600 h-2 rounded-full" style={{width: '68.7%'}}></div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Strategy Performance</CardTitle>
          <CardDescription>Historical performance and backtesting results</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="cumulative" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}

function MainDashboard() {
  const [activeTab, setActiveTab] = useState('overview')
  const [user] = useState({ username: 'Admin', role: 'admin' })

  const handleLogout = () => {
    // Implement logout logic
    console.log('Logout clicked')
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header user={user} onLogout={handleLogout} />
      
      <div className="p-6">
        <MarketOverview />
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="trading">Trading</TabsTrigger>
            <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="config">Configuration</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TradingSignals />
              <Card>
                <CardHeader>
                  <CardTitle>System Status</CardTitle>
                  <CardDescription>AI trading system health and performance</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Trading Engine</span>
                      <Badge variant="default" className="bg-green-100 text-green-800">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Active
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Risk Management</span>
                      <Badge variant="default" className="bg-green-100 text-green-800">
                        <Shield className="h-3 w-3 mr-1" />
                        Protected
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Data Feed</span>
                      <Badge variant="default" className="bg-green-100 text-green-800">
                        <Activity className="h-3 w-3 mr-1" />
                        Live
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">AI Models</span>
                      <Badge variant="default" className="bg-blue-100 text-blue-800">
                        <Brain className="h-3 w-3 mr-1" />
                        Learning
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="trading">
            <OrderManagement />
          </TabsContent>
          
          <TabsContent value="portfolio">
            <PortfolioView />
          </TabsContent>
          
          <TabsContent value="analytics">
            <AnalyticsView />
          </TabsContent>
          
          <TabsContent value="config">
            <ZerodhaConfig />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(true) // Set to true for demo

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-center">AI Trading Agent</CardTitle>
            <CardDescription className="text-center">
              Sign in to access your trading dashboard
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="username">Username</Label>
                <Input id="username" placeholder="Enter username" />
              </div>
              <div>
                <Label htmlFor="password">Password</Label>
                <Input id="password" type="password" placeholder="Enter password" />
              </div>
              <Button className="w-full" onClick={() => setIsAuthenticated(true)}>
                Sign In
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainDashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App

