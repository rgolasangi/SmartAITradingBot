import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { 
  Settings, 
  Key, 
  Shield, 
  CheckCircle, 
  XCircle, 
  Eye, 
  EyeOff,
  RefreshCw,
  AlertTriangle
} from 'lucide-react'

const API_BASE = "/api"

export function ZerodhaConfig() {
  const [credentials, setCredentials] = useState({
    apiKey: '',
    apiSecret: '',
    accessToken: ''
  })
  
  const [showSecrets, setShowSecrets] = useState({
    apiSecret: false,
    accessToken: false
  })
  
  const [connectionStatus, setConnectionStatus] = useState('disconnected') // disconnected, connecting, connected, error
  const [statusMessage, setStatusMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  // Load existing credentials on component mount
  useEffect(() => {
    loadCredentials()
    checkConnectionStatus()
  }, [])

  const loadCredentials = async () => {
    try {
      const response = await fetch(`${API_BASE}/zerodha/credentials`)
      if (response.ok) {
        const data = await response.json()
        setCredentials({
          apiKey: data.api_key || '',
          apiSecret: data.api_secret ? '••••••••••••••••' : '',
          accessToken: data.access_token ? '••••••••••••••••••••••••••••••••' : ''
        })
      }
    } catch (error) {
      console.error('Failed to load credentials:', error)
    }
  }

  const checkConnectionStatus = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${API_BASE}/zerodha/status`)
      if (response.ok) {
        const data = await response.json()
        setConnectionStatus(data.status)
        setStatusMessage(data.message || '')
      } else {
        setConnectionStatus('error')
        setStatusMessage('Failed to check connection status')
      }
    } catch (error) {
      setConnectionStatus('error')
      setStatusMessage('Unable to connect to backend')
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (field, value) => {
    setCredentials(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const toggleShowSecret = (field) => {
    setShowSecrets(prev => ({
      ...prev,
      [field]: !prev[field]
    }))
  }

  const handleSaveCredentials = async () => {
    setIsSaving(true)
    try {
      const response = await fetch(`${API_BASE}/zerodha/credentials`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          api_key: credentials.apiKey,
          api_secret: credentials.apiSecret,
          access_token: credentials.accessToken
        })
      })

      if (response.ok) {
        setStatusMessage('Credentials saved successfully')
        // Test connection after saving
        await testConnection()
      } else {
        const error = await response.json()
        setStatusMessage(error.message || 'Failed to save credentials')
        setConnectionStatus('error')
      }
    } catch (error) {
      setStatusMessage('Failed to save credentials')
      setConnectionStatus('error')
    } finally {
      setIsSaving(false)
    }
  }

  const testConnection = async () => {
    setConnectionStatus('connecting')
    setStatusMessage('Testing connection...')
    
    try {
      const response = await fetch(`${API_BASE}/zerodha/test-connection`, {
        method: 'POST'
      })

      if (response.ok) {
        const data = await response.json()
        setConnectionStatus('connected')
        setStatusMessage(`Connected successfully! User: ${data.user_name || 'Unknown'}`)
      } else {
        const error = await response.json()
        setConnectionStatus('error')
        setStatusMessage(error.message || 'Connection test failed')
      }
    } catch (error) {
      setConnectionStatus('error')
      setStatusMessage('Connection test failed')
    }
  }

  const getStatusBadge = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Badge className="bg-green-100 text-green-800 border-green-200"><CheckCircle className="h-3 w-3 mr-1" />Connected</Badge>
      case 'connecting':
        return <Badge className="bg-blue-100 text-blue-800 border-blue-200"><RefreshCw className="h-3 w-3 mr-1 animate-spin" />Connecting</Badge>
      case 'error':
        return <Badge className="bg-red-100 text-red-800 border-red-200"><XCircle className="h-3 w-3 mr-1" />Error</Badge>
      default:
        return <Badge className="bg-gray-100 text-gray-800 border-gray-200"><AlertTriangle className="h-3 w-3 mr-1" />Disconnected</Badge>
    }
  }

  const getStatusAlert = () => {
    if (!statusMessage) return null

    const variant = connectionStatus === 'connected' ? 'default' : 
                   connectionStatus === 'error' ? 'destructive' : 'default'

    return (
      <Alert className={`mt-4 ${variant === 'destructive' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}`}>
        <AlertDescription className={variant === 'destructive' ? 'text-red-800' : 'text-green-800'}>
          {statusMessage}
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <CardTitle>Zerodha API Configuration</CardTitle>
          </div>
          {getStatusBadge()}
        </div>
        <CardDescription>
          Configure your Zerodha Kite Connect API credentials to enable automated trading
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* API Key */}
        <div className="space-y-2">
          <Label htmlFor="apiKey" className="flex items-center space-x-2">
            <Key className="h-4 w-4" />
            <span>API Key</span>
          </Label>
          <Input
            id="apiKey"
            type="text"
            placeholder="Enter your Zerodha API Key"
            value={credentials.apiKey}
            onChange={(e) => handleInputChange('apiKey', e.target.value)}
            className="font-mono"
          />
          <p className="text-xs text-gray-500">
            Get this from your Kite Connect developer console
          </p>
        </div>

        {/* API Secret */}
        <div className="space-y-2">
          <Label htmlFor="apiSecret" className="flex items-center space-x-2">
            <Shield className="h-4 w-4" />
            <span>API Secret</span>
          </Label>
          <div className="relative">
            <Input
              id="apiSecret"
              type={showSecrets.apiSecret ? "text" : "password"}
              placeholder="Enter your Zerodha API Secret"
              value={credentials.apiSecret}
              onChange={(e) => handleInputChange('apiSecret', e.target.value)}
              className="font-mono pr-10"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3"
              onClick={() => toggleShowSecret('apiSecret')}
            >
              {showSecrets.apiSecret ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
          <p className="text-xs text-gray-500">
            Keep this secret and never share it publicly
          </p>
        </div>

        {/* Access Token */}
        <div className="space-y-2">
          <Label htmlFor="accessToken" className="flex items-center space-x-2">
            <Key className="h-4 w-4" />
            <span>Access Token</span>
          </Label>
          <div className="relative">
            <Input
              id="accessToken"
              type={showSecrets.accessToken ? "text" : "password"}
              placeholder="Enter your Zerodha Access Token"
              value={credentials.accessToken}
              onChange={(e) => handleInputChange('accessToken', e.target.value)}
              className="font-mono pr-10"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-0 top-0 h-full px-3"
              onClick={() => toggleShowSecret('accessToken')}
            >
              {showSecrets.accessToken ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
          <p className="text-xs text-gray-500">
            Generate this through the Zerodha login flow (valid for 24 hours)
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-3 pt-4">
          <Button 
            onClick={handleSaveCredentials} 
            disabled={!credentials.apiKey || !credentials.apiSecret || !credentials.accessToken || isSaving}
            className="flex-1"
          >
            {isSaving ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Shield className="h-4 w-4 mr-2" />
                Save Credentials
              </>
            )}
          </Button>
          
          <Button 
            variant="outline" 
            onClick={testConnection}
            disabled={connectionStatus === 'connecting' || isLoading}
          >
            {connectionStatus === 'connecting' || isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Testing...
              </>
            ) : (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Test Connection
              </>
            )}
          </Button>
        </div>

        {/* Status Message */}
        {getStatusAlert()}

        {/* Help Section */}
        <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="font-medium text-blue-900 mb-2">How to get your credentials:</h4>
          <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
            <li>Register for a Kite Connect developer account at <code className="bg-blue-100 px-1 rounded">kite.trade</code></li>
            <li>Create a new app to get your API Key and Secret</li>
            <li>Use the login flow to generate an Access Token (expires in 24 hours)</li>
            <li>Enter all three credentials above and test the connection</li>
          </ol>
        </div>
      </CardContent>
    </Card>
  )
}

export default ZerodhaConfig

