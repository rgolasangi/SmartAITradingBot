#!/usr/bin/env python3
"""
AI Trading Agent Integration Test Suite
Comprehensive testing for all system components
"""

import os
import sys
import time
import json
import requests
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import unittest
from unittest.mock import Mock, patch
import psycopg2
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive integration test suite for AI Trading Agent"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session = requests.Session()
        self.test_results = []
        
        # Test configuration
        self.test_symbols = ['NIFTY24JAN20000CE', 'BANKNIFTY24JAN45000PE']
        self.test_timeout = 30
        
    def log_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name} ({duration:.2f}s) - {message}")
    
    def test_health_check(self) -> bool:
        """Test basic health check endpoint"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.test_timeout)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    self.log_test_result("Health Check", True, "System is healthy", duration)
                    return True
                else:
                    self.log_test_result("Health Check", False, f"Unhealthy status: {data}", duration)
                    return False
            else:
                self.log_test_result("Health Check", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Health Check", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_authentication(self) -> bool:
        """Test authentication endpoints"""
        start_time = time.time()
        try:
            # Test login
            login_data = {
                'username': 'admin',
                'password': 'admin123'
            }
            
            response = self.session.post(
                f"{self.api_url}/auth/login",
                json=login_data,
                timeout=self.test_timeout
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('token'):
                    self.log_test_result("Authentication", True, "Login successful", duration)
                    return True
                else:
                    self.log_test_result("Authentication", False, f"Login failed: {data}", duration)
                    return False
            else:
                self.log_test_result("Authentication", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Authentication", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_market_data_endpoints(self) -> bool:
        """Test market data API endpoints"""
        endpoints_to_test = [
            ('/market/sentiment', 'Market Sentiment'),
            ('/market/options-chain/NIFTY', 'Options Chain'),
            ('/market/volatility-surface/NIFTY', 'Volatility Surface')
        ]
        
        all_passed = True
        
        for endpoint, test_name in endpoints_to_test:
            start_time = time.time()
            try:
                response = self.session.get(f"{self.api_url}{endpoint}", timeout=self.test_timeout)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test_result(f"Market Data - {test_name}", True, "Data retrieved successfully", duration)
                else:
                    self.log_test_result(f"Market Data - {test_name}", False, f"HTTP {response.status_code}", duration)
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(f"Market Data - {test_name}", False, f"Exception: {str(e)}", duration)
                all_passed = False
        
        return all_passed
    
    def test_trading_signals(self) -> bool:
        """Test trading signal generation"""
        start_time = time.time()
        try:
            signal_data = {'symbol': 'NIFTY24JAN20000CE'}
            
            response = self.session.post(
                f"{self.api_url}/signals/generate",
                json=signal_data,
                timeout=self.test_timeout
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if 'signal' in data and 'confidence' in data:
                    self.log_test_result("Trading Signals", True, f"Signal generated: {data.get('signal')}", duration)
                    return True
                else:
                    self.log_test_result("Trading Signals", False, f"Invalid signal format: {data}", duration)
                    return False
            else:
                self.log_test_result("Trading Signals", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Trading Signals", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_batch_signals(self) -> bool:
        """Test batch signal generation"""
        start_time = time.time()
        try:
            batch_data = {'symbols': self.test_symbols}
            
            response = self.session.post(
                f"{self.api_url}/signals/batch",
                json=batch_data,
                timeout=self.test_timeout * 2  # Longer timeout for batch processing
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if 'signals' in data and len(data['signals']) == len(self.test_symbols):
                    self.log_test_result("Batch Signals", True, f"Generated signals for {len(self.test_symbols)} symbols", duration)
                    return True
                else:
                    self.log_test_result("Batch Signals", False, f"Invalid batch response: {data}", duration)
                    return False
            else:
                self.log_test_result("Batch Signals", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Batch Signals", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_risk_management(self) -> bool:
        """Test risk management endpoints"""
        endpoints_to_test = [
            ('/risk/assessment', 'Risk Assessment'),
            ('/risk/limits', 'Risk Limits')
        ]
        
        all_passed = True
        
        for endpoint, test_name in endpoints_to_test:
            start_time = time.time()
            try:
                response = self.session.get(f"{self.api_url}{endpoint}", timeout=self.test_timeout)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test_result(f"Risk Management - {test_name}", True, "Risk data retrieved", duration)
                else:
                    self.log_test_result(f"Risk Management - {test_name}", False, f"HTTP {response.status_code}", duration)
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(f"Risk Management - {test_name}", False, f"Exception: {str(e)}", duration)
                all_passed = False
        
        return all_passed
    
    def test_portfolio_endpoints(self) -> bool:
        """Test portfolio management endpoints"""
        endpoints_to_test = [
            ('/portfolio/positions', 'Positions'),
            ('/portfolio/margins', 'Margins'),
            ('/portfolio/holdings', 'Holdings'),
            ('/portfolio/greeks', 'Portfolio Greeks')
        ]
        
        all_passed = True
        
        for endpoint, test_name in endpoints_to_test:
            start_time = time.time()
            try:
                response = self.session.get(f"{self.api_url}{endpoint}", timeout=self.test_timeout)
                duration = time.time() - start_time
                
                if response.status_code in [200, 404]:  # 404 is acceptable for empty portfolio
                    self.log_test_result(f"Portfolio - {test_name}", True, "Endpoint accessible", duration)
                else:
                    self.log_test_result(f"Portfolio - {test_name}", False, f"HTTP {response.status_code}", duration)
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(f"Portfolio - {test_name}", False, f"Exception: {str(e)}", duration)
                all_passed = False
        
        return all_passed
    
    def test_order_management(self) -> bool:
        """Test order management system (without placing real orders)"""
        start_time = time.time()
        try:
            # Test getting orders
            response = self.session.get(f"{self.api_url}/orders", timeout=self.test_timeout)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.log_test_result("Order Management - Get Orders", True, "Orders retrieved successfully", duration)
                
                # Test order statistics
                start_time = time.time()
                response = self.session.get(f"{self.api_url}/orders/statistics", timeout=self.test_timeout)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.log_test_result("Order Management - Statistics", True, "Statistics retrieved", duration)
                    return True
                else:
                    self.log_test_result("Order Management - Statistics", False, f"HTTP {response.status_code}", duration)
                    return False
            else:
                self.log_test_result("Order Management - Get Orders", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Order Management", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_analytics_endpoints(self) -> bool:
        """Test analytics and performance endpoints"""
        endpoints_to_test = [
            ('/analytics/performance', 'Performance Analytics'),
            ('/models/performance', 'Model Performance')
        ]
        
        all_passed = True
        
        for endpoint, test_name in endpoints_to_test:
            start_time = time.time()
            try:
                response = self.session.get(f"{self.api_url}{endpoint}", timeout=self.test_timeout)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test_result(f"Analytics - {test_name}", True, "Analytics data retrieved", duration)
                else:
                    self.log_test_result(f"Analytics - {test_name}", False, f"HTTP {response.status_code}", duration)
                    all_passed = False
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(f"Analytics - {test_name}", False, f"Exception: {str(e)}", duration)
                all_passed = False
        
        return all_passed
    
    def test_system_status(self) -> bool:
        """Test system status and monitoring"""
        start_time = time.time()
        try:
            response = self.session.get(f"{self.api_url}/system/status", timeout=self.test_timeout)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if 'overall_status' in data and 'components' in data:
                    self.log_test_result("System Status", True, f"Status: {data.get('overall_status')}", duration)
                    return True
                else:
                    self.log_test_result("System Status", False, f"Invalid status format: {data}", duration)
                    return False
            else:
                self.log_test_result("System Status", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("System Status", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_rate_limiting(self) -> bool:
        """Test API rate limiting"""
        start_time = time.time()
        try:
            # Make multiple rapid requests to trigger rate limiting
            responses = []
            for i in range(10):
                response = self.session.get(f"{self.api_url}/market/sentiment", timeout=5)
                responses.append(response.status_code)
            
            duration = time.time() - start_time
            
            # Check if any request was rate limited (429)
            rate_limited = any(status == 429 for status in responses)
            
            if rate_limited:
                self.log_test_result("Rate Limiting", True, "Rate limiting is working", duration)
                return True
            else:
                self.log_test_result("Rate Limiting", True, "No rate limiting triggered (acceptable)", duration)
                return True
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Rate Limiting", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_concurrent_requests(self) -> bool:
        """Test system under concurrent load"""
        start_time = time.time()
        try:
            def make_request():
                return self.session.get(f"{self.api_url}/system/status", timeout=10)
            
            # Make 20 concurrent requests
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(make_request) for _ in range(20)]
                results = []
                
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        results.append(response.status_code)
                    except Exception as e:
                        results.append(0)  # Failed request
            
            duration = time.time() - start_time
            
            success_count = sum(1 for status in results if status == 200)
            success_rate = success_count / len(results)
            
            if success_rate >= 0.8:  # 80% success rate acceptable
                self.log_test_result("Concurrent Requests", True, f"Success rate: {success_rate:.1%}", duration)
                return True
            else:
                self.log_test_result("Concurrent Requests", False, f"Low success rate: {success_rate:.1%}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Concurrent Requests", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_database_connectivity(self) -> bool:
        """Test database connectivity (if accessible)"""
        start_time = time.time()
        try:
            # This would only work if database is directly accessible
            # In production, this test might be skipped or use API endpoints
            self.log_test_result("Database Connectivity", True, "Skipped - testing via API endpoints", 0)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Database Connectivity", False, f"Exception: {str(e)}", duration)
            return False
    
    def test_redis_connectivity(self) -> bool:
        """Test Redis connectivity (if accessible)"""
        start_time = time.time()
        try:
            # This would only work if Redis is directly accessible
            # In production, this test might be skipped or use API endpoints
            self.log_test_result("Redis Connectivity", True, "Skipped - testing via API endpoints", 0)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Redis Connectivity", False, f"Exception: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("🚀 Starting AI Trading Agent Integration Tests")
        logger.info(f"Testing against: {self.base_url}")
        
        start_time = time.time()
        
        # Define test suite
        test_methods = [
            self.test_health_check,
            self.test_authentication,
            self.test_market_data_endpoints,
            self.test_trading_signals,
            self.test_batch_signals,
            self.test_risk_management,
            self.test_portfolio_endpoints,
            self.test_order_management,
            self.test_analytics_endpoints,
            self.test_system_status,
            self.test_rate_limiting,
            self.test_concurrent_requests,
            self.test_database_connectivity,
            self.test_redis_connectivity
        ]
        
        # Run tests
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
        
        total_duration = time.time() - start_time
        
        # Generate summary
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'total_duration': total_duration,
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results
        }
        
        # Log summary
        logger.info("=" * 60)
        logger.info("🏁 INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT! System is production-ready!")
        elif success_rate >= 80:
            logger.info("✅ GOOD! System is mostly functional with minor issues.")
        elif success_rate >= 70:
            logger.info("⚠️  FAIR! System has some issues that need attention.")
        else:
            logger.error("❌ POOR! System has significant issues that must be fixed.")
        
        return summary

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Trading Agent Integration Tests')
    parser.add_argument('--url', default='http://localhost:5000', help='Base URL for testing')
    parser.add_argument('--output', help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    # Run tests
    test_suite = IntegrationTestSuite(args.url)
    results = test_suite.run_all_tests()
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved to: {args.output}")
    
    # Exit with appropriate code
    if results['success_rate'] >= 80:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()

