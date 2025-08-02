#!/usr/bin/env python3
"""
Simplified AI Trading Agent Integration Test
"""

import requests
import time
import json
from datetime import datetime

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    api_url = f"{base_url}/api"
    
    test_results = []
    
    def log_test(name, success, message="", duration=0):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name} ({duration:.2f}s) - {message}")
        test_results.append({
            'test': name,
            'success': success,
            'message': message,
            'duration': duration
        })
    
    print("🚀 Starting AI Trading Agent Integration Tests")
    print("=" * 60)
    
    # Test 1: Health Check
    start = time.time()
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        duration = time.time() - start
        if response.status_code == 200 and response.json().get('status') == 'healthy':
            log_test("Health Check", True, "System is healthy", duration)
        else:
            log_test("Health Check", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Health Check", False, str(e), duration)
    
    # Test 2: Authentication
    start = time.time()
    try:
        response = requests.post(f"{api_url}/auth/login", 
                               json={'username': 'admin', 'password': 'admin123'}, 
                               timeout=10)
        duration = time.time() - start
        if response.status_code == 200 and response.json().get('success'):
            log_test("Authentication", True, "Login successful", duration)
        else:
            log_test("Authentication", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Authentication", False, str(e), duration)
    
    # Test 3: Market Sentiment
    start = time.time()
    try:
        response = requests.get(f"{api_url}/market/sentiment", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Market Sentiment", True, "Sentiment data retrieved", duration)
        else:
            log_test("Market Sentiment", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Market Sentiment", False, str(e), duration)
    
    # Test 4: Options Chain
    start = time.time()
    try:
        response = requests.get(f"{api_url}/market/options-chain/NIFTY", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Options Chain", True, "Options data retrieved", duration)
        else:
            log_test("Options Chain", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Options Chain", False, str(e), duration)
    
    # Test 5: Trading Signal
    start = time.time()
    try:
        response = requests.post(f"{api_url}/signals/generate", 
                               json={'symbol': 'NIFTY24JAN20000CE'}, 
                               timeout=10)
        duration = time.time() - start
        if response.status_code == 200 and 'signal' in response.json():
            log_test("Trading Signals", True, f"Signal: {response.json().get('signal')}", duration)
        else:
            log_test("Trading Signals", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Trading Signals", False, str(e), duration)
    
    # Test 6: Batch Signals
    start = time.time()
    try:
        response = requests.post(f"{api_url}/signals/batch", 
                               json={'symbols': ['NIFTY24JAN20000CE', 'BANKNIFTY24JAN45500PE']}, 
                               timeout=15)
        duration = time.time() - start
        if response.status_code == 200 and 'signals' in response.json():
            log_test("Batch Signals", True, f"Generated {len(response.json()['signals'])} signals", duration)
        else:
            log_test("Batch Signals", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Batch Signals", False, str(e), duration)
    
    # Test 7: Risk Assessment
    start = time.time()
    try:
        response = requests.get(f"{api_url}/risk/assessment", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Risk Assessment", True, "Risk data retrieved", duration)
        else:
            log_test("Risk Assessment", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Risk Assessment", False, str(e), duration)
    
    # Test 8: Portfolio Positions
    start = time.time()
    try:
        response = requests.get(f"{api_url}/portfolio/positions", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Portfolio Positions", True, "Position data retrieved", duration)
        else:
            log_test("Portfolio Positions", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Portfolio Positions", False, str(e), duration)
    
    # Test 9: Order Management
    start = time.time()
    try:
        response = requests.get(f"{api_url}/orders", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Order Management", True, "Order data retrieved", duration)
        else:
            log_test("Order Management", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Order Management", False, str(e), duration)
    
    # Test 10: System Status
    start = time.time()
    try:
        response = requests.get(f"{api_url}/system/status", timeout=10)
        duration = time.time() - start
        if response.status_code == 200 and 'overall_status' in response.json():
            status = response.json().get('overall_status')
            log_test("System Status", True, f"Status: {status}", duration)
        else:
            log_test("System Status", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("System Status", False, str(e), duration)
    
    # Test 11: Model Performance
    start = time.time()
    try:
        response = requests.get(f"{api_url}/models/performance", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Model Performance", True, "Performance data retrieved", duration)
        else:
            log_test("Model Performance", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Model Performance", False, str(e), duration)
    
    # Test 12: Analytics
    start = time.time()
    try:
        response = requests.get(f"{api_url}/analytics/performance", timeout=10)
        duration = time.time() - start
        if response.status_code == 200:
            log_test("Analytics", True, "Analytics data retrieved", duration)
        else:
            log_test("Analytics", False, f"HTTP {response.status_code}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Analytics", False, str(e), duration)
    
    # Test 13: Concurrent Requests
    start = time.time()
    try:
        import concurrent.futures
        
        def make_request():
            return requests.get(f"{api_url}/system/status", timeout=5)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    response = future.result()
                    results.append(response.status_code)
                except:
                    results.append(0)
        
        duration = time.time() - start
        success_count = sum(1 for status in results if status == 200)
        success_rate = success_count / len(results)
        
        if success_rate >= 0.8:
            log_test("Concurrent Load", True, f"Success rate: {success_rate:.1%}", duration)
        else:
            log_test("Concurrent Load", False, f"Low success rate: {success_rate:.1%}", duration)
    except Exception as e:
        duration = time.time() - start
        log_test("Concurrent Load", False, str(e), duration)
    
    # Calculate summary
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['success'])
    success_rate = (passed_tests / total_tests) * 100
    
    print("=" * 60)
    print("🏁 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT! System is production-ready!")
    elif success_rate >= 80:
        print("✅ GOOD! System is mostly functional with minor issues.")
    elif success_rate >= 70:
        print("⚠️  FAIR! System has some issues that need attention.")
    else:
        print("❌ POOR! System has significant issues that must be fixed.")
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump({
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat(),
            'test_results': test_results
        }, f, indent=2)
    
    print(f"\nTest results saved to: test_results.json")
    return success_rate >= 80

if __name__ == "__main__":
    success = test_api_endpoints()
    exit(0 if success else 1)

