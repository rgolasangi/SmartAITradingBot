"""
News Data Collection Module for AI Trading Agent
"""
import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
from newsapi import NewsApiClient
import feedparser
from bs4 import BeautifulSoup
import json
import redis
from dotenv import load_dotenv

load_dotenv()

class NewsCollector:
    """Collect financial news from multiple sources"""
    
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.news_api_client = NewsApiClient(api_key=self.news_api_key) if self.news_api_key else None
        
        # Redis for caching and deduplication
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # RSS feeds for financial news
        self.rss_feeds = [
            {
                'name': 'Economic Times Markets',
                'url': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                'category': 'markets'
            },
            {
                'name': 'Moneycontrol Markets',
                'url': 'https://www.moneycontrol.com/rss/marketstories.xml',
                'category': 'markets'
            },
            {
                'name': 'Business Standard Markets',
                'url': 'https://www.business-standard.com/rss/markets-106.rss',
                'category': 'markets'
            },
            {
                'name': 'Livemint Markets',
                'url': 'https://www.livemint.com/rss/markets',
                'category': 'markets'
            },
            {
                'name': 'Financial Express Markets',
                'url': 'https://www.financialexpress.com/market/rss',
                'category': 'markets'
            }
        ]
        
        # Keywords for relevance filtering
        self.relevant_keywords = [
            'nifty', 'bank nifty', 'sensex', 'bse', 'nse',
            'options', 'derivatives', 'futures', 'volatility',
            'market', 'trading', 'stocks', 'equity', 'index',
            'rbi', 'monetary policy', 'interest rates', 'inflation',
            'gdp', 'economic', 'finance', 'banking', 'fii', 'dii'
        ]
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_duplicate(self, url: str) -> bool:
        """Check if article URL has been processed before"""
        cache_key = f"news_processed:{url}"
        return self.redis_client.exists(cache_key)
    
    def _mark_processed(self, url: str, ttl: int = 86400 * 7):  # 7 days
        """Mark article as processed"""
        cache_key = f"news_processed:{url}"
        self.redis_client.setex(cache_key, ttl, "1")
    
    def _calculate_relevance_score(self, title: str, content: str) -> float:
        """Calculate relevance score based on keywords"""
        text = f"{title} {content}".lower()
        
        keyword_matches = 0
        total_keywords = len(self.relevant_keywords)
        
        for keyword in self.relevant_keywords:
            if keyword in text:
                keyword_matches += 1
        
        # Base relevance score
        relevance_score = keyword_matches / total_keywords
        
        # Boost for specific high-value keywords
        high_value_keywords = ['nifty', 'bank nifty', 'options', 'derivatives', 'volatility']
        for keyword in high_value_keywords:
            if keyword in text:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _extract_mentioned_symbols(self, title: str, content: str) -> List[str]:
        """Extract mentioned stock symbols and indices"""
        text = f"{title} {content}".lower()
        mentioned_symbols = []
        
        # Common indices and symbols
        symbols_to_check = [
            'nifty', 'nifty 50', 'bank nifty', 'sensex', 'bse',
            'reliance', 'tcs', 'hdfc', 'icici', 'sbi', 'infosys',
            'wipro', 'bharti airtel', 'itc', 'hul', 'bajaj'
        ]
        
        for symbol in symbols_to_check:
            if symbol in text:
                mentioned_symbols.append(symbol.upper())
        
        return list(set(mentioned_symbols))  # Remove duplicates
    
    def collect_from_newsapi(self, 
                           query: str = "nifty OR \"bank nifty\" OR sensex OR \"stock market\"",
                           hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect news from News API"""
        if not self.news_api_client:
            self.logger.warning("News API client not configured")
            return []
        
        try:
            self._rate_limit()
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(hours=hours_back)
            
            # Fetch articles
            articles = self.news_api_client.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                from_param=from_date.isoformat(),
                to=to_date.isoformat(),
                page_size=100
            )
            
            collected_articles = []
            
            for article in articles.get('articles', []):
                # Skip if already processed
                if self._is_duplicate(article['url']):
                    continue
                
                # Extract content
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                # Calculate relevance
                relevance_score = self._calculate_relevance_score(title, f"{description} {content}")
                
                # Skip low relevance articles
                if relevance_score < 0.1:
                    continue
                
                # Extract mentioned symbols
                mentioned_symbols = self._extract_mentioned_symbols(title, f"{description} {content}")
                
                article_data = {
                    'title': title,
                    'content': f"{description} {content}",
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article['url'],
                    'author': article.get('author'),
                    'published_at': datetime.fromisoformat(
                        article['publishedAt'].replace('Z', '+00:00')
                    ),
                    'relevance_score': relevance_score,
                    'mentioned_symbols': json.dumps(mentioned_symbols)
                }
                
                collected_articles.append(article_data)
                self._mark_processed(article['url'])
            
            self.logger.info(f"Collected {len(collected_articles)} articles from News API")
            return collected_articles
            
        except Exception as e:
            self.logger.error(f"Error collecting from News API: {e}")
            return []
    
    def collect_from_rss(self) -> List[Dict[str, Any]]:
        """Collect news from RSS feeds"""
        collected_articles = []
        
        for feed_config in self.rss_feeds:
            try:
                self._rate_limit()
                
                self.logger.info(f"Fetching from {feed_config['name']}")
                
                # Parse RSS feed
                feed = feedparser.parse(feed_config['url'])
                
                for entry in feed.entries:
                    # Skip if already processed
                    if self._is_duplicate(entry.link):
                        continue
                    
                    # Extract content
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    
                    # Try to get full content
                    content = self._extract_full_content(entry.link)
                    if not content:
                        content = summary
                    
                    # Calculate relevance
                    relevance_score = self._calculate_relevance_score(title, content)
                    
                    # Skip low relevance articles
                    if relevance_score < 0.1:
                        continue
                    
                    # Extract mentioned symbols
                    mentioned_symbols = self._extract_mentioned_symbols(title, content)
                    
                    # Parse published date
                    published_at = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_at = datetime(*entry.published_parsed[:6])
                    
                    article_data = {
                        'title': title,
                        'content': content,
                        'source': feed_config['name'],
                        'url': entry.link,
                        'author': entry.get('author'),
                        'published_at': published_at,
                        'relevance_score': relevance_score,
                        'mentioned_symbols': json.dumps(mentioned_symbols)
                    }
                    
                    collected_articles.append(article_data)
                    self._mark_processed(entry.link)
                
            except Exception as e:
                self.logger.error(f"Error collecting from {feed_config['name']}: {e}")
                continue
        
        self.logger.info(f"Collected {len(collected_articles)} articles from RSS feeds")
        return collected_articles
    
    def _extract_full_content(self, url: str) -> Optional[str]:
        """Extract full article content from URL"""
        try:
            self._rate_limit()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try common content selectors
            content_selectors = [
                'article',
                '.article-content',
                '.story-content',
                '.post-content',
                '.entry-content',
                '#content',
                '.content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            # Fallback to body text
            if not content:
                content = soup.get_text()
            
            # Clean up content
            content = ' '.join(content.split())  # Remove extra whitespace
            
            return content[:5000] if content else None  # Limit content length
            
        except Exception as e:
            self.logger.debug(f"Error extracting content from {url}: {e}")
            return None
    
    def collect_all_news(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect news from all sources"""
        all_articles = []
        
        # Collect from News API
        if self.news_api_client:
            newsapi_articles = self.collect_from_newsapi(hours_back=hours_back)
            all_articles.extend(newsapi_articles)
        
        # Collect from RSS feeds
        rss_articles = self.collect_from_rss()
        all_articles.extend(rss_articles)
        
        # Sort by relevance score and published date
        all_articles.sort(
            key=lambda x: (x['relevance_score'], x['published_at']),
            reverse=True
        )
        
        self.logger.info(f"Total collected articles: {len(all_articles)}")
        return all_articles
    
    def get_market_news_summary(self, hours_back: int = 6) -> Dict[str, Any]:
        """Get summarized market news for the specified period"""
        articles = self.collect_all_news(hours_back=hours_back)
        
        # Filter high relevance articles
        high_relevance_articles = [
            article for article in articles 
            if article['relevance_score'] >= 0.3
        ]
        
        # Categorize by mentioned symbols
        symbol_mentions = {}
        for article in high_relevance_articles:
            symbols = json.loads(article['mentioned_symbols'])
            for symbol in symbols:
                if symbol not in symbol_mentions:
                    symbol_mentions[symbol] = []
                symbol_mentions[symbol].append(article)
        
        summary = {
            'total_articles': len(articles),
            'high_relevance_articles': len(high_relevance_articles),
            'symbol_mentions': symbol_mentions,
            'top_articles': high_relevance_articles[:10],
            'collection_timestamp': datetime.now().isoformat()
        }
        
        return summary

