"""
Social Sentiment Data Collection Module for AI Trading Agent
"""
import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import tweepy
import praw
import requests
import json
import redis
from dotenv import load_dotenv

load_dotenv()

class SentimentCollector:
    """Collect social sentiment data from multiple platforms"""
    
    def __init__(self):
        # Twitter API credentials
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        # Initialize Twitter client
        self.twitter_client = None
        if self.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.twitter_bearer_token,
                    consumer_key=self.twitter_api_key,
                    consumer_secret=self.twitter_api_secret,
                    access_token=self.twitter_access_token,
                    access_token_secret=self.twitter_access_token_secret,
                    wait_on_rate_limit=True
                )
            except Exception as e:
                logging.error(f"Error initializing Twitter client: {e}")
        
        # Redis for caching and deduplication
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(redis_url)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Search queries for different platforms
        self.twitter_queries = [
            "nifty OR \"nifty 50\" OR \"bank nifty\"",
            "sensex OR bse OR nse",
            "options trading OR derivatives",
            "stock market india OR indian stocks",
            "volatility OR vix india"
        ]
        
        # Keywords for relevance filtering
        self.relevant_keywords = [
            'nifty', 'bank nifty', 'sensex', 'bse', 'nse',
            'options', 'calls', 'puts', 'strike', 'expiry',
            'volatility', 'vix', 'premium', 'theta', 'delta',
            'bullish', 'bearish', 'rally', 'crash', 'correction',
            'resistance', 'support', 'breakout', 'breakdown'
        ]
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_duplicate(self, post_id: str, platform: str) -> bool:
        """Check if post has been processed before"""
        cache_key = f"sentiment_processed:{platform}:{post_id}"
        return self.redis_client.exists(cache_key)
    
    def _mark_processed(self, post_id: str, platform: str, ttl: int = 86400 * 7):
        """Mark post as processed"""
        cache_key = f"sentiment_processed:{platform}:{post_id}"
        self.redis_client.setex(cache_key, ttl, "1")
    
    def _calculate_relevance_score(self, content: str) -> float:
        """Calculate relevance score based on keywords"""
        content_lower = content.lower()
        
        keyword_matches = 0
        total_keywords = len(self.relevant_keywords)
        
        for keyword in self.relevant_keywords:
            if keyword in content_lower:
                keyword_matches += 1
        
        # Base relevance score
        relevance_score = keyword_matches / total_keywords
        
        # Boost for specific high-value keywords
        high_value_keywords = ['nifty', 'bank nifty', 'options', 'calls', 'puts']
        for keyword in high_value_keywords:
            if keyword in content_lower:
                relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _extract_mentioned_symbols(self, content: str) -> List[str]:
        """Extract mentioned symbols from content"""
        content_lower = content.lower()
        mentioned_symbols = []
        
        # Common symbols and indices
        symbols_to_check = [
            'nifty', 'nifty 50', 'bank nifty', 'sensex',
            'reliance', 'tcs', 'hdfc', 'icici', 'sbi',
            'infosys', 'wipro', 'bharti', 'itc'
        ]
        
        for symbol in symbols_to_check:
            if symbol in content_lower:
                mentioned_symbols.append(symbol.upper())
        
        return list(set(mentioned_symbols))
    
    def _calculate_influence_score(self, 
                                 followers_count: int,
                                 likes_count: int,
                                 shares_count: int) -> float:
        """Calculate influence score based on engagement metrics"""
        # Normalize followers count (log scale)
        followers_score = min(followers_count / 100000, 1.0)  # Max at 100k followers
        
        # Engagement score
        total_engagement = likes_count + shares_count
        engagement_score = min(total_engagement / 1000, 1.0)  # Max at 1k engagement
        
        # Weighted influence score
        influence_score = (followers_score * 0.6) + (engagement_score * 0.4)
        
        return influence_score
    
    def collect_twitter_sentiment(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect sentiment data from Twitter"""
        if not self.twitter_client:
            self.logger.warning("Twitter client not configured")
            return []
        
        collected_posts = []
        
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            for query in self.twitter_queries:
                self._rate_limit()
                
                try:
                    # Search tweets
                    tweets = tweepy.Paginator(
                        self.twitter_client.search_recent_tweets,
                        query=query,
                        tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                        user_fields=['followers_count', 'verified'],
                        expansions=['author_id'],
                        start_time=start_time,
                        end_time=end_time,
                        max_results=100
                    ).flatten(limit=500)
                    
                    for tweet in tweets:
                        # Skip if already processed
                        if self._is_duplicate(str(tweet.id), 'twitter'):
                            continue
                        
                        # Get tweet content
                        content = tweet.text
                        
                        # Calculate relevance
                        relevance_score = self._calculate_relevance_score(content)
                        
                        # Skip low relevance tweets
                        if relevance_score < 0.1:
                            continue
                        
                        # Get user info
                        user_info = None
                        if hasattr(tweet, 'includes') and 'users' in tweet.includes:
                            user_info = tweet.includes['users'][0]
                        
                        followers_count = user_info.followers_count if user_info else 0
                        
                        # Get engagement metrics
                        metrics = tweet.public_metrics
                        likes_count = metrics.get('like_count', 0)
                        retweet_count = metrics.get('retweet_count', 0)
                        reply_count = metrics.get('reply_count', 0)
                        
                        # Calculate influence score
                        influence_score = self._calculate_influence_score(
                            followers_count, likes_count, retweet_count + reply_count
                        )
                        
                        # Extract mentioned symbols
                        mentioned_symbols = self._extract_mentioned_symbols(content)
                        
                        post_data = {
                            'platform': 'twitter',
                            'post_id': str(tweet.id),
                            'content': content,
                            'author': user_info.username if user_info else 'unknown',
                            'followers_count': followers_count,
                            'likes_count': likes_count,
                            'shares_count': retweet_count,
                            'posted_at': tweet.created_at,
                            'relevance_score': relevance_score,
                            'influence_score': influence_score,
                            'mentioned_symbols': json.dumps(mentioned_symbols)
                        }
                        
                        collected_posts.append(post_data)
                        self._mark_processed(str(tweet.id), 'twitter')
                
                except Exception as e:
                    self.logger.error(f"Error collecting tweets for query '{query}': {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in Twitter sentiment collection: {e}")
        
        self.logger.info(f"Collected {len(collected_posts)} Twitter posts")
        return collected_posts
    
    def collect_reddit_sentiment(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect sentiment data from Reddit (using public API)"""
        collected_posts = []
        
        try:
            # Reddit subreddits related to Indian stock market
            subreddits = [
                'IndiaInvestments',
                'IndianStreetBets',
                'SecurityAnalysis',
                'StockMarket'
            ]
            
            for subreddit in subreddits:
                try:
                    self._rate_limit()
                    
                    # Use Reddit's JSON API (no authentication required)
                    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=100"
                    headers = {'User-Agent': 'TradingBot/1.0'}
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    for post in data['data']['children']:
                        post_data_raw = post['data']
                        
                        # Skip if already processed
                        if self._is_duplicate(post_data_raw['id'], 'reddit'):
                            continue
                        
                        # Check if post is recent enough
                        post_time = datetime.fromtimestamp(post_data_raw['created_utc'])
                        if post_time < datetime.now() - timedelta(hours=hours_back):
                            continue
                        
                        # Get content
                        title = post_data_raw.get('title', '')
                        selftext = post_data_raw.get('selftext', '')
                        content = f"{title} {selftext}"
                        
                        # Calculate relevance
                        relevance_score = self._calculate_relevance_score(content)
                        
                        # Skip low relevance posts
                        if relevance_score < 0.1:
                            continue
                        
                        # Get engagement metrics
                        upvotes = post_data_raw.get('ups', 0)
                        comments = post_data_raw.get('num_comments', 0)
                        
                        # Calculate influence score (Reddit doesn't have follower count)
                        influence_score = min((upvotes + comments) / 100, 1.0)
                        
                        # Extract mentioned symbols
                        mentioned_symbols = self._extract_mentioned_symbols(content)
                        
                        reddit_post_data = {
                            'platform': 'reddit',
                            'post_id': post_data_raw['id'],
                            'content': content,
                            'author': post_data_raw.get('author', 'unknown'),
                            'followers_count': 0,  # Not available for Reddit
                            'likes_count': upvotes,
                            'shares_count': comments,
                            'posted_at': post_time,
                            'relevance_score': relevance_score,
                            'influence_score': influence_score,
                            'mentioned_symbols': json.dumps(mentioned_symbols)
                        }
                        
                        collected_posts.append(reddit_post_data)
                        self._mark_processed(post_data_raw['id'], 'reddit')
                
                except Exception as e:
                    self.logger.error(f"Error collecting from r/{subreddit}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error in Reddit sentiment collection: {e}")
        
        self.logger.info(f"Collected {len(collected_posts)} Reddit posts")
        return collected_posts
    
    def collect_all_sentiment(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Collect sentiment data from all platforms"""
        all_posts = []
        
        # Collect from Twitter
        twitter_posts = self.collect_twitter_sentiment(hours_back=hours_back)
        all_posts.extend(twitter_posts)
        
        # Collect from Reddit
        reddit_posts = self.collect_reddit_sentiment(hours_back=hours_back)
        all_posts.extend(reddit_posts)
        
        # Sort by relevance and influence scores
        all_posts.sort(
            key=lambda x: (x['relevance_score'] * x['influence_score']),
            reverse=True
        )
        
        self.logger.info(f"Total collected social posts: {len(all_posts)}")
        return all_posts
    
    def get_sentiment_summary(self, hours_back: int = 6) -> Dict[str, Any]:
        """Get summarized sentiment data for the specified period"""
        posts = self.collect_all_sentiment(hours_back=hours_back)
        
        # Filter high relevance posts
        high_relevance_posts = [
            post for post in posts 
            if post['relevance_score'] >= 0.3
        ]
        
        # Categorize by platform
        platform_breakdown = {}
        for post in high_relevance_posts:
            platform = post['platform']
            if platform not in platform_breakdown:
                platform_breakdown[platform] = []
            platform_breakdown[platform].append(post)
        
        # Symbol mentions analysis
        symbol_mentions = {}
        for post in high_relevance_posts:
            symbols = json.loads(post['mentioned_symbols'])
            for symbol in symbols:
                if symbol not in symbol_mentions:
                    symbol_mentions[symbol] = {
                        'count': 0,
                        'total_influence': 0,
                        'posts': []
                    }
                symbol_mentions[symbol]['count'] += 1
                symbol_mentions[symbol]['total_influence'] += post['influence_score']
                symbol_mentions[symbol]['posts'].append(post)
        
        # Calculate average influence per symbol
        for symbol in symbol_mentions:
            count = symbol_mentions[symbol]['count']
            symbol_mentions[symbol]['avg_influence'] = (
                symbol_mentions[symbol]['total_influence'] / count if count > 0 else 0
            )
        
        summary = {
            'total_posts': len(posts),
            'high_relevance_posts': len(high_relevance_posts),
            'platform_breakdown': {
                platform: len(posts) for platform, posts in platform_breakdown.items()
            },
            'symbol_mentions': symbol_mentions,
            'top_posts': high_relevance_posts[:20],
            'collection_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def get_real_time_sentiment_stream(self):
        """Get real-time sentiment stream (placeholder for WebSocket implementation)"""
        # This would implement real-time streaming from Twitter API v2
        # For now, return a placeholder
        self.logger.info("Real-time sentiment streaming not implemented yet")
        return None

