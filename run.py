from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
from newspaper import Article
import nltk
from collections import Counter
import logging
import sys
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter, Retry
import re
from anthropic import Anthropic
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s %(lineno)d',
    handlers=[
        logging.FileHandler('news_aggregator.log'),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger('TechNewsAggregator')

nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))
try:
    nltk.download('punkt', quiet=True, download_dir='nltk_data')
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

@dataclass
class NewsItem:
    title: str
    url: str
    content: str
    published_date: str
    discussion_score: float
    mention_count: int
    sources: list

    def __hash__(self):
        return hash(self.url)

class NewsletterGenerator:
    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)
        
    def generate_newsletter(self, newsletter_items):
        prompt = f"""You are a tech industry newsletter writer. Given the following news items, create a professional markdown newsletter.
            Focus on key insights and implications for tech industry professionals.
            Format should include:
            - A brief introduction
            - Individual sections for each article with clear headers
            - Key takeaways for each article
            - A conclusion summarizing the main themes

            News items:
            {json.dumps(newsletter_items, indent=2)}

            Create a markdown newsletter based on these items. Use professional language and highlight business implications.
        """

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.content[0].text

        except Exception as e:
            logger.error(f"Error generating newsletter with Claude: {str(e)}")
            return None

class TechNewsAggregator:
    def __init__(self):
        load_dotenv()
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.newsapi_key:
            logger.error("NEWS_API_KEY environment variable not found")
            raise ValueError("NEWS_API_KEY environment variable is required")
            
        if not self.anthropic_key:
            logger.error("ANTHROPIC_API_KEY environment variable not found")
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.hn_base_url = "https://hacker-news.firebaseio.com/v0"
        self.newsletter_generator = NewsletterGenerator(self.anthropic_key)
        
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.keywords = [
            'startup funding',
            'tech acquisition',
            'developer salary',
            'software outsourcing',
            'remote development',
            'engineering costs',
            'tech talent',
            'CTO',
            'technical debt',
            'development team',
        ]

        self.paywall_indicators = [
            'premium',
            'membership',
            'register to read',
            'sign up to read',
            'paid content',
            'premium content',
            'unlock this article',
            'subscribe now',
            'premium article',
            'member exclusive'
        ]

        logger.info(f"Initialized TechNewsAggregator with {len(self.keywords)} keywords")
        self.mention_tracker = Counter()
        self.url_tracker: Dict[str, NewsItem] = {}

    def is_paywalled(self, url: str, content: str = '') -> bool:
        if not content:
            try:
                response = self.session.get(url, timeout=10)
                content = response.text.lower()
            except Exception:
                return False

        paywall_pattern = '|'.join(self.paywall_indicators)
        if re.search(paywall_pattern, content.lower()):
            logging.info(f"Paywall detected for URL: {url}")
            return True

        return False

    def make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_news_api_content(self) -> List[NewsItem]:
        logger.info("Starting NewsAPI content fetch")
        base_url = "https://newsapi.org/v2/everything"
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        def process_keyword(keyword: str) -> None:
            logger.info(f"Fetching articles for keyword: {keyword}")
            params = {
                'q': keyword,
                'from': yesterday,
                'sortBy': 'relevancy',
                'language': 'en',
                'apiKey': self.newsapi_key,
                'pageSize': 10,
            }

            try:
                data = self.make_request(base_url, params)
                
                if data.get('articles'):
                    logger.info(f"Found {len(data['articles'])} articles for keyword '{keyword}'")
                    for article in data['articles']:
                        url = article.get('url')
                        if not url:
                            continue

                        if self.is_paywalled(url):
                            logger.info(f"Skipping paywalled content: {url}")
                            continue

                        self.mention_tracker[url] += 1

                        if url not in self.url_tracker:
                            logger.debug(f"Processing new article: {url}")
                            article_content = self.extract_article_content(url)
                            
                            if not article_content or self.is_paywalled(url, article_content):
                                logger.info(f"Skipping content with paywall indicators: {url}")
                                continue

                            news_item = NewsItem(
                                title=article.get('title', ''),
                                url=url,
                                content=article_content,
                                published_date=article.get('publishedAt', ''),
                                discussion_score=0,
                                mention_count=1,
                                sources=['NewsAPI'],
                            )
                            self.url_tracker[url] = news_item
                        else:
                            if 'NewsAPI' not in self.url_tracker[url].sources:
                                self.url_tracker[url].sources.append('NewsAPI')

            except Exception as e:
                logger.error(f"Error processing keyword '{keyword}': {str(e)}", exc_info=True)

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(process_keyword, self.keywords)

        logger.info(f"Completed NewsAPI fetch with {len(self.url_tracker)} unique articles")
        return list(self.url_tracker.values())

    def get_hackernews_content(self) -> List[NewsItem]:
        logger.info("Starting HackerNews content fetch")
        try:
            story_ids = self.make_request(f"{self.hn_base_url}/topstories.json")[:100]
            logger.info(f"Retrieved {len(story_ids)} top stories from HackerNews")

            def process_story(story_id: int) -> None:
                try:
                    story = self.make_request(f"{self.hn_base_url}/item/{story_id}.json")
                    url = story.get('url')
                    title = story.get('title', '').lower()

                    if url and any(keyword.lower() in title for keyword in self.keywords):
                        if self.is_paywalled(url):
                            logger.info(f"Skipping paywalled content: {url}")
                            return

                        logger.debug(f"Processing HN story {story_id}: {url}")
                        self.mention_tracker[url] += 1

                        if url not in self.url_tracker:
                            article_content = self.extract_article_content(url)
                            
                            if not article_content or self.is_paywalled(url, article_content):
                                logger.info(f"Skipping content with paywall indicators: {url}")
                                return

                            news_item = NewsItem(
                                title=story.get('title', ''),
                                url=url,
                                content=article_content,
                                published_date=datetime.fromtimestamp(
                                    story.get('time', 0)
                                ).isoformat(),
                                discussion_score=story.get('score', 0),
                                mention_count=1,
                                sources=['HackerNews'],
                            )
                            self.url_tracker[url] = news_item
                        else:
                            self.url_tracker[url].discussion_score += story.get('score', 0)
                            if 'HackerNews' not in self.url_tracker[url].sources:
                                self.url_tracker[url].sources.append('HackerNews')

                except Exception as e:
                    logger.error(f"Error processing HN story {story_id}: {str(e)}", exc_info=True)

            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(process_story, story_ids)

        except Exception:
            logger.error("Failed to fetch HackerNews content", exc_info=True)

        return list(self.url_tracker.values())

    def extract_article_content(self, url: str) -> str:
        logger.debug(f"Extracting content from: {url}")
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text
            return content if content else ""
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return ""

    def get_most_discussed_news(self, min_mentions: int = 2) -> List[NewsItem]:
        logger.info(f"Finding most discussed news (minimum {min_mentions} mentions)")

        try:
            self.get_news_api_content()
            self.get_hackernews_content()

            for url, item in self.url_tracker.items():
                item.mention_count = self.mention_tracker[url]

            discussed_news = [
                news for news in self.url_tracker.values()
                if news.mention_count >= min_mentions
            ]

            sorted_news = sorted(
                discussed_news,
                key=lambda x: (x.mention_count, x.discussion_score),
                reverse=True,
            )[:5]

            logger.info(f"Found {len(sorted_news)} articles with {min_mentions}+ mentions")
            return sorted_news

        except Exception:
            logger.error("Error in get_most_discussed_news", exc_info=True)
            return []

    def format_for_newsletter(self, news_items: List[NewsItem]) -> List[Dict]:
        logger.info("Formatting newsletter content")
        newsletter_content = []

        for item in news_items:
            try:
                summary = (
                    item.content[:200].strip() + "..."
                    if item.content
                    else "No content available"
                )

                formatted_item = {
                    "title": item.title,
                    "summary": summary,
                    "url": item.url,
                    "sources": item.sources,
                    "mention_count": item.mention_count,
                    "discussion_score": item.discussion_score,
                }
                newsletter_content.append(formatted_item)
            except Exception as e:
                logger.error(f"Error formatting news item: {str(e)}", exc_info=True)

        logger.info(f"Formatted {len(newsletter_content)} items for newsletter")
        return newsletter_content

    def generate_markdown_newsletter(self, news_items):
        logger.info("Generating markdown newsletter")
        try:
            newsletter_content = self.format_for_newsletter(news_items)
            markdown_content = self.newsletter_generator.generate_newsletter(newsletter_content)
            
            if markdown_content:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'tech_newsletter_{timestamp}.md'
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                logger.info(f"Newsletter saved to {filename}")
                return filename
            
            return None

        except Exception as e:
            logger.error(f"Error in generate_markdown_newsletter: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        logger.info("Starting news aggregation process")
        aggregator = TechNewsAggregator()
        discussed_news = aggregator.get_most_discussed_news()
        
        newsletter_file = aggregator.generate_markdown_newsletter(discussed_news)
        
        if newsletter_file:
            print(f"\n✨ Newsletter generated successfully! Check {newsletter_file}")
        else:
            print("\n❌ Failed to generate newsletter")
            
        logger.info("News aggregation process completed successfully")
    except Exception:
        logger.critical("Fatal error in main process", exc_info=True)
        sys.exit(1)

