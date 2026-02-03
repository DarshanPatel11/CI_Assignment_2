"""
Wikipedia Data Collection Module

Collects Wikipedia articles:
- 200 fixed URLs (stored in fixed_urls.json)
- 300 random URLs (sampled each run)
"""

import json
import random
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import wikipediaapi
from tqdm import tqdm


@dataclass
class WikiArticle:
    """Represents a Wikipedia article with metadata."""
    url: str
    title: str
    content: str
    word_count: int
    url_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


class WikipediaCollector:
    """Collects and processes Wikipedia articles."""

    # Diverse topic categories for fixed URL collection
    TOPIC_CATEGORIES = [
        "Science", "Technology", "History", "Geography", "Arts",
        "Philosophy", "Literature", "Mathematics", "Biology", "Physics",
        "Chemistry", "Medicine", "Economics", "Politics", "Sports",
        "Music", "Film", "Architecture", "Psychology", "Sociology"
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_urls_path = self.data_dir / "fixed_urls.json"

        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='HybridRAG/1.0 (Educational Project)',
            language='en'
        )

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HybridRAG/1.0 (Educational Project)'
        })

    def _get_url_hash(self, url: str) -> str:
        """Generate unique hash for URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _extract_article_content(self, url: str) -> Optional[WikiArticle]:
        """Extract clean text content from Wikipedia URL."""
        try:
            # Extract page title from URL
            title = url.split("/wiki/")[-1].replace("_", " ")

            # Get page via Wikipedia API
            page = self.wiki.page(title)

            if not page.exists():
                return None

            content = page.text
            word_count = len(content.split())

            # Filter: minimum 200 words
            if word_count < 200:
                return None

            return WikiArticle(
                url=url,
                title=page.title,
                content=content,
                word_count=word_count,
                url_hash=self._get_url_hash(url)
            )

        except Exception as e:
            print(f"Error extracting {url}: {e}")
            return None

    def _get_random_wikipedia_urls(self, count: int) -> List[str]:
        """Get random Wikipedia article URLs using the API."""
        urls = []
        api_url = "https://en.wikipedia.org/w/api.php"

        while len(urls) < count:
            try:
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": 0,  # Main articles only
                    "rnlimit": min(50, count - len(urls) + 20)  # Extra buffer for filtering
                }

                response = self.session.get(api_url, params=params)
                data = response.json()

                for item in data.get("query", {}).get("random", []):
                    title = item["title"]
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

                    # Verify article meets word count requirement
                    article = self._extract_article_content(url)
                    if article and article.word_count >= 200:
                        urls.append(url)
                        if len(urls) >= count:
                            break

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"Error fetching random URLs: {e}")
                time.sleep(1)

        return urls[:count]

    def _get_category_articles(self, category: str, limit: int = 20) -> List[str]:
        """Get article URLs from a specific category."""
        urls = []
        api_url = "https://en.wikipedia.org/w/api.php"

        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmtype": "page",
                "cmlimit": limit * 2  # Buffer for filtering
            }

            response = self.session.get(api_url, params=params)
            data = response.json()

            for item in data.get("query", {}).get("categorymembers", []):
                title = item["title"]
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                urls.append(url)
                if len(urls) >= limit:
                    break

        except Exception as e:
            print(f"Error fetching category {category}: {e}")

        return urls

    def generate_fixed_urls(self, count: int = 200, force: bool = False) -> List[str]:
        """
        Generate and store 200 fixed Wikipedia URLs.
        Samples from diverse categories for topic coverage.
        """
        if self.fixed_urls_path.exists() and not force:
            print(f"Fixed URLs already exist at {self.fixed_urls_path}")
            return self.load_fixed_urls()

        print(f"Generating {count} fixed diverse Wikipedia URLs...")

        all_urls = []
        urls_per_category = count // len(self.TOPIC_CATEGORIES) + 1

        # Collect from each category
        for category in tqdm(self.TOPIC_CATEGORIES, desc="Collecting from categories"):
            category_urls = self._get_category_articles(category, urls_per_category)
            all_urls.extend(category_urls)
            time.sleep(0.3)

        # Deduplicate and validate
        unique_urls = list(set(all_urls))
        valid_urls = []

        print("Validating URLs (checking word count >= 200)...")
        for url in tqdm(unique_urls, desc="Validating"):
            article = self._extract_article_content(url)
            if article:
                valid_urls.append(url)
            if len(valid_urls) >= count:
                break
            time.sleep(0.2)

        # If not enough, supplement with random articles
        if len(valid_urls) < count:
            print(f"Need {count - len(valid_urls)} more URLs, fetching random...")
            additional = self._get_random_wikipedia_urls(count - len(valid_urls))
            valid_urls.extend(additional)

        valid_urls = valid_urls[:count]

        # Save to JSON
        with open(self.fixed_urls_path, 'w') as f:
            json.dump({
                "description": "200 fixed Wikipedia URLs for Hybrid RAG System",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(valid_urls),
                "urls": valid_urls
            }, f, indent=2)

        print(f"Saved {len(valid_urls)} fixed URLs to {self.fixed_urls_path}")
        return valid_urls

    def load_fixed_urls(self) -> List[str]:
        """Load fixed URLs from JSON file."""
        if not self.fixed_urls_path.exists():
            raise FileNotFoundError(f"Fixed URLs not found at {self.fixed_urls_path}")

        with open(self.fixed_urls_path, 'r') as f:
            data = json.load(f)

        return data["urls"]

    def get_random_urls(self, count: int = 300, exclude: List[str] = None) -> List[str]:
        """
        Get random Wikipedia URLs for each indexing run.
        Excludes any URLs in the exclude list (e.g., fixed URLs).
        """
        exclude = set(exclude or [])

        print(f"Fetching {count} random Wikipedia URLs...")
        urls = []

        while len(urls) < count:
            batch = self._get_random_wikipedia_urls(min(50, count - len(urls) + 10))
            for url in batch:
                if url not in exclude and url not in urls:
                    urls.append(url)
                if len(urls) >= count:
                    break

        return urls[:count]

    def collect_all_articles(
        self,
        fixed_count: int = 200,
        random_count: int = 300
    ) -> List[WikiArticle]:
        """
        Collect all articles: fixed + random URLs.
        Returns list of WikiArticle objects with full content.
        """
        # Get fixed URLs (generate if needed)
        fixed_urls_count = 0
        if self.fixed_urls_path.exists():
            fixed_urls_count = len(self.load_fixed_urls())
        if fixed_urls_count < fixed_count:
            fixed_urls = self.load_fixed_urls()
        else:
            fixed_urls = self.generate_fixed_urls(fixed_count)

        # Get random URLs (different each run)
        random_urls = self.get_random_urls(random_count, exclude=fixed_urls)

        all_urls = fixed_urls + random_urls
        print(f"Total URLs to process: {len(all_urls)}")

        # Extract content from all URLs
        articles = []

        print("Extracting article content...")
        for url in tqdm(all_urls, desc="Extracting"):
            article = self._extract_article_content(url)
            if article:
                articles.append(article)
            time.sleep(0.1)  # Rate limiting

        print(f"Successfully extracted {len(articles)} articles")
        return articles

    def save_corpus(self, articles: List[WikiArticle], filename: str = "corpus.json"):
        """Save collected articles to JSON."""
        corpus_path = self.data_dir / "corpus" / filename
        corpus_path.parent.mkdir(parents=True, exist_ok=True)

        with open(corpus_path, 'w') as f:
            json.dump({
                "count": len(articles),
                "articles": [a.to_dict() for a in articles]
            }, f, indent=2)

        print(f"Saved corpus to {corpus_path}")
        return corpus_path


if __name__ == "__main__":
    # Test data collection
    collector = WikipediaCollector()

    # Generate fixed URLs
    fixed = collector.generate_fixed_urls(200)
    print(f"Fixed URLs: {len(fixed)}")

    # Collect sample articles
    articles = collector.collect_all_articles(fixed_count=5, random_count=5)
    print(f"Sample articles: {len(articles)}")

    for article in articles[:3]:
        print(f"  - {article.title}: {article.word_count} words")
