import time
from datetime import datetime

import feedparser
from pydantic import BaseModel


class NewsArticleContract(BaseModel):
    source: str
    headline: str
    published_at: datetime
    summary: str | None = None
    url: str


class RSSIngestionEngine:
    def __init__(self, feeds: list[str] = None):
        self.feeds = feeds or [
            "https://news.google.com/rss/search?q=NSE+India+stock+market",
            "https://www.moneycontrol.com/rss/business.xml",
        ]
        # Track seen URLs for basic deduplication
        self._seen_urls: set[str] = set()

    def fetch_news(self) -> list[NewsArticleContract]:
        articles = []
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    try:
                        url = getattr(entry, "link", "")
                        # Skip duplicates
                        if url in self._seen_urls:
                            continue
                        self._seen_urls.add(url)

                        # Enforce data contract
                        # some feeds use published_parsed
                        pub_time = time.gmtime()
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            pub_time = entry.published_parsed

                        article = NewsArticleContract(
                            source=feed_url,
                            headline=entry.title,
                            published_at=datetime(*pub_time[:6]),
                            summary=getattr(entry, "summary", ""),
                            url=url,
                        )
                        articles.append(article)
                    except Exception as e:
                        print(f"Skipping malformed article from {feed_url}: {e}")
            except Exception as e:
                print(f"Failed to fetch feed {feed_url}: {e}")

        return articles


if __name__ == "__main__":
    engine = RSSIngestionEngine()
    news = engine.fetch_news()
    print(f"Fetched {len(news)} articles successfully matching the data contract.")
