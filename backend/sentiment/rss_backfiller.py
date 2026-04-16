"""
One-time RSS Backfiller for missing sentiment data.
Uses Google News RSS to fetch historical headlines for dates
where the ET/Moneycontrol archive scraper returned 0 results.
"""
import feedparser
import hashlib
import pandas as pd
import time
import random
import logging
import sqlite3
from datetime import datetime, timedelta
from urllib.parse import quote
from backend.sentiment.news_aggregator import RELEVANT_KEYWORDS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rss_backfill.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RSSBackfiller:
    """
    Fetches historical news headlines via RSS feeds for dates missing
    from the ET archive. Writes directly to SQLite.
    """

    GOOGLE_NEWS_QUERIES = [
        "India stock market nifty sensex",
        "RBI monetary policy interest rate",
        "India economy GDP inflation",
        "Indian rupee forex trade",
    ]

    # Static RSS feeds (these return "latest" items, useful for recent gaps)
    STATIC_FEEDS = [
        ("LiveMint Markets", "https://www.livemint.com/rss/markets"),
        ("LiveMint Economy", "https://www.livemint.com/rss/economy"),
        ("ET Markets RSS", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
        ("MC Latest", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ]

    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def _get_md5(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _is_relevant(self, headline: str) -> bool:
        text_lower = headline.lower()
        return any(kw in text_lower for kw in RELEVANT_KEYWORDS)

    def _fetch_google_news_rss(self, query: str, date_obj: datetime) -> list:
        """
        Fetch Google News RSS for a specific query scoped to a single day.
        Google News RSS supports 'after:' and 'before:' date operators.
        """
        date_str = date_obj.strftime("%Y-%m-%d")
        next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

        # Google News RSS date-scoped query
        full_query = f"{query} after:{date_str} before:{next_day}"
        encoded_query = quote(full_query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        headlines = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                pub_date = entry.get("published", "")

                if title and self._is_relevant(title):
                    headlines.append({
                        "timestamp": date_obj.strftime("%Y-%m-%d 10:00:00"),
                        "source": "Google News RSS",
                        "headline": title,
                        "sentiment_score": None,
                        "link": link,
                        "md5_hash": self._get_md5(title),
                        "sector": "markets",  # Default; refined later if needed
                        "is_pre_market": 1,
                    })
        except Exception as e:
            logger.error(f"Error fetching Google News RSS for '{query}' on {date_str}: {e}")

        return headlines

    def _save_to_db(self, headlines: list):
        """Insert headlines into SQLite, skipping duplicates via md5_hash."""
        if not headlines:
            return 0

        cursor = self.conn.cursor()
        inserted = 0
        for h in headlines:
            try:
                cursor.execute(
                    """INSERT INTO news_sentiment 
                       (timestamp, source, headline, sentiment_score, link, md5_hash, sector, is_pre_market)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (h["timestamp"], h["source"], h["headline"], h["sentiment_score"],
                     h["link"], h["md5_hash"], h["sector"], h["is_pre_market"])
                )
                inserted += 1
            except sqlite3.IntegrityError:
                continue  # Duplicate md5_hash
            except Exception as e:
                logger.error(f"DB insert error: {e}")
        self.conn.commit()
        return inserted

    def get_missing_dates(self) -> list:
        """
        Find weekdays in the last year that have 0 headlines in the DB.
        Returns a list of datetime objects.
        """
        cursor = self.conn.cursor()

        # Get dates that already have headlines
        cursor.execute("""
            SELECT DISTINCT date(timestamp) as dt 
            FROM news_sentiment 
            WHERE headline IS NOT NULL AND headline != ''
        """)
        existing = set(row[0] for row in cursor.fetchall())

        # Build full date range (1 year back)
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=365)

        missing = []
        current = start_date
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            # Skip weekends
            if current.weekday() < 5 and date_str not in existing:
                missing.append(current)
            current += timedelta(days=1)

        return missing

    def run_backfill(self):
        """Main backfill loop."""
        missing_dates = self.get_missing_dates()
        logger.info(f"Found {len(missing_dates)} missing weekdays to backfill via RSS.")

        if not missing_dates:
            logger.info("No missing dates found. Database is complete.")
            return

        total_inserted = 0
        for idx, date_obj in enumerate(missing_dates):
            day_headlines = []

            # Query Google News RSS with each financial query
            for query in self.GOOGLE_NEWS_QUERIES:
                results = self._fetch_google_news_rss(query, date_obj)
                day_headlines.extend(results)
                time.sleep(random.uniform(1.0, 2.0))

            # Deduplicate by md5_hash before inserting
            seen_hashes = set()
            unique_headlines = []
            for h in day_headlines:
                if h["md5_hash"] not in seen_hashes:
                    seen_hashes.add(h["md5_hash"])
                    unique_headlines.append(h)

            inserted = self._save_to_db(unique_headlines)
            total_inserted += inserted

            logger.info(
                f"[{idx+1}/{len(missing_dates)}] {date_obj.date()} | "
                f"Fetched {len(day_headlines)} -> {len(unique_headlines)} unique -> {inserted} inserted"
            )

            # Rate limiting between days
            time.sleep(random.uniform(2.0, 4.0))

        logger.info(f"Backfill complete. Total new headlines inserted: {total_inserted}")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    filler = RSSBackfiller()
    try:
        filler.run_backfill()
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user.")
    finally:
        filler.close()
