"""
News Sentiment Pipeline - Consolidated SQLite version.

This pipeline fetches headlines, scores them with VADER, and stores
individual headlines in the SQLite database. The trading environment
reads aggregated sentiment from SQLite via LocalDBManager.get_sentiment_range().

Production: runs on a 5-minute check cycle against live RSS feeds.
Historical: use rss_backfiller.py + batch_scorer.py for gap filling.
"""
import pandas as pd
import hashlib
import os
from datetime import datetime, timedelta
import time
import random
import logging
from .news_aggregator import NewsAggregator
from .sentiment_engine import SentimentEngine
from backend.db.local_db_manager import LocalDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NewsPipeline")


class NewsSentimentPipeline:
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db = LocalDBManager(db_path)
        self.aggregator = NewsAggregator()
        self.engine = SentimentEngine()

    def _get_md5(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def sync_history(self, lookback_days=365):
        """Sync sentiment history for all missing dates in the last N days."""
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        start_date = end_date - timedelta(days=lookback_days)

        # Get dates already in DB
        existing_dates = set()
        try:
            stats_df = pd.read_sql(
                "SELECT DISTINCT date(timestamp) as dt FROM news_sentiment WHERE headline IS NOT NULL",
                self.db.conn
            )
            existing_dates = set(stats_df['dt'].tolist())
        except Exception as e:
            logger.error(f"Error reading existing dates: {e}")

        # Build missing dates list (weekdays only)
        all_dates = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
        missing_dates = [
            d for d in all_dates
            if d.strftime('%Y-%m-%d') not in existing_dates and d.weekday() < 5
        ]

        if not missing_dates:
            logger.info("Sentiment data is already complete for the requested range.")
            return

        logger.info(f"Syncing {len(missing_dates)} missing dates from {min(missing_dates).date()} to {max(missing_dates).date()}...")

        for idx, current_date in enumerate(missing_dates):
            try:
                # 1. Fetch Headlines
                headlines = self.aggregator.fetch_all_sources(current_date)

                if headlines:
                    # 2. Score and save each headline individually
                    news_rows = []
                    for h in headlines:
                        headline_text = h if isinstance(h, str) else h.get('headline', str(h))
                        sentiment = self.engine.analyze_headline(headline_text)
                        news_rows.append({
                            'timestamp': current_date.strftime('%Y-%m-%d 10:00:00'),
                            'source': 'ET/MC Pipeline',
                            'headline': headline_text,
                            'sentiment_score': sentiment['compound'],
                            'link': h.get('link', '') if isinstance(h, dict) else '',
                            'md5_hash': self._get_md5(headline_text),
                            'sector': 'markets',
                            'is_pre_market': 1,
                        })

                    if news_rows:
                        news_df = pd.DataFrame(news_rows)
                        self.db.save_news(news_df)
                        avg_score = sum(r['sentiment_score'] for r in news_rows) / len(news_rows)
                        logger.info(
                            f"[{idx+1}/{len(missing_dates)}] {current_date.date()}: "
                            f"Saved {len(news_rows)} headlines. Avg sentiment: {avg_score:.4f}"
                        )
                else:
                    logger.info(f"[{idx+1}/{len(missing_dates)}] {current_date.date()}: No headlines found.")

            except Exception as e:
                logger.error(f"Unexpected error in pipeline loop for {current_date.date()}: {e}")

            # Rate limiting
            time.sleep(random.uniform(0.5, 1.5))

        logger.info("Sync completed successfully.")

    def run_live_check(self):
        """
        Production mode: fetch latest headlines and score them.
        Intended to be called on a 5-minute interval.
        """
        now = datetime.now()
        try:
            headlines = self.aggregator.fetch_all_sources(now)
            if headlines:
                news_rows = []
                for h in headlines:
                    headline_text = h if isinstance(h, str) else h.get('headline', str(h))
                    sentiment = self.engine.analyze_headline(headline_text)
                    news_rows.append({
                        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                        'source': 'Live Feed',
                        'headline': headline_text,
                        'sentiment_score': sentiment['compound'],
                        'link': h.get('link', '') if isinstance(h, dict) else '',
                        'md5_hash': self._get_md5(headline_text),
                        'sector': 'markets',
                        'is_pre_market': 0,
                    })

                if news_rows:
                    news_df = pd.DataFrame(news_rows)
                    self.db.save_news(news_df)
                    logger.info(f"Live check: saved {len(news_rows)} new headlines.")
        except Exception as e:
            logger.error(f"Live check error: {e}")

    def close(self):
        self.db.close()


if __name__ == "__main__":
    pipeline = NewsSentimentPipeline()
    # Initial sync for 1 year
    pipeline.sync_history(lookback_days=365)
    pipeline.close()
