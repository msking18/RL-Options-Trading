import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import random
import hashlib
import logging
import pandas as pd
from typing import List, Dict
from backend.db.local_db_manager import LocalDBManager
from backend.sentiment.news_aggregator import NewsAggregator, RELEVANT_KEYWORDS

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("archive_filler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArchiveFiller:
    def __init__(self):
        self.db = LocalDBManager()
        self.aggregator = NewsAggregator()
        # ET base dates: Dec 30, 1899 is 0 or 1?
        # Calibration: April 9, 2026 is starttime 46121
        self.ref_date = datetime(2026, 4, 9)
        self.ref_starttime = 46121

    def calculate_starttime(self, target_date: datetime) -> int:
        delta = (target_date - self.ref_date).days
        return self.ref_starttime + delta

    def get_md5(self, text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def scrape_et_archive_day(self, date_obj: datetime, sector: str = "economy") -> List[Dict]:
        starttime = self.calculate_starttime(date_obj)
        
        # Sector filtering keywords in the URL
        sector_keywords = {
            "economy": ["/news/economy/"],
            "markets": ["/markets/", "nifty", "sensex", "stocks"],
            "international": ["/news/international/", "global"]
        }
        
        # General Archive URL - Essential for reliable extraction
        url = f"https://economictimes.indiatimes.com/archivelist/year-{date_obj.year},month-{date_obj.month},starttime-{starttime}.cms"
        
        logger.info(f"Scraping ET Archive for {date_obj.date()} (starttime: {starttime}, URL keywords: {sector_keywords.get(sector)})")
        
        response = self.aggregator._fetch_with_retry(url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = []
        
        # The primary ET archive container
        content_ul = soup.find('ul', class_='content')
        if content_ul:
            links = content_ul.find_all('a')
            for a in links:
                text = a.get_text(strip=True)
                href = a.get('href', '')
                
                # Check if relevant AND matches the sector URL path
                is_sector_match = any(kw in href.lower() for kw in sector_keywords.get(sector, []))
                
                if text and is_sector_match and self.aggregator._is_relevant(text):
                    headlines.append({
                        'timestamp': date_obj.strftime("%Y-%m-%d 10:00:00"),
                        'source': 'Economic Times',
                        'headline': text,
                        'sentiment_score': None,
                        'link': "https://economictimes.indiatimes.com" + href if not href.startswith('http') else href,
                        'md5_hash': self.get_md5(text),
                        'sector': sector,
                        'is_pre_market': 1
                    })
        
        return headlines

    def run_backfill(self, days: int = 365, sectors: List[str] = ["economy", "markets"]):
        """Runs the loop to backfill historical news."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_date = end_date
        while current_date >= start_date:
            combined_news = []
            for sector in sectors:
                day_news = self.scrape_et_archive_day(current_date, sector)
                combined_news.extend(day_news)
                # Respectful delay between sectors
                time.sleep(random.uniform(1, 2))
            
            if combined_news:
                df = pd.DataFrame(combined_news)
                self.db.save_news(df)
            
            # Progress update
            logger.info(f"Completed {current_date.date()} | Extracted {len(combined_news)} items.")
            
            # Delay between days to avoid rate limiting
            current_date -= timedelta(days=1)
            time.sleep(random.uniform(2, 4))

    def close(self):
        self.db.close()

if __name__ == "__main__":
    filler = ArchiveFiller()
    try:
        # Start by filling the last 30 days as a test, then we can expand
        logger.info("Starting historical news backfill (Last 365 days)...")
        filler.run_backfill(days=365)
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user.")
    finally:
        filler.close()
