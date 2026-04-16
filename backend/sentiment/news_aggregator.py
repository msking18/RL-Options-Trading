import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Categories relevant to options trading on Indian indices
RELEVANT_KEYWORDS = [
    # Economics
    'rbi', 'inflation', 'gdp', 'fiscal', 'monetary', 'interest rate', 'repo',
    'cpi', 'iip', 'wpi', 'trade deficit', 'current account', 'forex', 'rupee',
    'budget', 'gst', 'tax', 'subsidy', 'policy', 'economic',
    # Business / Markets
    'nifty', 'sensex', 'stock', 'market', 'share', 'ipo', 'equity', 'bond',
    'mutual fund', 'fii', 'dii', 'sebi', 'nse', 'bse', 'trading', 'rally',
    'crash', 'correction', 'bull', 'bear', 'volatility', 'option', 'futures',
    'earnings', 'profit', 'revenue', 'quarterly', 'annual', 'dividend',
    'merger', 'acquisition', 'banking', 'fintech', 'insurance', 'nbfc',
    'adani', 'reliance', 'tata', 'infosys', 'hdfc', 'icici', 'sbi',
    'oil', 'crude', 'gold', 'silver', 'commodity', 'metal',
    # Geopolitical
    'geopoliti', 'war', 'conflict', 'sanction', 'tariff', 'trade war',
    'china', 'us ', 'usa', 'fed ', 'federal reserve', 'ecb', 'boj',
    'opec', 'nato', 'election', 'government', 'parliament', 'modi',
    'trump', 'biden', 'xi jinping', 'putin', 'brexit', 'diplomacy',
]

class NewsAggregator:
    """
    Scrapes financial news headlines from multiple Indian sources.
    Filters for economics, business, and geopolitical categories.
    Supports: Moneycontrol, Economic Times.
    """
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        self.session = requests.Session()

    def _get_headers(self):
        """Rotate User-Agents for each request."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=10.0',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
        }

    def _fetch_with_retry(self, url, method='GET', payload=None, retries=3, backoff=2):
        """Helper to fetch URLs with exponential backoff and randomized jitter."""
        for i in range(retries):
            try:
                headers = self._get_headers()
                if method == 'GET':
                    response = self.session.get(url, headers=headers, timeout=15)
                else:
                    response = self.session.post(url, data=payload, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    return response
                
                if response.status_code == 429:
                    wait_time = backoff * (2 ** i) + random.uniform(1, 4)
                    logger.warning(f"Rate limited (429) for {url}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    logger.warning(f"Access Forbidden (403) for {url}. Rotating IP/UA and waiting...")
                    time.sleep(random.uniform(5, 10))
                else:
                    logger.warning(f"Request failed with status {response.status_code} for {url}")
                    break
            except Exception as e:
                logger.error(f"Request error for {url}: {str(e)}")
                time.sleep(random.uniform(1, 3))
        return None

    def _is_relevant(self, headline: str) -> bool:
        """Check if a headline is relevant to economics, business, or geopolitics."""
        text_lower = headline.lower()
        return any(kw in text_lower for kw in RELEVANT_KEYWORDS)

    def fetch_moneycontrol(self, date_obj: datetime):
        """
        Scrapes Moneycontrol news archive for a specific date.
        URL Format: https://www.moneycontrol.com/news/news-all/date/DD-MM-YYYY/
        """
        date_str = date_obj.strftime("%d-%m-%Y")
        url = f"https://www.moneycontrol.com/news/news-all/date/{date_str}/"
        
        headlines = []
        try:
            logger.info(f"Fetching Moneycontrol headlines for {date_str}...")
            response = self._fetch_with_retry(url)
            
            if not response:
                logger.warning(f"Failed to fetch Moneycontrol for {date_str}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Moneycontrol archive headlines are typically in <h2> tags inside #leftContent
            content_div = soup.find('div', id='leftContent')
            if not content_div:
                # Fallback for newer layouts if any
                content_div = soup.find('ul', id='newslist')
            
            if content_div:
                # Find all <h2> tags which contain the headlines
                h2_elements = content_div.find_all('h2')
                for idx, h2 in enumerate(h2_elements):
                    a_tag = h2.find('a')
                    if a_tag:
                        headline = a_tag.get_text(strip=True)
                        link = a_tag.get('href')
                        if headline and self._is_relevant(headline):
                            # Top 5 headlines get higher weight (featured)
                            weight = 1.5 if idx < 5 else 1.0
                            headlines.append({
                                'headline': headline,
                                'source': 'Moneycontrol',
                                'url': link,
                                'date': date_obj.strftime("%Y-%m-%d"),
                                'weight': weight
                            })
            
            logger.info(f"Extracted {len(headlines)} headlines from Moneycontrol.")
            return headlines

        except Exception as e:
            logger.error(f"Error fetching Moneycontrol for {date_str}: {str(e)}")
            return []

    def fetch_et(self, date_obj: datetime):
        """
        Scrapes Economic Times archive.
        The URL uses a 'starttime' parameter representing days since Jan 1, 1900.
        """
        # Jan 1, 1900 is 1. (Excel/Unix date logic varies, ET uses 1 for 1/1/1900)
        # 1-1-1900 is offset 0? No, usually starttime 36892 is some date in 2001.
        # Reference: April 9, 2026 is starttime 46121? Let's calibrate.
        # Based on successful check earlier: 03 Apr 2026 is starttime 46115.
        
        # Calculation:
        ref_date = datetime(2026, 4, 3)
        ref_start = 46115
        delta = (date_obj - ref_date).days
        starttime = ref_start + delta
        
        url = f"https://economictimes.indiatimes.com/archivelist/year-{date_obj.year},month-{date_obj.month},starttime-{starttime}.cms"
        
        headlines = []
        try:
            logger.info(f"Fetching Economic Times headlines for {date_obj.date()}...")
            response = self._fetch_with_retry(url)
            
            if not response:
                logger.warning(f"Failed to fetch Economic Times headlines for {date_obj.date()}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')
            # ET archive headlines are in <ul> with class 'content' -> <li> -> <a>
            # Or inside a table. Based on recent probe:
            content_ul = soup.find('ul', class_='content')
            if content_ul:
                links = content_ul.find_all('a')
                for idx, a in enumerate(links):
                    text = a.get_text(strip=True)
                    if text and self._is_relevant(text):
                        # First 10 ET headlines are typically featured
                        weight = 1.5 if idx < 10 else 1.0
                        headlines.append({
                            'headline': text,
                            'source': 'Economic Times',
                            'url': "https://economictimes.indiatimes.com" + a.get('href', ''),
                            'date': date_obj.strftime("%Y-%m-%d"),
                            'weight': weight
                        })
            
            # Fallback for table structure if present
            if not headlines:
                table = soup.find('table', class_='content')
                if table:
                    links = table.find_all('a')
                    for idx, a in enumerate(links):
                        text = a.get_text(strip=True)
                        if text and self._is_relevant(text):
                            weight = 1.5 if idx < 10 else 1.0
                            headlines.append({
                                'headline': text,
                                'source': 'Economic Times',
                                'url': "https://economictimes.indiatimes.com" + a.get('href', ''),
                                'date': date_obj.strftime("%Y-%m-%d"),
                                'weight': weight
                            })

            logger.info(f"Extracted {len(headlines)} headlines from Economic Times.")
            return headlines

        except Exception as e:
            logger.error(f"Error fetching Economic Times: {str(e)}")
            return []

    def fetch_all_sources(self, date_obj: datetime):
        """
        Aggregates news from all implemented sources.
        """
        all_headlines = []
        
        # Moneycontrol (primary)
        mc = self.fetch_moneycontrol(date_obj)
        all_headlines.extend(mc)
        
        # Economic Times (secondary)
        et = self.fetch_et(date_obj)
        all_headlines.extend(et)
        
        # Add TOI here as it is implemented
        
        # Add a random sleep to avoid bot detection
        time.sleep(random.uniform(1, 3))
        
        return all_headlines

if __name__ == "__main__":
    # Quick test
    aggregator = NewsAggregator()
    test_date = datetime(2026, 4, 8)
    results = aggregator.fetch_all_sources(test_date)
    for res in results[:5]:
        print(f"[{res['source']}] {res['headline']}")
