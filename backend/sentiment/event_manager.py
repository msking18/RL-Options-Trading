import requests
import logging
from bs4 import BeautifulSoup
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from backend.db.local_db_manager import LocalDBManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EconomicEventManager:
    """
    Fetches and stores Indian economic calendar events from Investing.com.
    Uses the internal API endpoint for structured data retrieval.
    """
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db = LocalDBManager(db_path)
        self.data_dir = os.path.dirname(db_path)
        self.session = requests.Session()
        self.headers_list = [
            {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36', 'X-Requested-With': 'XMLHttpRequest', 'Accept': 'text/html, */*; q=0.01', 'Referer': 'https://www.investing.com/economic-calendar/'},
            {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36', 'X-Requested-With': 'XMLHttpRequest', 'Accept': 'text/html, */*; q=0.01', 'Referer': 'https://www.investing.com/economic-calendar/'},
            {'User-Agent': 'Mozilla/5.0 (X11; Linux x87_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36', 'X-Requested-With': 'XMLHttpRequest', 'Accept': 'text/html, */*; q=0.01', 'Referer': 'https://www.investing.com/economic-calendar/'}
        ]
        self.headers = self.headers_list[0]
        
        # India's country ID on Investing.com
        self.india_country_id = 14
        
        # Key high/medium impact Indian economic events to track
        self.key_events = [
            'rbi interest rate decision',
            'rbi monetary policy',
            'cpi', 'consumer price index',
            'gdp', 'gross domestic product',
            'wpi', 'wholesale price index',
            'manufacturing pmi', 'services pmi', 'composite pmi',
            'trade balance', 'export', 'import',
            'industrial production', 'iip',
            'bank loan growth',
            'foreign exchange reserves',
            'budget', 'fiscal deficit',
            'unemployment',
        ]
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_events(self, from_date: datetime, to_date: datetime):
        """
        Fetch economic events from Investing.com for India within a date range.
        Uses the internal API endpoint.
        """
        url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
        
        all_events = []
        
        # Process in monthly chunks to avoid overwhelming the API
        current_start = from_date
        while current_start < to_date:
            current_end = min(current_start + timedelta(days=30), to_date)
            
            payload = {
                'country[]': self.india_country_id,
                'dateFrom': current_start.strftime('%Y-%m-%d'),
                'dateTo': current_end.strftime('%Y-%m-%d'),
                'timeZone': 55,  # IST
                'timeFilter': 'timeRemain',
                'currentTab': 'custom',
                'limit_from': 0,
            }
            
            max_retries = 5
            retry_delay = 5
            success = False
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching events from {current_start.date()} to {current_end.date()} (Attempt {attempt+1})...")
                    response = self.session.post(url, data=payload, headers=self.headers, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        html_content = data.get('data', '')
                        
                        if html_content:
                            events = self._parse_events_html(html_content)
                            all_events.extend(events)
                            logger.info(f"Parsed {len(events)} events for this month chunk.")
                        success = True
                        break
                    elif response.status_code == 429:
                        logger.warning(f"Investing.com returned 429 (Too Many Requests). Backing off for {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.warning(f"Investing.com returned status {response.status_code}. Skipping chunk.")
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching symbols on attempt {attempt+1}: {str(e)}")
                    time.sleep(1)
            
            if not success:
                logger.error(f"Failed to fetch data for range {current_start.date()} to {current_end.date()} after {max_retries} attempts.")
            
            current_start = current_end + timedelta(days=1)
            import random
            time.sleep(random.uniform(5, 10)) # Increased randomized delay
            self.headers = random.choice(self.headers_list) # Rotate UA
        
        return all_events
    
    def _parse_events_html(self, html: str):
        """Parse the HTML table returned by the Investing.com API."""
        events = []
        soup = BeautifulSoup(html, 'html.parser')
        
        rows = soup.find_all('tr', class_='js-event-item')
        for row in rows:
            try:
                # Extract date from data attribute
                event_date_str = row.get('data-event-datetime', '')
                
                # Event name
                event_td = row.find('td', class_='event')
                event_name = event_td.get_text(strip=True) if event_td else ''
                
                # Impact level (number of bull icons)
                impact_td = row.find('td', class_='sentiment')
                impact = 0
                if impact_td:
                    bulls = impact_td.find_all('i', class_='grayFullBullishIcon')
                    impact = len(bulls)
                
                # Actual, Forecast, Previous values
                actual_td = row.find('td', class_='act')
                forecast_td = row.find('td', class_='fore')
                previous_td = row.find('td', class_='prev')
                
                actual = actual_td.get_text(strip=True) if actual_td else ''
                forecast = forecast_td.get_text(strip=True) if forecast_td else ''
                previous = previous_td.get_text(strip=True) if previous_td else ''
                
                if event_name:
                    events.append({
                        'Date': event_date_str,
                        'Event_Name': event_name,
                        'Impact_Level': impact,
                        'Actual': actual,
                        'Forecast': forecast,
                        'Previous': previous
                    })
            except Exception as e:
                continue
        
        return events

    def sync_history(self, lookback_days=365):
        """Sync event history for the last N days + 30 days ahead."""
        end_date = datetime.now() + timedelta(days=30)  # Include upcoming events
        start_date = datetime.now() - timedelta(days=lookback_days)
        
        events = self.fetch_events(start_date, end_date)
        
        if not events:
            logger.warning("No events were fetched. The API may be blocking requests.")
            return
        
        df = pd.DataFrame(events)
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Filter for medium/high impact (2+ bulls) or key named events
        df_filtered = df[
            (df['Impact_Level'] >= 2) | 
            (df['Event_Name'].str.lower().apply(
                lambda x: any(kw in x for kw in self.key_events)
            ))
        ].copy()
        
        # Feature engineering: Create daily event flags
        df_filtered['Date_Only'] = df_filtered['Date'].dt.normalize()
        
        # Save to SQLite
        self.db.save_economic_events(df_filtered)
        logger.info(f"Saved {len(df_filtered)} events to SQLite.")
        
        return df_filtered

    def create_daily_features(self):
        """
        Generate daily feature vectors from event data in SQLite.
        Features: is_event_day, days_until_next_rbi, days_until_next_high_impact
        """
        # Fetch all events from DB
        try:
            events_df = self.db.get_events_range("2000-01-01", "2100-01-01")
            if events_df.empty:
                logger.warning("No event data found in DB.")
                return None
            
            # Match schema expected by joining logic
            events_df = events_df.rename(columns={'timestamp': 'Date'})
            events_df['Date_Only'] = pd.to_datetime(events_df['Date']).dt.normalize()
            
            # Create a complete date range
            date_range = pd.date_range(
                start=events_df['Date_Only'].min(),
                end=events_df['Date_Only'].max(),
                freq='D'
            )
            
            daily_df = pd.DataFrame({'Date': date_range})
            
            # is_event_day: 1 if any event scheduled
            event_dates = set(events_df['Date_Only'].unique())
            daily_df['Is_Event_Day'] = daily_df['Date'].apply(lambda d: 1 if d in event_dates else 0)
            
            # Count events per day
            event_counts = events_df.groupby('Date_Only').size().reset_index(name='Event_Count')
            daily_df = daily_df.merge(event_counts, left_on='Date', right_on='Date_Only', how='left')
            daily_df['Event_Count'] = daily_df['Event_Count'].fillna(0).astype(int)
            daily_df = daily_df.drop(columns=['Date_Only'], errors='ignore')
            
            # Max impact level for the day
            max_impact = events_df.groupby('Date_Only')['Impact_Level'].max().reset_index(name='Max_Impact')
            daily_df = daily_df.merge(max_impact, left_on='Date', right_on='Date_Only', how='left')
            daily_df['Max_Impact'] = daily_df['Max_Impact'].fillna(0).astype(int)
            daily_df = daily_df.drop(columns=['Date_Only'], errors='ignore')
            
            # Days until next RBI event
            rbi_dates = sorted(events_df[
                events_df['Event_Name'].str.lower().str.contains('rbi|interest rate', na=False)
            ]['Date_Only'].unique())
            
            if rbi_dates:
                daily_df['Days_To_RBI'] = daily_df['Date'].apply(
                    lambda d: min((rd - d).days for rd in rbi_dates if rd >= d) if any(rd >= d for rd in rbi_dates) else 999
                )
            else:
                daily_df['Days_To_RBI'] = 999
            
            return daily_df
        except Exception as e:
            logger.error(f"Error creating daily features: {e}")
            return None

if __name__ == "__main__":
    manager = EconomicEventManager()
    df = manager.sync_history(lookback_days=65)
    if df is not None:
        print(f"\nSample events:")
        print(df.head(10))
        
        features = manager.create_daily_features()
        if features is not None:
            print(f"\nDaily features sample:")
            print(features.tail(10))
