import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

class LocalDBManager:
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
        logging.info(f"Connected to local DB: {self.db_path}")

    def _init_db(self):
        cursor = self.conn.cursor()
        
        # 1. News Sentiment Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                source TEXT,
                headline TEXT NOT NULL,
                sentiment_score REAL,
                link TEXT,
                md5_hash TEXT UNIQUE,
                sector TEXT,
                is_pre_market INTEGER DEFAULT 0
            )
        """)
        
        # 2. 5-Minute Candles Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles_5min (
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                iv REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                PRIMARY KEY (timestamp, symbol)
            )
        """)

        # 3. Macro Data Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                date DATETIME PRIMARY KEY,
                shanghaicomp REAL,
                gold REAL,
                silver REAL,
                dowjones REAL,
                cac40 REAL,
                ftse100 REAL,
                dax REAL,
                sp500 REAL,
                hangseng REAL,
                nikkei225 REAL,
                nasdaq100 REAL,
                nifty50_daily REAL
            )
        """)

        # 4. Economic Events Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                event_name TEXT,
                impact_level INTEGER,
                actual TEXT,
                forecast TEXT,
                previous TEXT
            )
        """)
        
        # 5. Create Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_time ON news_sentiment(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_hash ON news_sentiment(md5_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_candle_time ON candles_5min(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_macro_time ON macro_data(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_time ON economic_events(timestamp)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_event_unique ON economic_events(timestamp, event_name)")
        
        self.conn.commit()

    def save_news(self, news_df: pd.DataFrame):
        """
        Expects a DataFrame with columns: 
        ['timestamp', 'source', 'headline', 'sentiment_score', 'link', 'md5_hash', 'sector']
        """
        # Ensure correct data types for SQLite
        news_df['sentiment_score'] = news_df['sentiment_score'].astype(object).where(news_df['sentiment_score'].notnull(), None)
        
        try:
            news_df.to_sql('news_sentiment', self.conn, if_exists='append', index=False, method='multi')
            logging.info(f"Saved {len(news_df)} news items.")
        except Exception as e:
            # Handle duplicates one by one if bulk fails
            logging.warning(f"Bulk insert failed ({e}). Inserting items individually...")
            for _, row in news_df.iterrows():
                try:
                    # Convert Series to dict for individual insertion to avoid further pandas to_sql issues
                    data = row.to_dict()
                    columns = ', '.join(data.keys())
                    placeholders = ', '.join(['?'] * len(data))
                    sql = f"INSERT INTO news_sentiment ({columns}) VALUES ({placeholders})"
                    self.conn.execute(sql, list(data.values()))
                except sqlite3.IntegrityError:
                    continue # Skip duplicates
                except Exception as inner_e:
                    logging.error(f"Error inserting individual row: {inner_e}")
            self.conn.commit()

    def get_pre_market_sentiment(self, date_str: str) -> float:
        """
        Calculates average sentiment for headlines on a given date.
        Used for quick lookups of a single day's sentiment.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT AVG(sentiment_score) FROM news_sentiment WHERE date(timestamp) = ? AND sentiment_score IS NOT NULL",
            (date_str,)
        )
        result = cursor.fetchone()[0]
        return result if result is not None else 0.0

    def get_unscored_headlines(self, batch_size: int = 0) -> pd.DataFrame:
        """Returns headlines where sentiment_score IS NULL."""
        query = "SELECT id, headline FROM news_sentiment WHERE sentiment_score IS NULL"
        if batch_size > 0:
            query += f" LIMIT {batch_size}"
        return pd.read_sql(query, self.conn)

    def update_sentiment_scores(self, scores: list):
        """
        Batch update sentiment scores.
        scores: list of (sentiment_score, id) tuples
        """
        cursor = self.conn.cursor()
        cursor.executemany(
            "UPDATE news_sentiment SET sentiment_score = ? WHERE id = ?",
            scores
        )
        self.conn.commit()
        logging.info(f"Updated sentiment scores for {len(scores)} headlines.")

    def get_daily_aggregated_sentiment(self, date_str: str) -> dict:
        """
        Returns aggregated sentiment for a single date.
        Uses weighted average: first 10 headlines per source get 1.5x weight.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT sentiment_score 
            FROM news_sentiment 
            WHERE date(timestamp) = ? AND sentiment_score IS NOT NULL
            ORDER BY id ASC
        """, (date_str,))
        scores = [row[0] for row in cursor.fetchall()]

        if not scores:
            return {
                "Sentiment_Score": 0.0,
                "Pos_Score": 0.0,
                "Neg_Score": 0.0,
                "Headline_Count": 0.0
            }

        # Weighted: first 10 get 1.5x, rest get 1.0x
        weights = [1.5 if i < 10 else 1.0 for i in range(len(scores))]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return {
            "Sentiment_Score": weighted_sum / total_weight,
            "Pos_Score": sum(1 for s in scores if s > 0.05) / len(scores),
            "Neg_Score": sum(1 for s in scores if s < -0.05) / len(scores),
            "Headline_Count": float(len(scores))
        }

    def get_sentiment_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Returns daily aggregated sentiment for a date range.
        Output columns: Date, Sentiment_Score, Pos_Score, Neg_Score, Headline_Count
        Used by trading_env.py to merge sentiment into the observation space.
        """
        query = """
            SELECT 
                date(timestamp) as Date,
                AVG(sentiment_score) as Sentiment_Score,
                CAST(SUM(CASE WHEN sentiment_score > 0.05 THEN 1 ELSE 0 END) AS REAL) / 
                    MAX(COUNT(*), 1) as Pos_Score,
                CAST(SUM(CASE WHEN sentiment_score < -0.05 THEN 1 ELSE 0 END) AS REAL) / 
                    MAX(COUNT(*), 1) as Neg_Score,
                COUNT(*) as Headline_Count
            FROM news_sentiment
            WHERE date(timestamp) BETWEEN ? AND ?
                AND sentiment_score IS NOT NULL
            GROUP BY date(timestamp)
            ORDER BY date(timestamp)
        """
        df = pd.read_sql(query, self.conn, params=(start_date, end_date))
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    def get_headline_stats(self) -> dict:
        """Returns summary stats about the main tables."""
        cursor = self.conn.cursor()
        
        # News Stats
        cursor.execute("SELECT COUNT(*) FROM news_sentiment")
        total_news = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM news_sentiment WHERE sentiment_score IS NOT NULL")
        scored_news = cursor.fetchone()[0]
        
        # Macro Stats
        cursor.execute("SELECT COUNT(*) FROM macro_data")
        total_macro = cursor.fetchone()[0]
        
        # Event Stats
        cursor.execute("SELECT COUNT(*) FROM economic_events")
        total_events = cursor.fetchone()[0]

        return {
            "total_headlines": total_news,
            "scored_headlines": scored_news,
            "macro_rows": total_macro,
            "economic_events": total_events
        }

    def save_macro_data(self, df: pd.DataFrame):
        """Saves macro data to the macro_data table, replacing existing rows on primary key conflict."""
        try:
            # Ensure column names are lowercase to match SQL schema
            df.columns = [c.lower() for c in df.columns]
            df.to_sql('macro_data', self.conn, if_exists='append', index=False, method='multi')
            logging.info(f"Saved {len(df)} macro data rows.")
        except sqlite3.IntegrityError:
            # Handle conflicts by updating one by one if necessary
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                columns = ', '.join(row_dict.keys())
                placeholders = ', '.join(['?'] * len(row_dict))
                sql = f"INSERT OR REPLACE INTO macro_data ({columns}) VALUES ({placeholders})"
                self.conn.execute(sql, list(row_dict.values()))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Error saving macro data: {e}")

    def save_economic_events(self, df: pd.DataFrame):
        """Saves economic events to the economic_events table, avoiding duplicates."""
        try:
            # Match schema: timestamp, event_name, impact_level, actual, forecast, previous
            cols_to_map = {
                'Date': 'timestamp',
                'Event_Name': 'event_name',
                'Impact_Level': 'impact_level',
                'Actual': 'actual',
                'Forecast': 'forecast',
                'Previous': 'previous'
            }
            subset = df[[c for c in cols_to_map.keys() if c in df.columns]].rename(columns=cols_to_map)
            
            # Use individual inserts with OR IGNORE to respect the UNIQUE index on (timestamp, event_name)
            for _, row in subset.iterrows():
                row_dict = row.to_dict()
                columns = ', '.join(row_dict.keys())
                placeholders = ', '.join(['?'] * len(row_dict))
                sql = f"INSERT OR IGNORE INTO economic_events ({columns}) VALUES ({placeholders})"
                self.conn.execute(sql, list(row_dict.values()))
            
            self.conn.commit()
            logging.info(f"Processed {len(subset)} economic events (duplicates ignored).")
        except Exception as e:
            logging.error(f"Error saving economic events: {e}")

    def get_macro_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieves macro data for a date range."""
        query = "SELECT * FROM macro_data WHERE date BETWEEN ? AND ? ORDER BY date ASC"
        df = pd.read_sql(query, self.conn, params=(start_date, end_date))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def get_events_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieves economic events for a date range."""
        query = "SELECT * FROM economic_events WHERE date(timestamp) BETWEEN ? AND ? ORDER BY timestamp ASC"
        df = pd.read_sql(query, self.conn, params=(start_date, end_date))
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    db = LocalDBManager()
    stats = db.get_headline_stats()
    print("Database stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    db.close()

