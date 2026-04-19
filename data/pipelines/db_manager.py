import pandas as pd
import os
from datetime import datetime
from typing import Optional, List

class HistoricalDBManager:
    """
    Manages the storage and retrieval of 5-minute historical OHLCV data.
    Ensures data consistency and avoids duplicate fetches from the API.
    """
    def __init__(self, base_path: str = "data/historical"):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def get_file_path(self, symbol: str, date: Optional[datetime] = None) -> str:
        """Get path for a specific symbol's CSV file."""
        # We store one file per symbol for simplicity, or we could branch by month
        return os.path.join(self.base_path, f"{symbol.replace(':', '_')}.parquet")

    def save_data(self, symbol: str, df: pd.DataFrame, append: bool = True):
        """Save OHLCV data to Parquet (more efficient than CSV for time-series)."""
        file_path = self.get_file_path(symbol)
        
        if append and os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            # Combine and remove duplicates based on timestamp
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['time'])
            combined_df.sort_values('time', inplace=True)
            combined_df.to_parquet(file_path, index=False)
        else:
            df.to_parquet(file_path, index=False)

    def load_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load data from the local database."""
        file_path = self.get_file_path(symbol)
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            import logging
            logging.warning(f"File not found for {symbol}: {abs_path} (CWD: {os.getcwd()})")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        df['time'] = pd.to_datetime(df['time'])
        
        if start_date:
            df = df[df['time'] >= start_date]
        if end_date:
            df = df[df['time'] <= end_date]
            
        return df

    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """Check the latest data point we have to resume fetching."""
        file_path = self.get_file_path(symbol)
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_parquet(file_path)
        return pd.to_datetime(df['time']).max()
