import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Dict
import time

# Import the DB manager we implemented earlier
from data.pipelines.db_manager import HistoricalDBManager

class UpstoxDataPipeline:
    """
    Pipeline to fetch historical OHLCV data from Upstox API v2 and store it locally.
    """
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        self.token = os.getenv("UPSTOX_ACCESS_TOKEN")
        if not self.token:
            print("WARNING: UPSTOX_ACCESS_TOKEN not found in .env file!", flush=True)
            
        self.db_manager = HistoricalDBManager()
        self.base_url = "https://api.upstox.com/v3" # Base URL is now directly v3
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

    def fetch_historical_candles(self, instrument_key: str, interval_val: int, to_date: datetime, from_date: datetime) -> pd.DataFrame:
        """Fetch candles for a specific range from Upstox API V3."""
        # V3 format: /v3/historical-candle/{instrumentKey}/{time_unit}/{interval_value}/{to_date}/{from_date}
        url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/{interval_val}/{to_date.strftime('%Y-%m-%d')}/{from_date.strftime('%Y-%m-%d')}"
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json().get('data', {}).get('candles', [])
                if not data:
                    return pd.DataFrame()
                # Upstox returns: [timestamp, open, high, low, close, volume, open_interest]
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
                return df
            else:
                print(f"Error fetching data for {instrument_key}: {response.status_code} - {response.text}", flush=True)
                return pd.DataFrame()
        except Exception as e:
            print(f"Exception during fetch for {instrument_key}: {e}", flush=True)
            return pd.DataFrame()

    def sync_index(self, instrument_key: str, symbol_display: str, lookback_years: float = 1.0):
        """Sync history for a specific index using V3."""
        print(f"\n--- Syncing {symbol_display} ({instrument_key}) ---", flush=True)
        
        # Use yesterday as the end date for historical sync to avoid "Invalid date range" errors
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        start_date = end_date - timedelta(days=int(365 * lookback_years))
        
        # Initialize loop variables
        fetch_from = start_date
        current_to = end_date
        total_fetched = 0
        
        # Check current coverage
        first_recorded = None
        last_recorded = None
        file_path = self.db_manager.get_file_path(symbol_display)
        if os.path.exists(file_path):
            existing_df = pd.read_parquet(file_path)
            if not existing_df.empty:
                existing_df['time'] = pd.to_datetime(existing_df['time'])
                first_recorded = existing_df['time'].min()
                last_recorded = existing_df['time'].max()

        # If lookback was increased (start_date < first_recorded), we must fetch the history gap.
        # If we need new data (end_date > last_recorded), we must fetch the future gap.
        if last_recorded and first_recorded:
            if start_date >= first_recorded and end_date <= last_recorded:
                print(f"Data for {symbol_display} is already up to date ({first_recorded.date()} to {last_recorded.date()}).", flush=True)
                return
            
            if end_date > last_recorded:
                print(f"Resuming sync for future data from {last_recorded.strftime('%Y-%m-%d')}", flush=True)
                # fetch_from is already start_date, and current_to is already end_date. 
                # The while loop will go from now down to April 2024, and save_data handles duplicates.
            elif start_date < first_recorded:
                print(f"Backfilling historical data from {start_date.strftime('%Y-%m-%d')} to {first_recorded.strftime('%Y-%m-%d')}", flush=True)
        else:
            print(f"Starting full sync from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", flush=True)

        # Chunking in 10-day blocks for V3 stability
        chunk_size = timedelta(days=10)
        while current_to > fetch_from:
            current_chunk_from = max(fetch_from, current_to - chunk_size)
            print(f"[{symbol_display}] Fetching {current_chunk_from.strftime('%Y-%m-%d')} to {current_to.strftime('%Y-%m-%d')}...", flush=True)
            
            if current_chunk_from >= current_to:
                break
                
            df_chunk = self.fetch_historical_candles(instrument_key, 5, current_to, current_chunk_from)
            if not df_chunk.empty:
                # Save incrementally to show progress and avoid loss if interrupted
                self.db_manager.save_data(symbol_display, df_chunk, append=True)
                total_fetched += len(df_chunk)
                # If we get data, update current_to to the day before the earliest timestamp in the chunk to avoid same-day loops
                # V3 API only has day granularity, so we must move to the previous day.
                current_to = df_chunk['time'].min().replace(hour=0, minute=0, second=0) - timedelta(seconds=1)
            else:
                print(f"[{symbol_display}] No data in this window, skipping...", flush=True)
                # If no data in this chunk, move back regardless to avoid infinite loop
                current_to = current_chunk_from - timedelta(days=1)
            
            # Rate limiting - Upstox has a limit on requests per second
            time.sleep(1.0)
            
        print(f"Sync complete for {symbol_display}. Added {total_fetched} new candles.", flush=True)

    def resolve_instrument_keys(self) -> Dict[str, str]:
        """Resolves the best available instrument keys for all indices."""
        print("Resolving instrument keys from master list...", flush=True)
        url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
        try:
            response = requests.get(url)
            import gzip
            import io
            with gzip.open(io.BytesIO(response.content), 'rt') as f:
                master_df = pd.read_csv(f)
        except Exception as e:
            print(f"Failed to fetch instrument master: {e}")
            return {}

        # 1. Start with Spot fallback
        final_keys = {
            "Nifty 50": "NSE_INDEX|Nifty 50",
            "Nifty Bank": "NSE_INDEX|Nifty Bank",
            "Nifty Fin Service": "NSE_INDEX|Nifty Fin Service",
            "Nifty Midcap Select": "NSE_INDEX|NIFTY MID SELECT",
            "Nifty IT": "NSE_INDEX|Nifty IT",
            "SENSEX": "BSE_INDEX|SENSEX",
            "SENSEX50": "BSE_INDEX|SENSEX50"
        }

        # 2. Try to upgrade to Futures for eligible indices (ONLY for short lookbacks)
        # Futures contracts expire and don't have multi-year history under a single key.
        # For long backfills, we MUST use Spot indices.
        if hasattr(self, '_current_lookback') and self._current_lookback > 0.5:
            print("Long lookback detected: Using Spot Index keys for maximum historical coverage.", flush=True)
            return final_keys

        futures_search = {
            "Nifty 50": "NIFTY",
            "Nifty Bank": "BANKNIFTY",
            "Nifty Fin Service": "FINNIFTY",
            "Nifty Midcap Select": "MIDCPNIFTY"
        }

        futs_df = master_df[master_df['instrument_type'] == 'FUTIDX'].copy()
        
        # Exact matching for tradingsymbol prefix to avoid 'NIFTY' matching 'NIFTYNXT50'
        # We look for [SEARCH_TERM][YY][MMM][FUT]
        import re

        futs_df['expiry'] = pd.to_datetime(futs_df['expiry'])
        futs_df = futs_df.sort_values('expiry')

        for display_name, search_term in futures_search.items():
            # Regex to match SYMBOL + YY + (Optional MMM) + FUT
            # e.g. NIFTY24OCTFUT or NIFTY24OUTFUT
            pattern = rf"^{search_term}\d{{2}}[A-Z]{{3}}FUT$"
            match = futs_df[futs_df['tradingsymbol'].str.match(pattern)].head(1)
            
            if not match.empty:
                key = match.iloc[0]['instrument_key']
                symbol = match.iloc[0]['tradingsymbol']
                expiry = match.iloc[0]['expiry'].strftime('%Y-%m-%d')
                print(f"[{display_name}] Upgraded to Futures: {symbol} (Key: {key}, Expiry: {expiry})")
                final_keys[display_name] = key
            else:
                # Fallback to simple startswith if regex is too strict
                match_fallback = futs_df[futs_df['tradingsymbol'] == f"{search_term}FUT"].head(1)
                if not match_fallback.empty:
                    key = match_fallback.iloc[0]['instrument_key']
                    final_keys[display_name] = key
                else:
                    print(f"[{display_name}] No exact Futures found, using Spot Index.")

        return final_keys

    def sync_all_indices(self, lookback_years=1.5):
        """Sync the top indices as per user preference."""
        self._current_lookback = lookback_years
        indices = self.resolve_instrument_keys()
        if not indices:
            print("Error: Could not resolve keys. Aborting sync.")
            return

        for name, key in indices.items():
            try:
                # For historical sync, we usually want at least 1 year
                self.sync_index(key, name, lookback_years=lookback_years)
            except Exception as e:
                print(f"Failed to sync {name}: {e}")

if __name__ == "__main__":
    # Full historical sync for indices (2.2 years back to April 2024)
    pipeline = UpstoxDataPipeline()
    pipeline.sync_all_indices(lookback_years=2.2)
