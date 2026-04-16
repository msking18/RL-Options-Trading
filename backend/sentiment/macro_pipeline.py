import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class MacroDataPipeline:
    """
    Pipeline to fetch global macro indicators (Indices, Gold, Silver) 
    using yfinance for historical training and correlation analysis.
    """
    def __init__(self, data_dir="data/historical"):
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, "macro_data.parquet")
        # 10 Major Indices + Gold/Silver
        self.symbols = {
            "^GSPC": "SP500",
            "^NDX": "Nasdaq100",
            "^DJI": "DowJones",
            "^FTSE": "FTSE100",
            "^GDAXI": "DAX",
            "^N225": "Nikkei225",
            "^HSI": "HangSeng",
            "000001.SS": "ShanghaiComp",
            "^FCHI": "CAC40",
            "^NSEI": "Nifty50_Daily",
            "GC=F": "Gold",
            "SI=F": "Silver"
        }

    def sync_data(self, period="1y"):
        """Fetch and store macro data."""
        print(f"--- Syncing Macro Data (Period: {period}) ---", flush=True)
        try:
            # Fetch data
            symbol_list = list(self.symbols.keys())
            data = yf.download(symbol_list, period=period, interval="1d")
            
            if data.empty:
                print("Error: No data returned from yfinance.", flush=True)
                return

            # We focus on Adjusted Close for macro trends to account for dividends/splits correctly
            # If Adj Close is missing (common in some indexes), fall back to Close
            if 'Adj Close' in data:
                macro_df = data['Adj Close']
            else:
                macro_df = data['Close']
            
            # Rename columns to friendly names
            macro_df = macro_df.rename(columns=self.symbols)
            
            # Add simple returns for internal environment use later
            # macro_returns = macro_df.pct_change()
            
            # Ensure the directory exists
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            
            # Reset index to make Date a column for Parquet compatibility
            macro_df = macro_df.reset_index()
            # Ensure Date is tz-naive for parquet
            macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.tz_localize(None)
            
            macro_df.to_parquet(self.file_path, index=False)
            print(f"Successfully saved macro data to {self.file_path}", flush=True)
            print(f"Columns: {list(macro_df.columns)}", flush=True)
            print(f"Latest Date: {macro_df['Date'].max()}", flush=True)
            
        except Exception as e:
            print(f"Exception during macro sync: {e}", flush=True)

    def load_data(self):
        """Helper to load the macro data."""
        if os.path.exists(self.file_path):
            return pd.read_parquet(self.file_path)
        return None

if __name__ == "__main__":
    pipeline = MacroDataPipeline()
    pipeline.sync_data(period="1y")
