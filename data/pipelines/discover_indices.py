import os
import requests
import pandas as pd
from dotenv import load_dotenv

def discover_indices():
    load_dotenv()
    url = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    print(f"Downloading instrument master file...")
    
    try:
        df = pd.read_csv(url, compression='gzip')
    except Exception as e:
        print(f"Failed to download master list: {e}")
        return

    # Filter for NSE Indices
    # Note: Exchange 'NSE_INDEX' is correct. Instrument type is often 'index' or 'INDEX'.
    indices_df = df[df['exchange'] == 'NSE_INDEX'].copy()
    
    # Target names to find and their common aliases
    target_map = {
        "Nifty 50": ["Nifty 50", "NIFTY_50"],
        "Nifty Bank": ["Nifty Bank", "BANKNIFTY"],
        "Nifty Fin Service": ["Nifty Fin Service", "FINNIFTY"],
        "Nifty Midcap Select": ["Nifty Midcap Select", "MIDCPNIFTY", "NIFTY MID SELECT"],
        "Nifty Next 50": ["Nifty Next 50", "NIFTYNXT50"],
        "Nifty 100": ["Nifty 100"],
        "Nifty 500": ["Nifty 500"],
        "Nifty IT": ["Nifty IT"],
        "Nifty Auto": ["Nifty Auto"],
        "Nifty Pharma": ["Nifty Pharma"]
    }
    
    print(f"Total Indices found in NSE_INDEX: {len(indices_df)}")
    
    found_keys = []
    for display_name, aliases in target_map.items():
        match = pd.DataFrame()
        
        # Try exact matches first
        for alias in aliases:
            for col in ['tradingsymbol', 'instrument_key', 'name']:
                if col in indices_df.columns:
                    # Exact match (ignoring case)
                    temp_match = indices_df[indices_df[col].str.lower() == alias.lower()]
                    if not temp_match.empty:
                        match = temp_match
                        break
            if not match.empty: break
            
        # If no exact match, try contains with word boundaries
        if match.empty:
            for alias in aliases:
                for col in ['tradingsymbol', 'name']:
                    if col in indices_df.columns:
                        # Regex for word boundary to avoid 50 matching 500
                        regex = rf"\b{alias}\b"
                        temp_match = indices_df[indices_df[col].str.contains(regex, case=False, na=False, regex=True)]
                        if not temp_match.empty:
                            match = temp_match
                            break
                if not match.empty: break
        
        if not match.empty:
            key = match.iloc[0]['instrument_key']
            symbol = match.iloc[0].get('tradingsymbol', 'N/A')
            print(f"- {display_name:20}: Key='{key:25}', Symbol='{symbol}'")
            found_keys.append({"name": display_name, "key": key, "symbol": symbol})
        else:
            print(f"- {display_name:20}: NOT FOUND (Aliases checked: {aliases})")
            
    return found_keys

if __name__ == "__main__":
    discover_indices()
