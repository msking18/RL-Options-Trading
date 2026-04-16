import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

def test_v3_connection():
    load_dotenv()
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    
    if not token:
        print("Error: UPSTOX_ACCESS_TOKEN not found in .env")
        return

    # NSE_INDEX|Nifty 50
    instrument_key = "NSE_INDEX|Nifty 50"
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # V3 URL format: /v3/historical-candle/{instrumentKey}/{time_unit}/{interval_value}/{to_date}/{from_date}
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/minutes/5/{to_date}/{from_date}"
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    print(f"Testing V3 API URL: {url}")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles']
            print(f"Success! Fetched {len(candles)} candles using V3 API.")
            if len(candles) > 0:
                print(f"Sample: {candles[0]}")
        else:
            print(f"Response format mismatch: {data}")
    else:
        print(f"V3 Connection failed: {response.text}")

if __name__ == "__main__":
    test_v3_connection()
