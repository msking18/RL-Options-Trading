import os
import requests
from dotenv import load_dotenv

def test_connection():
    load_dotenv()
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    
    if not token:
        print("Error: UPSTOX_ACCESS_TOKEN not found in .env")
        return

    url = "https://api.upstox.com/v2/user/profile"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}'
    }

    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        profile = response.json()['data']
        print(f"Successfully connected to Upstox!")
        print(f"User: {profile.get('user_name')} ({profile.get('user_id')})")
        print(f"Broker: {profile.get('broker')}")
    else:
        print(f"Connection failed: {response.text}")

if __name__ == "__main__":
    test_connection()
