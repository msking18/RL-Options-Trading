import os
import requests
from urllib.parse import urlencode
from dotenv import load_dotenv, set_key

class UpstoxAuth:
    """
    Handles OAuth2 authentication for Upstox API v2.
    """
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        self.env_path = env_path
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")

    def get_auth_url(self) -> str:
        """Generate the URL the user needs to visit to authorize the app."""
        base_url = "https://api.upstox.com/v2/login/authorization/dialog"
        params = {
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri,
            "response_type": "code"
        }
        return f"{base_url}?{urlencode(params)}"

    def exchange_code_for_token(self, auth_code: str) -> str:
        """Exchange the authorization code for an access token."""
        url = "https://api.upstox.com/v2/login/authorization/token"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {
            'code': auth_code,
            'client_id': self.api_key,
            'client_secret': self.api_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.save_token(self.access_token)
            return self.access_token
        else:
            raise Exception(f"Failed to exchange code for token: {response.text}")

    def save_token(self, token: str):
        """Save the access token to the .env file for future use."""
        set_key(self.env_path, "UPSTOX_ACCESS_TOKEN", token)
        print(f"Access token saved to {self.env_path}")

if __name__ == "__main__":
    # Interactive login flow
    auth = UpstoxAuth()
    
    if not auth.api_key or not auth.api_secret:
        print("Error: Missing UPSTOX_API_KEY or UPSTOX_API_SECRET in .env file.")
    else:
        print("Visit this URL to authorize the app:")
        print(auth.get_auth_url())
        
        code = input("\nAfter logging in, copy the 'code' from the URL and paste it here: ").strip()
        if code:
            try:
                token = auth.exchange_code_for_token(code)
                print(f"Success! Access Token generated.")
            except Exception as e:
                print(f"Error: {e}")
