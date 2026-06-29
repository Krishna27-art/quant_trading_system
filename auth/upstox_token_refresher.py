import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger

logger = get_logger("upstox_auth")


def generate_auth_url(client_id: str, redirect_uri: str) -> str:
    """Generates the Upstox OAuth login URL."""
    return f"https://api.upstox.com/v2/oauth/authorize?client_id={client_id}&redirect_uri={redirect_uri}&response_type=code"


def exchange_code_for_token(
    client_id: str, client_secret: str, redirect_uri: str, code: str
) -> str:
    """Exchanges the temporary authorization code for a permanent daily access token."""
    url = "https://api.upstox.com/v2/oauth/token"
    headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    resp = requests.post(url, headers=headers, data=data, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    return result["access_token"]


def update_env_file(token: str) -> None:
    """Writes the access token to the local .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    lines = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()

    token_line = f"UPSTOX_ACCESS_TOKEN={token}\n"
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("UPSTOX_ACCESS_TOKEN="):
            lines[i] = token_line
            replaced = True
            break

    if not replaced:
        lines.append(token_line)

    with open(env_path, "w") as f:
        f.writelines(lines)

    logger.info("Successfully updated .env with new UPSTOX_ACCESS_TOKEN")


def main():
    load_dotenv()

    client_id = os.getenv("UPSTOX_API_KEY")
    client_secret = os.getenv("UPSTOX_API_SECRET")
    redirect_uri = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/callback")

    if not client_id or not client_secret:
        logger.critical("UPSTOX_API_KEY and UPSTOX_API_SECRET must be set in .env")
        sys.exit(1)

    if len(sys.argv) > 1:
        # Code passed directly
        code = sys.argv[1]
        try:
            token = exchange_code_for_token(client_id, client_secret, redirect_uri, code)
            update_env_file(token)
            print("Access token generated and saved successfully.")
        except Exception as e:
            logger.critical(f"Failed to exchange code: {e}")
            sys.exit(1)
    else:
        # Print auth URL
        auth_url = generate_auth_url(client_id, redirect_uri)
        print("Please visit the following URL to log in and authorize Upstox:")
        print(auth_url)
        print("\nAfter logging in, you will be redirected to a URL that looks like:")
        print(f"{redirect_uri}?code=XXXXXX")
        print(
            "\nCopy the 'code' parameter value and run this script again with the code as an argument:"
        )
        print("python auth/upstox_token_refresher.py YOUR_CODE_HERE")


if __name__ == "__main__":
    main()
