import os
from typing import Any

from dotenv import load_dotenv

from utils.logger import get_logger

logger = get_logger("secrets_manager")

# Load environment variables
load_dotenv()


class SecretsManager:
    def __init__(self):
        self.vault_url = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.use_vault = os.getenv("USE_VAULT", "false").lower() == "true"
        self.client = None

        if self.use_vault:
            if not self.vault_token:
                logger.warning("Vault is enabled but VAULT_TOKEN is missing. Trying local file.")
                token_path = os.getenv("VAULT_TOKEN_PATH", "/etc/vault/token")
                try:
                    with open(token_path) as f:
                        self.vault_token = f.read().strip()
                except Exception as e:
                    logger.error(f"Failed to read vault token from {token_path}: {e}")

            try:
                import hvac

                self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
                if not self.client.is_authenticated():
                    logger.error("Vault authentication failed.")
                    self.client = None
            except Exception as e:
                logger.error(f"Could not connect to Vault at {self.vault_url}: {e}")
                self.client = None

    def get_secret(self, key: str, default: str | None = None, cast_type: type = str) -> Any:
        """
        Fetch a secret, checking Hashicorp Vault first, then falling back to .env / os.getenv.
        Validates the type (e.g. cast_type=int ensures the value is an integer).
        """
        value = None

        # 1. Try Vault
        if self.client:
            try:
                # Assuming secrets are stored in a standard KV store
                # e.g., secret/data/quant/settings
                response = self.client.secrets.kv.v2.read_secret_version(
                    mount_point="secret", path="quant/settings"
                )
                value = response["data"]["data"].get(key)
            except Exception as e:
                logger.debug(f"Key {key} not found in Vault: {e}")

        # 2. Try OS Env (.env)
        if value is None:
            value = os.getenv(key, default)

        # 3. Apply validation
        if value is not None:
            try:
                if cast_type is bool:
                    return str(value).lower() in ("true", "1", "yes")
                return cast_type(value)
            except ValueError:
                logger.error(
                    f"Config validation error: {key} must be of type {cast_type.__name__}, got '{value}'"
                )
                raise TypeError(
                    f"Invalid type for configuration {key}: expected {cast_type.__name__}"
                )

        return default


_secrets_manager = SecretsManager()


def get_secret(key: str, default: str | None = None, cast_type: type = str) -> Any:
    return _secrets_manager.get_secret(key, default, cast_type)
