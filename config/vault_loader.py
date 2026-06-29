"""
HashiCorp Vault Integration

Loads secrets and credentials securely from HashiCorp Vault.
Provides graceful degradation to local environment variables if Vault is unavailable.
Includes token refresh and caching logic.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

# Optional import for HashiCorp Vault client
try:
    import hvac

    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class VaultConfig:
    """Configuration for Vault connection."""

    vault_addr: str = os.getenv("VAULT_ADDR", "http://127.0.0.1:8200")
    token_path: str = os.getenv("VAULT_TOKEN_PATH", "/etc/vault/token")
    mount_point: str = "secret"
    enabled: bool = os.getenv("VAULT_ENABLED", "false").lower() == "true"


class VaultSecretLoader:
    """Loads and caches secrets from HashiCorp Vault."""

    def __init__(self, config: VaultConfig | None = None):
        self.config = config or VaultConfig()
        self.client: Any | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes

        if self.config.enabled and HVAC_AVAILABLE:
            self._initialize_client()
        elif self.config.enabled and not HVAC_AVAILABLE:
            logger.warning(
                "Vault is enabled but 'hvac' library is not installed. Falling back to env vars."
            )

    def _initialize_client(self):
        """Initialize the hvac client and authenticate."""
        token = self._read_token()
        if not token:
            token = os.getenv("VAULT_TOKEN")

        if token:
            try:
                self.client = hvac.Client(url=self.config.vault_addr, token=token)
                if self.client.is_authenticated():
                    logger.info("Successfully authenticated with HashiCorp Vault.")
                else:
                    logger.error("Vault authentication failed. Token may be invalid.")
                    self.client = None
            except Exception as e:
                logger.error(f"Failed to connect to Vault at {self.config.vault_addr}: {e}")
                self.client = None
        else:
            logger.warning("No Vault token found. Operating without Vault integration.")

    def _read_token(self) -> str | None:
        """Read the Vault token from the filesystem."""
        try:
            if os.path.exists(self.config.token_path):
                with open(self.config.token_path) as f:
                    return f.read().strip()
        except Exception as e:
            logger.debug(f"Could not read Vault token from {self.config.token_path}: {e}")
        return None

    def load_secret(self, path: str) -> dict[str, Any]:
        """
        Load a secret from the given path in Vault.
        Falls back to environment variables if Vault is unavailable or fetch fails.
        """
        # Check cache first
        now = time.time()
        if path in self._cache:
            cache_entry = self._cache[path]
            if now - cache_entry["timestamp"] < self._cache_ttl:
                return cache_entry["data"]

        # Attempt to fetch from Vault
        secret_data = {}
        if self.client and self.client.is_authenticated():
            try:
                # Assuming KV v2 engine
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path, mount_point=self.config.mount_point
                )
                secret_data = response.get("data", {}).get("data", {})

                # Update cache
                self._cache[path] = {"data": secret_data, "timestamp": now}
                return secret_data
            except Exception as e:
                logger.critical(
                    f"Error reading secret from Vault ({path}): {e}. Strict mode enabled."
                )
                raise RuntimeError(f"Vault error: {e}") from e
        else:
            logger.critical(f"Vault client not authenticated. Cannot fetch secret {path}.")
            raise RuntimeError("Vault is required but not authenticated.")

    def load_broker_credentials(self, broker_name: str) -> dict[str, str]:
        """Load credentials for a specific broker."""
        path = f"trading/brokers/{broker_name.lower()}"
        return self.load_secret(path)

    def load_db_credentials(self) -> dict[str, str]:
        """Load database connection credentials."""
        return self.load_secret("trading/database")

    def refresh_token(self):
        """Attempt to renew the current Vault token."""
        if self.client and self.client.is_authenticated():
            try:
                self.client.auth.token.renew_self()
                logger.info("Successfully renewed Vault token.")
            except Exception as e:
                logger.error(f"Failed to renew Vault token: {e}")
                # Try re-initializing completely
                self._initialize_client()
