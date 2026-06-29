import json
import os
from typing import Any

import yaml


class StrategyConfig:
    """
    Configuration loader and validator for strategies in Research OS v2.
    Decouples strategy parameter definitions from python execution code.
    """

    def __init__(self, config_dict: dict[str, Any] | None = None):
        self.config = config_dict or {}

    @classmethod
    def from_json(cls, file_path: str) -> "StrategyConfig":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Strategy config file not found: {file_path}")
        with open(file_path) as f:
            return cls(json.load(f))

    @classmethod
    def from_yaml(cls, file_path: str) -> "StrategyConfig":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Strategy config file not found: {file_path}")
        with open(file_path) as f:
            return cls(yaml.safe_load(f))

    def get_strategy_name(self) -> str:
        return self.config.get("strategy_name", "unknown_strategy")

    def get_param(self, name: str, default: Any = None) -> Any:
        params = self.config.get("parameters", {})
        return params.get(name, default)

    def get_risk_param(self, name: str, default: Any = None) -> Any:
        risk = self.config.get("risk", {})
        return risk.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return self.config
