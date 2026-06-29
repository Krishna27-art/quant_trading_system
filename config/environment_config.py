"""
Environment Configuration

Institutional-grade environment separation.
Separate Research, Paper Trading, and Production environments with independent databases.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Environment(str, Enum):
    """Environment enumeration."""

    RESEARCH = "research"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"


class EnvironmentConfig(BaseModel):
    """
    Environment configuration.

    Each environment has independent databases and configurations.
    """

    environment: Environment = Field(..., description="Environment type")
    database_path: str = Field(..., description="Database path")
    data_path: str = Field(..., description="Data path")
    feature_path: str = Field(..., description="Feature path")
    label_path: str = Field(..., description="Label path")
    experiment_path: str = Field(..., description="Experiment path")
    model_path: str = Field(..., description="Model path")
    log_path: str = Field(..., description="Log path")

    # Risk limits
    max_position_size: float | None = Field(None, description="Maximum position size")
    max_daily_loss: float | None = Field(None, description="Maximum daily loss")
    max_drawdown: float | None = Field(None, description="Maximum drawdown")

    # Trading limits
    max_trades_per_day: int | None = Field(None, description="Maximum trades per day")
    max_order_size: float | None = Field(None, description="Maximum order size")

    # Data access
    real_time_data: bool = Field(default=False, description="Access to real-time data")
    delayed_data_minutes: int | None = Field(None, description="Data delay in minutes")

    # Execution
    execution_mode: str = Field(default="simulation", description="Execution mode")
    slippage_model: str = Field(default="fixed", description="Slippage model")
    commission_model: str = Field(default="fixed", description="Commission model")

    # Monitoring
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    alert_on_errors: bool = Field(default=True, description="Alert on errors")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "environment": Environment.RESEARCH,
                "database_path": "data/research/db",
                "data_path": "data/research/data",
                "feature_path": "data/research/features",
                "label_path": "data/research/labels",
                "experiment_path": "data/research/experiments",
                "model_path": "data/research/models",
                "log_path": "logs/research",
                "max_position_size": None,
                "max_daily_loss": None,
                "max_drawdown": None,
                "max_trades_per_day": None,
                "max_order_size": None,
                "real_time_data": False,
                "delayed_data_minutes": 15,
                "execution_mode": "simulation",
                "slippage_model": "fixed",
                "commission_model": "fixed",
                "enable_monitoring": False,
                "alert_on_errors": False,
            }
        }


class EnvironmentManager:
    """
    Environment manager.

    Manages separate Research, Paper Trading, and Production environments.
    """

    def __init__(self, base_path: str = "."):
        """
        Initialize the environment manager.

        Args:
            base_path: Base path for all environments
        """
        self.base_path = Path(base_path)
        self.configs: dict[Environment, EnvironmentConfig] = {}

        # Initialize environments
        self._initialize_environments()

    def _initialize_environments(self):
        """Initialize all environment configurations."""
        # Research environment
        self.configs[Environment.RESEARCH] = EnvironmentConfig(
            environment=Environment.RESEARCH,
            database_path=str(self.base_path / "data" / "research" / "db"),
            data_path=str(self.base_path / "data" / "research" / "data"),
            feature_path=str(self.base_path / "data" / "research" / "features"),
            label_path=str(self.base_path / "data" / "research" / "labels"),
            experiment_path=str(self.base_path / "data" / "research" / "experiments"),
            model_path=str(self.base_path / "data" / "research" / "models"),
            log_path=str(self.base_path / "logs" / "research"),
            max_position_size=None,
            max_daily_loss=None,
            max_drawdown=None,
            max_trades_per_day=None,
            max_order_size=None,
            real_time_data=False,
            delayed_data_minutes=15,
            execution_mode="simulation",
            slippage_model="fixed",
            commission_model="fixed",
            enable_monitoring=False,
            alert_on_errors=False,
        )

        # Paper Trading environment
        self.configs[Environment.PAPER_TRADING] = EnvironmentConfig(
            environment=Environment.PAPER_TRADING,
            database_path=str(self.base_path / "data" / "paper_trading" / "db"),
            data_path=str(self.base_path / "data" / "paper_trading" / "data"),
            feature_path=str(self.base_path / "data" / "paper_trading" / "features"),
            label_path=str(self.base_path / "data" / "paper_trading" / "labels"),
            experiment_path=str(self.base_path / "data" / "paper_trading" / "experiments"),
            model_path=str(self.base_path / "data" / "paper_trading" / "models"),
            log_path=str(self.base_path / "logs" / "paper_trading"),
            max_position_size=1000000,
            max_daily_loss=0.05,
            max_drawdown=0.15,
            max_trades_per_day=100,
            max_order_size=100000,
            real_time_data=True,
            delayed_data_minutes=0,
            execution_mode="simulation",
            slippage_model="realistic",
            commission_model="realistic",
            enable_monitoring=True,
            alert_on_errors=True,
        )

        # Production environment
        self.configs[Environment.PRODUCTION] = EnvironmentConfig(
            environment=Environment.PRODUCTION,
            database_path=str(self.base_path / "data" / "production" / "db"),
            data_path=str(self.base_path / "data" / "production" / "data"),
            feature_path=str(self.base_path / "data" / "production" / "features"),
            label_path=str(self.base_path / "data" / "production" / "labels"),
            experiment_path=str(self.base_path / "data" / "production" / "experiments"),
            model_path=str(self.base_path / "data" / "production" / "models"),
            log_path=str(self.base_path / "logs" / "production"),
            max_position_size=10000000,
            max_daily_loss=0.03,
            max_drawdown=0.10,
            max_trades_per_day=50,
            max_order_size=1000000,
            real_time_data=True,
            delayed_data_minutes=0,
            execution_mode="live",
            slippage_model="realistic",
            commission_model="realistic",
            enable_monitoring=True,
            alert_on_errors=True,
        )

        # Create directories
        for config in self.configs.values():
            self._create_environment_directories(config)

    def _create_environment_directories(self, config: EnvironmentConfig):
        """
        Create directories for an environment.

        Args:
            config: Environment configuration
        """
        paths = [
            config.database_path,
            config.data_path,
            config.feature_path,
            config.label_path,
            config.experiment_path,
            config.model_path,
            config.log_path,
        ]

        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def get_config(self, environment: Environment) -> EnvironmentConfig:
        """
        Get configuration for an environment.

        Args:
            environment: Environment type

        Returns:
            Environment configuration
        """
        return self.configs[environment]

    def set_current_environment(self, environment: Environment) -> None:
        """
        Set the current environment.

        Args:
            environment: Environment type
        """
        self.current_environment = environment
        config = self.configs[environment]

        # Update paths globally
        import os

        os.environ["ENVIRONMENT"] = environment.value
        os.environ["DATABASE_PATH"] = config.database_path
        os.environ["DATA_PATH"] = config.data_path
        os.environ["FEATURE_PATH"] = config.feature_path
        os.environ["LABEL_PATH"] = config.label_path
        os.environ["EXPERIMENT_PATH"] = config.experiment_path
        os.environ["MODEL_PATH"] = config.model_path
        os.environ["LOG_PATH"] = config.log_path

    def get_current_environment(self) -> Environment:
        """
        Get the current environment.

        Returns:
            Current environment
        """
        import os

        env_str = os.environ.get("ENVIRONMENT", Environment.RESEARCH.value)
        return Environment(env_str)

    def validate_environment_transition(self, from_env: Environment, to_env: Environment) -> bool:
        """
        Validate environment transition.

        Args:
            from_env: Source environment
            to_env: Target environment

        Returns:
            True if transition is valid, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            Environment.RESEARCH: [Environment.PAPER_TRADING, Environment.PRODUCTION],
            Environment.PAPER_TRADING: [Environment.PRODUCTION],
            Environment.PRODUCTION: [],  # No transitions from production
        }

        return to_env in valid_transitions.get(from_env, [])

    def promote_to_production(self, experiment_id: str) -> bool:
        """
        Promote an experiment from paper trading to production.

        Args:
            experiment_id: Experiment ID to promote

        Returns:
            True if promotion successful, False otherwise
        """
        # Validate transition
        if not self.validate_environment_transition(
            Environment.PAPER_TRADING, Environment.PRODUCTION
        ):
            raise ValueError("Invalid environment transition")

        # Copy experiment from paper trading to production
        paper_config = self.configs[Environment.PAPER_TRADING]
        prod_config = self.configs[Environment.PRODUCTION]

        # Copy experiment artifacts
        import shutil

        src_experiment_path = Path(paper_config.experiment_path) / experiment_id
        dst_experiment_path = Path(prod_config.experiment_path) / experiment_id

        if src_experiment_path.exists():
            shutil.copytree(src_experiment_path, dst_experiment_path)

        # Copy model artifacts
        src_model_path = Path(paper_config.model_path) / experiment_id
        dst_model_path = Path(prod_config.model_path) / experiment_id

        if src_model_path.exists():
            shutil.copytree(src_model_path, dst_model_path)

        return True


# Global environment manager instance
environment_manager = EnvironmentManager()


def get_current_config() -> EnvironmentConfig:
    """
    Get the current environment configuration.

    Returns:
        Current environment configuration
    """
    env = environment_manager.get_current_environment()
    return environment_manager.get_config(env)


def set_environment(environment: Environment) -> None:
    """
    Set the current environment.

    Args:
        environment: Environment type
    """
    environment_manager.set_current_environment(environment)
