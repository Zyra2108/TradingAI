
"""Central configuration for the ECL AI trading project."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

@dataclass(frozen=True)
class AlpacaSettings:
    """Alpaca settings; credential fields are excluded from repr/log output."""

    api_key_id: str | None = field(repr=False)
    api_secret_key: str | None = field(repr=False)
    display_authorized: bool = False
    model_path: Path | None = None
    model_spec_path: Path | None = None
    database_path: Path | None = None

    @property
    def credentials_configured(self) -> bool:
        return bool(self.api_key_id and self.api_secret_key)


def get_alpaca_settings(environ=None) -> AlpacaSettings:
    """Resolve Alpaca configuration; accepts a mapping for secret-safe tests."""
    source = os.environ if environ is None else environ

    def configured_path(variable: str, default: Path) -> Path:
        raw = source.get(variable)
        if not raw:
            return default
        path = Path(raw)
        return path if path.is_absolute() else BASE_DIR / path

    return AlpacaSettings(
        api_key_id=source.get("APCA_API_KEY_ID") or None,
        api_secret_key=source.get("APCA_API_SECRET_KEY") or None,
        display_authorized=str(source.get("ALPACA_IEX_DISPLAY_AUTHORIZED", "false")).strip().lower() == "true",
        model_path=configured_path("ALPACA_MODEL_PATH", BASE_DIR / "models" / "alpaca_final_freeze_v1" / "candidate_pipeline.joblib"),
        model_spec_path=configured_path("ALPACA_MODEL_SPEC_PATH", BASE_DIR / "models" / "alpaca_final_freeze_v1" / "frozen_candidate_spec.json"),
        database_path=configured_path("ALPACA_DB_PATH", BASE_DIR / "data" / "live" / "ecl_alpaca_iex.sqlite3"),
    )


ALPACA_SETTINGS = get_alpaca_settings()
APCA_API_KEY_ID = ALPACA_SETTINGS.api_key_id
APCA_API_SECRET_KEY = ALPACA_SETTINGS.api_secret_key
ALPACA_IEX_DISPLAY_AUTHORIZED = ALPACA_SETTINGS.display_authorized
ALPACA_CREDENTIALS_CONFIGURED = ALPACA_SETTINGS.credentials_configured
ALPACA_MODEL_PATH = ALPACA_SETTINGS.model_path
ALPACA_MODEL_SPEC_PATH = ALPACA_SETTINGS.model_spec_path
ALPACA_DB_PATH = ALPACA_SETTINGS.database_path

# Stock configuration
SYMBOL = "ECL"
EXCHANGE = "NYSE"

# Market data configuration
INTERVAL = "5min"
PREDICTION_HORIZON_MINUTES = 15

# Project directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
