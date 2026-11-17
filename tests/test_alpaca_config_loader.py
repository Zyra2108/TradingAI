from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
import live_app.model_loader as model_loader

FEATURES = [
    "close", "return_1", "return_3", "return_6", "return_12",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_20", "ema_ratio_5",
    "ema_ratio_10", "ema_ratio_20", "rsi_14", "volatility_5",
    "volatility_10", "volatility_20", "candle_body", "candle_range",
    "upper_wick", "lower_wick", "minutes_since_open", "minutes_until_close",
    "day_of_week", "time_sin", "time_cos",
]
CLASSES = ["DOWN", "NEUTRAL", "UP"]


def test_alpaca_configuration_defaults_and_overrides():
    defaults = config.get_alpaca_settings({})
    assert defaults.api_key_id is None
    assert defaults.api_secret_key is None
    assert defaults.credentials_configured is False
    assert defaults.display_authorized is False
    assert defaults.model_path == config.BASE_DIR / "models/alpaca_final_freeze_v1/candidate_pipeline.joblib"
    assert defaults.model_spec_path == config.BASE_DIR / "models/alpaca_final_freeze_v1/frozen_candidate_spec.json"
    assert defaults.database_path == config.BASE_DIR / "data/live/ecl_alpaca_iex.sqlite3"

    settings = config.get_alpaca_settings({
        "APCA_API_KEY_ID": "unit-test-key-id",
        "APCA_API_SECRET_KEY": "unit-test-secret",
        "ALPACA_IEX_DISPLAY_AUTHORIZED": "true",
        "ALPACA_MODEL_PATH": "custom/model.joblib",
        "ALPACA_MODEL_SPEC_PATH": "custom/spec.json",
        "ALPACA_DB_PATH": "custom/live.sqlite3",
    })
    assert settings.credentials_configured is True
    assert settings.display_authorized is True
    assert settings.model_path == config.BASE_DIR / "custom/model.joblib"
    assert settings.model_spec_path == config.BASE_DIR / "custom/spec.json"
    assert settings.database_path == config.BASE_DIR / "custom/live.sqlite3"
    # Secret fields are deliberately excluded from the settings repr.
    assert "unit-test-secret" not in repr(settings)


def test_configuration_exposes_no_alternate_market_data_provider():
    assert not hasattr(config, "TWELVE_DATA_API_KEY")
    assert not hasattr(config, "TWELVE_DATA_BASE_URL")
    # Legacy provider environment names cannot affect Alpaca provider selection.
    settings = config.get_alpaca_settings({
        "TWELVE_DATA_API_KEY": "ignored-test-only",
        "TWELVE_DATA_DISPLAY_AUTHORIZED": "true",
    })
    assert settings.display_authorized is False
    assert settings.credentials_configured is False


def test_missing_credentials_and_display_gate_remain_disabled():
    settings = config.get_alpaca_settings({"ALPACA_IEX_DISPLAY_AUTHORIZED": "false"})
    assert settings.credentials_configured is False
    assert settings.display_authorized is False


def _candidate_files(tmp_path: Path, model_features=None, model_classes=None):
    artifact = tmp_path / "candidate_pipeline.joblib"
    artifact.write_bytes(b"test-only-candidate-artifact")
    spec = {
        "candidate": "unit-test-candidate",
        "features_exact_order": list(FEATURES),
        "feature_count": 24,
        "class_order": CLASSES,
        "prediction_policy": "raw three-class argmax",
        "target": {"horizon_minutes": 15},
        "source_candidate_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    spec_path = tmp_path / "frozen_candidate_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    model = SimpleNamespace(
        feature_names_in_=model_features if model_features is not None else FEATURES,
        classes_=model_classes if model_classes is not None else CLASSES,
    )
    return artifact, spec_path, model


def test_frozen_candidate_model_loads_and_contract_is_checked(tmp_path, monkeypatch):
    artifact, spec_path, expected_model = _candidate_files(tmp_path)
    monkeypatch.setattr(model_loader.joblib, "load", lambda path: expected_model)

    loaded = model_loader.load_alpaca_candidate(artifact, spec_path)

    assert loaded.model is expected_model
    assert loaded.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert loaded.specification["feature_count"] == 24


def test_frozen_candidate_hash_mismatch_is_rejected_before_load(tmp_path, monkeypatch):
    artifact, spec_path, _ = _candidate_files(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["source_candidate_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(model_loader.joblib, "load", lambda _path: pytest.fail("artifact should not be loaded"))

    with pytest.raises(ValueError, match="SHA256"):
        model_loader.load_alpaca_candidate(artifact, spec_path)


def test_frozen_candidate_feature_order_mismatch_is_rejected(tmp_path, monkeypatch):
    artifact, spec_path, model = _candidate_files(tmp_path)
    model.feature_names_in_ = list(reversed(FEATURES))
    monkeypatch.setattr(model_loader.joblib, "load", lambda _path: model)

    with pytest.raises(ValueError, match="feature order"):
        model_loader.load_alpaca_candidate(artifact, spec_path)


def test_frozen_candidate_class_order_mismatch_is_rejected(tmp_path, monkeypatch):
    artifact, spec_path, model = _candidate_files(tmp_path)
    model.classes_ = ["UP", "NEUTRAL", "DOWN"]
    monkeypatch.setattr(model_loader.joblib, "load", lambda _path: model)

    with pytest.raises(ValueError, match="class order"):
        model_loader.load_alpaca_candidate(artifact, spec_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prediction_policy", "thresholded", "prediction policy"),
        ("horizon_minutes", 30, "horizon"),
    ],
)
def test_frozen_candidate_prediction_contract_mismatch_is_rejected(tmp_path, monkeypatch, field, value, message):
    artifact, spec_path, _ = _candidate_files(tmp_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if field == "prediction_policy":
        spec[field] = value
    else:
        spec["target"][field] = value
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(model_loader.joblib, "load", lambda _path: pytest.fail("artifact should not be loaded"))

    with pytest.raises(ValueError, match=message):
        model_loader.load_alpaca_candidate(artifact, spec_path)
