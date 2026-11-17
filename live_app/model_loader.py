"""Integrity-checked loader for the frozen Alpaca ECL candidate pipeline."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from config import ALPACA_MODEL_PATH, ALPACA_MODEL_SPEC_PATH

EXPECTED_CLASSES = ["DOWN", "NEUTRAL", "UP"]
EXPECTED_HORIZON_MINUTES = 15
EXPECTED_POLICY = "raw three-class argmax"


@dataclass(frozen=True)
class LoadedCandidate:
    model: Any
    specification: dict
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_alpaca_candidate(
    model_path: Path | str | None = None,
    spec_path: Path | str | None = None,
) -> LoadedCandidate:
    """Load and validate the Alpaca candidate without touching legacy models."""
    artifact_path = Path(model_path or ALPACA_MODEL_PATH)
    specification_path = Path(spec_path or ALPACA_MODEL_SPEC_PATH)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Alpaca candidate model not found: {artifact_path}")
    if not specification_path.is_file():
        raise FileNotFoundError(f"Alpaca candidate specification not found: {specification_path}")

    with specification_path.open("r", encoding="utf-8") as stream:
        specification = json.load(stream)

    expected_hash = specification.get("source_candidate_sha256")
    actual_hash = sha256_file(artifact_path)
    if not expected_hash or actual_hash.lower() != str(expected_hash).lower():
        raise ValueError("Alpaca candidate SHA256 does not match frozen specification")

    # If the adjacent freeze hash manifest is available, enforce it too.
    hash_manifest = artifact_path.parent / "SHA256SUMS.json"
    if hash_manifest.is_file():
        with hash_manifest.open("r", encoding="utf-8") as stream:
            hashes = json.load(stream)
        expected_manifest_hash = hashes.get(artifact_path.name)
        if not expected_manifest_hash or actual_hash.lower() != str(expected_manifest_hash).lower():
            raise ValueError("Alpaca candidate SHA256 does not match freeze hash manifest")

    features = specification.get("features_exact_order")
    if not isinstance(features, list) or len(features) != 24 or len(set(features)) != 24:
        raise ValueError("Frozen Alpaca specification must contain exactly 24 unique ordered features")
    if specification.get("feature_count") != 24:
        raise ValueError("Frozen Alpaca feature count is not 24")
    if specification.get("class_order") != EXPECTED_CLASSES:
        raise ValueError("Frozen Alpaca specification class order is invalid")
    target = specification.get("target", {})
    if target.get("horizon_minutes") != EXPECTED_HORIZON_MINUTES:
        raise ValueError("Frozen Alpaca prediction horizon is not 15 minutes")
    if specification.get("prediction_policy") != EXPECTED_POLICY:
        raise ValueError("Frozen Alpaca prediction policy is not raw three-class argmax")

    model = joblib.load(artifact_path)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != features:
        raise ValueError("Loaded Alpaca model feature order does not match frozen specification")
    model_classes = [str(value) for value in getattr(model, "classes_", [])]
    if model_classes != EXPECTED_CLASSES:
        raise ValueError("Loaded Alpaca model class order must be DOWN, NEUTRAL, UP")

    return LoadedCandidate(model=model, specification=specification, sha256=actual_hash)
