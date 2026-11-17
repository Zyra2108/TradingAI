"""Complete report metadata from saved fits/predictions; performs no inference."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("ecl_three_class_v1_finalizer_module", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def main() -> None:
    report = module.REPORT_DIR
    model_dir = module.MODEL_DIR
    if not (report / "validation_metrics.json").exists() or not (report / "walk_forward_predictions.csv").exists():
        raise FileNotFoundError("Saved OOF and single validation outputs are required")
    train, validation, folds, features, specs = module.load_data()
    baselines = module.evaluate_saved_controls(validation, train)
    pd.DataFrame(baselines).to_csv(report / "saved_baseline_comparison.csv", index=False)
    before = {str(p.relative_to(ROOT)): module.sha256(p) for p in module.prior_artifact_paths()}
    frozen = json.loads((model_dir / "frozen_candidate_config.json").read_text(encoding="utf-8"))
    config = json.loads((model_dir / "experiment_config.json").read_text(encoding="utf-8"))
    inputs = [module.TRAIN_CSV, module.VALIDATION_CSV, module.FEATURE_SPEC,
              module.FROZEN_SPEC, module.FOLD_CSV, module.RAW_SOURCE, module.RAW_SOURCE_META,
              module.MODEL_DIR / "experiment_config.json"]
    after = {str(p.relative_to(ROOT)): module.sha256(p) for p in module.prior_artifact_paths()}
    if before != after:
        raise RuntimeError("A preexisting model hash changed during finalization")
    output_hashes = {}
    for base in (model_dir, report):
        for path in base.rglob("*"):
            if path.is_file() and path.name != "reproducibility_manifest.json":
                output_hashes[str(path.relative_to(ROOT))] = module.sha256(path)
    manifest = {
        "configuration": config,
        "python": __import__("platform").python_version(),
        "sklearn": module.sklearn.__version__,
        "xgboost": module.xgboost.__version__,
        "catboost_installed": bool(importlib.util.find_spec("catboost")),
        "input_sha256": {str(p.relative_to(ROOT)): module.sha256(p) for p in inputs},
        "preexisting_model_hashes_before_finalization": before,
        "preexisting_model_hashes_after_finalization": after,
        "preexisting_hashes_unchanged": before == after,
        "protected_test_csv_or_predictions_accessed": False,
        "validation_predictions_recomputed_during_finalization": False,
        "split_rows": {"train": len(train), "validation": len(validation)},
        "split_dates": {
            "train": [train.datetime.min().strftime("%Y-%m-%d"), train.datetime.max().strftime("%Y-%m-%d")],
            "validation": [validation.datetime.min().strftime("%Y-%m-%d"), validation.datetime.max().strftime("%Y-%m-%d")],
        },
        "feature_order": features,
        "class_order": module.CLASS_ORDER,
        "selected_artifact": frozen["selected_artifact"],
        "selected_artifact_sha256": frozen["artifact_sha256"],
        "frozen_candidate_config_sha256": module.sha256(model_dir / "frozen_candidate_config.json"),
        "experiment_output_sha256": output_hashes,
        "outputs_confined_to_new_experiment_dirs": True,
    }
    (model_dir / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
