"""Generate immutable historical Model Indicator records from permitted history."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_app.historical_model_signals import generate_historical_signals


def main() -> None:
    result = generate_historical_signals()
    report_dir = ROOT / "reports" / "model_indicator_v1"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "historical_generation_report.json"
    previous_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    result["initial_generation_inserted"] = previous_report.get(
        "initial_generation_inserted",
        result["records_inserted_this_run"] or result["records_in_store"],
    )
    result["idempotency_verification_inserted"] = result["records_inserted_this_run"]
    result["note"] = "Historical signal rows are immutable; repeated runs may compute the same predictions but insert no duplicate candle timestamps."
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result["report_path"] = str(report_path)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
