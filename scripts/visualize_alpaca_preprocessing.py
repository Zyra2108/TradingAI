"""Create before/after plots for the permitted Alpaca IEX ECL 5-minute cache.

This is a read-only visualization. It reads only the 2025-11-13 through
2026-09-24 train/validation source listed in its fetch metadata. It never
opens the full-year cache, a split test file, model artifacts, or live data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/alpaca_1min_research_v1/raw_ecl_iex_5min_train_validation.csv"
METADATA = ROOT / "models/alpaca_1min_research_v1/fetch_5min_metadata.json"
DEFAULT_OUTPUT = ROOT / "reports/data_preprocessing_visuals_v1"
MIN_DATE = "2025-11-13"
MAX_DATE = "2026-09-24"
REQUIRED = {"timestamp_utc", "open", "high", "low", "close", "volume"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_approved_input() -> tuple[pd.DataFrame, dict]:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    expected = metadata.get("raw_csv_sha256")
    allowed = metadata.get("permitted_new_york_date_range")
    if (metadata.get("symbol") != "ECL" or metadata.get("feed") != "iex"
            or metadata.get("timeframe") != "5Min" or allowed != [MIN_DATE, MAX_DATE]
            or metadata.get("raw_csv") != SOURCE.name or not expected
            or sha256(SOURCE) != expected):
        raise ValueError("Approved ECL IEX train/validation source provenance check failed")

    frame = pd.read_csv(SOURCE)
    if not REQUIRED.issubset(frame.columns):
        raise ValueError("Approved cache has an unexpected column schema")
    frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True, errors="raise")
    frame["datetime_ny"] = frame.timestamp_utc.dt.tz_convert("America/New_York").dt.tz_localize(None)
    local_dates = frame.datetime_ny.dt.strftime("%Y-%m-%d")
    if (local_dates.lt(MIN_DATE) | local_dates.gt(MAX_DATE)).any():
        raise ValueError("Source contains a date outside the approved pre-test window")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, metadata


def preprocess(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ordered = frame.sort_values("datetime_ny", kind="stable").copy()
    duplicate = ordered.datetime_ny.duplicated(keep="first")
    deduplicated = ordered.loc[~duplicate].copy()
    ohlc = deduplicated[["open", "high", "low", "close"]].to_numpy(float)
    finite = np.isfinite(ohlc).all(axis=1)
    positive = (ohlc > 0).all(axis=1)
    structure = (
        (deduplicated.high >= deduplicated[["open", "close", "low"]].max(axis=1))
        & (deduplicated.low <= deduplicated[["open", "close", "high"]].min(axis=1))
    ).to_numpy()
    valid = finite & positive & structure
    valid_ohlc = deduplicated.loc[valid].copy()

    start, end = valid_ohlc.datetime_ny.min().date(), valid_ohlc.datetime_ny.max().date()
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    expected: set[pd.Timestamp] = set()
    for _, session in schedule.iterrows():
        opened = session.market_open.tz_convert("America/New_York").tz_localize(None)
        closed = session.market_close.tz_convert("America/New_York").tz_localize(None)
        expected.update(pd.date_range(opened, closed - pd.Timedelta(minutes=5), freq="5min"))
    session_mask = valid_ohlc.datetime_ny.isin(expected)
    cleaned = valid_ohlc.loc[session_mask].copy()
    zero_volume = cleaned.volume.eq(0)
    cleaned.loc[zero_volume, "volume"] = np.nan  # mark unknown; never create a candle or a volume value
    cleaned = cleaned.sort_values("datetime_ny", kind="stable").reset_index(drop=True)
    counts = {
        "input_rows": int(len(frame)),
        "duplicate_rows_removed": int(duplicate.sum()),
        "invalid_ohlc_rows_removed": int((~valid).sum()),
        "outside_or_misaligned_session_rows_removed": int((~session_mask).sum()),
        "zero_volume_rows_marked_missing": int(zero_volume.sum()),
        "output_rows": int(len(cleaned)),
        "missing_expected_intervals_preserved": int(len(expected - set(cleaned.datetime_ny))),
    }
    return cleaned, counts


def _plot_price(ax, frame: pd.DataFrame, color: str, expected_index: pd.DatetimeIndex) -> None:
    if frame.empty:
        ax.text(.5, .5, "No valid candles", ha="center", va="center", transform=ax.transAxes)
        return
    # Draw separate observed segments, so neither IEX gaps nor overnight and
    # session boundaries are connected. No OHLC row or missing value is added.
    actual = frame.sort_values("datetime_ny", kind="stable")
    new_segment = actual.datetime_ny.diff().ne(pd.Timedelta(minutes=5))
    new_segment |= actual.datetime_ny.dt.date.ne(actual.datetime_ny.shift().dt.date)
    for segment_number, (_, segment) in enumerate(actual.groupby(new_segment.cumsum(), sort=False)):
        ax.plot(segment.datetime_ny, segment.close, color=color, linewidth=.65,
                label="Observed close" if segment_number == 0 else None)
    ax.scatter(actual.datetime_ny, actual.close, s=1.2, color=color, alpha=.45, linewidths=0)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def generate(output: Path) -> dict:
    expected_outputs = {
        "before_after_preprocessing.png", "volume_and_return_distributions.png",
        "preprocessing_visual_report.json",
    }
    if output.exists():
        unexpected = {item.name for item in output.iterdir()} - expected_outputs
        if unexpected:
            raise FileExistsError(f"Refusing to overwrite unrelated output files: {sorted(unexpected)}")
    output.mkdir(parents=True, exist_ok=True)
    before, metadata = load_approved_input()
    after, counts = preprocess(before)

    sessions = mcal.get_calendar("NYSE").schedule(start_date=MIN_DATE, end_date=MAX_DATE)
    expected_index = pd.DatetimeIndex([])
    expected_values = []
    for _, session in sessions.iterrows():
        opened = session.market_open.tz_convert("America/New_York").tz_localize(None)
        closed = session.market_close.tz_convert("America/New_York").tz_localize(None)
        expected_values.extend(pd.date_range(opened, closed - pd.Timedelta(minutes=5), freq="5min"))
    expected_index = pd.DatetimeIndex(expected_values)

    # Mark raw rows failing the same OHLC gate or session-alignment gate.
    raw_sorted = before.sort_values("datetime_ny", kind="stable")
    raw_regular = raw_sorted[raw_sorted.datetime_ny.isin(expected_index)]
    outside = raw_sorted.loc[~raw_sorted.datetime_ny.isin(expected_index)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    _plot_price(axes[0], raw_regular, "#c76536", expected_index)
    if len(outside):
        axes[0].scatter(outside.datetime_ny, outside.close, marker="x", s=28, color="#b42318",
                        label="Outside / misaligned session row")
        axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].set_title(f"Before preprocessing · raw approved Alpaca IEX ECL 5-minute bars · {len(before):,} rows")
    axes[0].set_ylabel("Close ($)")
    axes[0].grid(alpha=.18)
    _plot_price(axes[1], after, "#13795b", expected_index)
    axes[1].set_title(f"After preprocessing · NYSE regular-session bars · {len(after):,} rows")
    axes[1].set_ylabel("Close ($)")
    axes[1].grid(alpha=.18)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("ECL market data before and after preprocessing (pre-test dates only)", fontsize=14, weight="bold")
    fig.savefig(output / "before_after_preprocessing.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for label, frame, color in (("Before", before, "#c76536"), ("After", after, "#13795b")):
        valid_volume = frame.volume.dropna()
        axes[0].hist(valid_volume.clip(upper=valid_volume.quantile(.99)), bins=45, alpha=.5,
                     color=color, label=label)
        ordered = frame.sort_values("datetime_ny", kind="stable").copy()
        if label == "Before":
            ordered = ordered.loc[ordered.datetime_ny.isin(expected_index)]
        same_segment = ordered.datetime_ny.diff().eq(pd.Timedelta(minutes=5))
        same_segment &= ordered.datetime_ny.dt.date.eq(ordered.datetime_ny.shift().dt.date)
        returns = (ordered.close.pct_change() * 100).where(same_segment).dropna()
        axes[1].hist(returns.clip(lower=returns.quantile(.01), upper=returns.quantile(.99)),
                     bins=60, alpha=.5, color=color, label=label)
    axes[0].set_title("Observed candle volume (upper 1% clipped for display only)")
    axes[0].set_xlabel("Volume")
    axes[1].set_title("Adjacent-row close change (upper/lower 1% clipped for display only)")
    axes[1].set_xlabel("Percent change")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
    fig.savefig(output / "volume_and_return_distributions.png", dpi=170)
    plt.close(fig)

    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "symbol": "ECL", "feed": "Alpaca IEX", "timeframe": "5Min",
        "permitted_new_york_dates": [MIN_DATE, MAX_DATE],
        "input_timestamp_range_utc": [before.timestamp_utc.min().isoformat(), before.timestamp_utc.max().isoformat()],
        "input_timestamp_range_new_york": [before.datetime_ny.min().isoformat(), before.datetime_ny.max().isoformat()],
        "clean_timestamp_range_new_york": [after.datetime_ny.min().isoformat(), after.datetime_ny.max().isoformat()],
        "preprocessing_counts": counts,
        "provenance_metadata_confirms_no_credentials_saved": metadata.get("credential_values_saved") is False,
        "test_dataset_accessed": False,
        "outputs": ["before_after_preprocessing.png", "volume_and_return_distributions.png", "preprocessing_visual_report.json"],
        "plot_note": "Price traces and close changes reset at missing five-minute timestamps and session boundaries. No missing OHLC bars are filled or synthesized.",
    }
    (output / "preprocessing_visual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"new output directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = generate(output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
