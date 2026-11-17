
"""
Step 17: Add time-based features to the ECL dataset.

Adds:
1. minutes_since_open
2. minutes_until_close
3. day_of_week
4. time_sin
5. time_cos

Uses the official NYSE trading calendar, including early closes.

Important:
- Historical timestamps represent candle opening times.
- Features are calculated using the candle completion time.
- No future price information is used.
- Original datasets are not modified.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "alpaca_features_v1" / "ECL_5min_features.csv"

OUTPUT_FILE = BASE_DIR / "data" / "alpaca_features_v1" / "ECL_5min_features_time.csv"

REPORT_DIR = BASE_DIR / "reports" / "alpaca_feature_engineering_v1"

CANDLE_MINUTES = 5

TIME_FEATURES = [
    "minutes_since_open",
    "minutes_until_close",
    "day_of_week",
    "time_sin",
    "time_cos",
]

EXISTING_FEATURES = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "sma_ratio_5",
    "sma_ratio_10",
    "sma_ratio_20",
    "ema_ratio_5",
    "ema_ratio_10",
    "ema_ratio_20",
    "rsi_14",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "candle_body",
    "candle_range",
    "upper_wick",
    "lower_wick",
]

TIMEZONE = "America/New_York"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    df["datetime"] = pd.to_datetime(df["datetime"])

    if df["datetime"].dt.tz is not None:
        raise ValueError(
            "Expected naive New York local timestamps. "
            "Do not silently convert timezone-aware input."
        )

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Input data is not chronologically sorted.")

    missing = [
        col for col in EXISTING_FEATURES + ["close"]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ---------------------------------------------------------
# GET OFFICIAL NYSE SCHEDULE
# ---------------------------------------------------------

def get_nyse_schedule(df):

    start_date = df["datetime"].min().date()
    end_date = df["datetime"].max().date()

    calendar = mcal.get_calendar("NYSE")

    schedule = calendar.schedule(
        start_date=start_date,
        end_date=end_date
    )

    if schedule.empty:
        raise ValueError("NYSE calendar returned an empty schedule.")

    # Calendar schedule times are timezone-aware.
    # Convert them into New York local time.
    schedule["market_open_local"] = (
        schedule["market_open"]
        .dt.tz_convert(TIMEZONE)
        .dt.tz_localize(None)
    )

    schedule["market_close_local"] = (
        schedule["market_close"]
        .dt.tz_convert(TIMEZONE)
        .dt.tz_localize(None)
    )

    schedule["trading_date"] = schedule.index.date

    return schedule


# ---------------------------------------------------------
# CREATE TIME FEATURES
# ---------------------------------------------------------

def add_time_features(df, schedule):

    df = df.copy()

    session_lookup = schedule.set_index(
        "trading_date"
    )[
        ["market_open_local", "market_close_local"]
    ]

    trading_dates = df["datetime"].dt.date

    # Match every candle to its official NYSE session.
    df["market_open"] = trading_dates.map(
        session_lookup["market_open_local"]
    )

    df["market_close"] = trading_dates.map(
        session_lookup["market_close_local"]
    )

    # All features must be available after the candle closes.
    candle_close_time = (
        df["datetime"]
        + pd.Timedelta(minutes=CANDLE_MINUTES)
    )

    # -----------------------------------------------------
    # VALIDATE SESSION BOUNDARIES
    # -----------------------------------------------------

    missing_session = (
        df["market_open"].isna()
        | df["market_close"].isna()
    )

    if missing_session.any():
        bad_dates = sorted(
            set(trading_dates[missing_session])
        )

        raise ValueError(
            f"Candles found outside the NYSE calendar: {bad_dates[:10]}"
        )

    candle_inside_session = (
        (df["datetime"] >= df["market_open"])
        & (candle_close_time <= df["market_close"])
    )

    if not candle_inside_session.all():
        invalid_count = (~candle_inside_session).sum()

        raise ValueError(
            f"{invalid_count} candles are outside valid "
            "NYSE session boundaries."
        )

    # -----------------------------------------------------
    # 1. MINUTES SINCE MARKET OPEN
    # -----------------------------------------------------

    df["minutes_since_open"] = (
        candle_close_time - df["market_open"]
    ).dt.total_seconds() / 60

    # -----------------------------------------------------
    # 2. MINUTES UNTIL MARKET CLOSE
    # -----------------------------------------------------

    df["minutes_until_close"] = (
        df["market_close"] - candle_close_time
    ).dt.total_seconds() / 60

    # -----------------------------------------------------
    # 3. DAY OF WEEK
    # -----------------------------------------------------

    df["day_of_week"] = df["datetime"].dt.day_of_week

    # -----------------------------------------------------
    # 4. CYCLICAL SESSION PROGRESS
    # -----------------------------------------------------

    session_duration = (
        df["market_close"] - df["market_open"]
    ).dt.total_seconds() / 60

    if (session_duration <= 0).any():
        raise ValueError("Invalid trading session duration.")

    session_progress = (
        df["minutes_since_open"] / session_duration
    )

    # A full session moves through one cycle.
    df["time_sin"] = np.sin(
        2 * np.pi * session_progress
    )

    df["time_cos"] = np.cos(
        2 * np.pi * session_progress
    )

    # -----------------------------------------------------
    # REMOVE TEMPORARY COLUMNS
    # -----------------------------------------------------

    df = df.drop(
        columns=[
            "market_open",
            "market_close",
        ]
    )

    return df


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("ECL TIME-BASED FEATURE ENGINEERING")
    print("=" * 70)

    # Load original feature dataset.
    df = load_data()

    print(f"\nInput feature rows: {len(df):,}")
    print(f"Original features:  {len(EXISTING_FEATURES)}")

    # Get official NYSE schedule.
    schedule = get_nyse_schedule(df)

    print(f"NYSE sessions in calendar: {len(schedule):,}")

    # Add time features.
    result = add_time_features(df, schedule)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("TIME FEATURE VALIDATION")
    print("-" * 70)

    checks = {
        "Row count unchanged": len(result) == len(df),

        "No missing time features": (
            result[TIME_FEATURES].isna().sum().sum() == 0
        ),

        "No infinite time features": (
            np.isfinite(
                result[TIME_FEATURES].to_numpy(dtype=float)
            ).all()
        ),

        "Minutes since open non-negative": (
            (result["minutes_since_open"] >= 0).all()
        ),

        "Minutes until close non-negative": (
            (result["minutes_until_close"] >= 0).all()
        ),

        "Day of week between 0 and 4": (
            result["day_of_week"].between(0, 4).all()
        ),

        "All original feature values unchanged": (
            result[EXISTING_FEATURES].equals(
                df[EXISTING_FEATURES]
            )
        ),

        "All original timestamps unchanged": (
            result["datetime"].equals(df["datetime"])
        ),
    }

    all_passed = True

    for description, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {description}")

        if not passed:
            all_passed = False

    if not all_passed:
        print("\nERROR: Time feature validation failed.")
        sys.exit(1)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "-" * 70)
    print("TIME FEATURE SUMMARY")
    print("-" * 70)

    print(
        result[TIME_FEATURES]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nFirst 5 time-feature rows:")

    print(
        result[
            ["datetime"] + TIME_FEATURES
        ].head().to_string(index=False)
    )

    # -----------------------------------------------------
    # SAVE EXPERIMENTAL DATASET
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Save descriptive report.
    summary_path = (
        REPORT_DIR / "time_feature_summary.csv"
    )

    result[TIME_FEATURES].describe().to_csv(
        summary_path
    )

    print("\n" + "=" * 70)
    print("TIME FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(result):,}")
    print(f"Original features: {len(EXISTING_FEATURES)}")
    print(f"New time features: {len(TIME_FEATURES)}")
    print(f"Total features: {len(EXISTING_FEATURES) + len(TIME_FEATURES)}")

    print(f"\nExperimental dataset saved:\n{OUTPUT_FILE}")
    print(f"Summary saved:\n{summary_path}")

    print("\nOriginal feature dataset was not modified.")

    print("=" * 70)


if __name__ == "__main__":
    main()
