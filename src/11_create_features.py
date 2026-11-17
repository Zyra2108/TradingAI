
import numpy as np
import pandas as pd

from config import BASE_DIR, PREDICTION_HORIZON_MINUTES


# Project settings
CANDLE_MINUTES = 5
NEUTRAL_THRESHOLD = 0.10

HORIZON_CANDLES = (
    PREDICTION_HORIZON_MINUTES // CANDLE_MINUTES
)

FEATURE_COLUMNS = [
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
    "lower_wick"
]


def load_data():
    """Load the approved Alpaca IEX native five-minute research cache."""
    file_path = BASE_DIR / "models" / "alpaca_1min_research_v1" / "raw_ecl_iex_5min_train_validation.csv"

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)

    if "timestamp_utc" not in df:
        raise ValueError("Alpaca IEX source must include timestamp_utc")
    df["datetime"] = (pd.to_datetime(df["timestamp_utc"], utc=True)
                       .dt.tz_convert("America/New_York").dt.tz_localize(None))

    numeric_columns = [
        "open", "high", "low", "close"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def create_segments(df):
    """
    Create continuous sequences.

    A new segment starts:
    - At the beginning of each trading day.
    - Whenever a 5-minute candle is missing.
    """

    df = df.copy()

    same_day = (
        df["datetime"].dt.date ==
        df["datetime"].shift(1).dt.date
    )

    consecutive = (
        df["datetime"].diff() ==
        pd.Timedelta(minutes=CANDLE_MINUTES)
    )

    new_segment = ~(same_day & consecutive)

    df["segment_id"] = new_segment.cumsum()

    return df


def calculate_rsi(close, period=14):
    """Calculate RSI using historical price changes."""

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(
        period,
        min_periods=period
    ).mean()

    avg_loss = losses.rolling(
        period,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain == 0),
        50
    )

    return rsi


def calculate_features(group):
    """Calculate features using only current and past candles."""

    group = group.copy()

    open_price = group["open"]
    high = group["high"]
    low = group["low"]
    close = group["close"]

    # Price returns
    for period in [1, 3, 6, 12]:
        group[f"return_{period}"] = (
            close.pct_change(period) * 100
        )

    # Simple moving average ratios
    for period in [5, 10, 20]:

        sma = close.rolling(
            period,
            min_periods=period
        ).mean()

        group[f"sma_ratio_{period}"] = (
            (close / sma) - 1
        ) * 100

    # Exponential moving average ratios
    for period in [5, 10, 20]:

        ema = close.ewm(
            span=period,
            adjust=False,
            min_periods=period
        ).mean()

        group[f"ema_ratio_{period}"] = (
            (close / ema) - 1
        ) * 100

    # RSI
    group["rsi_14"] = calculate_rsi(close, 14)

    # Historical volatility
    returns = close.pct_change() * 100

    for period in [5, 10, 20]:

        group[f"volatility_{period}"] = (
            returns.rolling(
                period,
                min_periods=period
            ).std()
        )

    # Candle body
    group["candle_body"] = (
        (close - open_price) / open_price
    ) * 100

    # Candle range
    group["candle_range"] = (
        (high - low) / close
    ) * 100

    # Upper wick
    group["upper_wick"] = (
        high - pd.concat(
            [open_price, close],
            axis=1
        ).max(axis=1)
    ) / close * 100

    # Lower wick
    group["lower_wick"] = (
        pd.concat(
            [open_price, close],
            axis=1
        ).min(axis=1) - low
    ) / close * 100

    return group


def create_labels(group):
    """Create exact 15-minute future-return labels."""

    group = group.copy()

    future_close = group["close"].shift(-HORIZON_CANDLES)

    future_datetime = group["datetime"].shift(
        -HORIZON_CANDLES
    )

    expected_datetime = (
        group["datetime"] +
        pd.Timedelta(minutes=PREDICTION_HORIZON_MINUTES)
    )

    valid_target = future_datetime.eq(expected_datetime)

    group["future_datetime"] = future_datetime

    group["future_return"] = (
        (future_close / group["close"]) - 1
    ) * 100

    group.loc[
        ~valid_target,
        "future_return"
    ] = np.nan

    group["target"] = pd.Series(
    pd.NA,
    index=group.index,
    dtype="string"
    )

    group.loc[
        valid_target &
        (group["future_return"] > NEUTRAL_THRESHOLD),
        "target"
    ] = "UP"

    group.loc[
        valid_target &
        (group["future_return"] < -NEUTRAL_THRESHOLD),
        "target"
    ] = "DOWN"

    group.loc[
        valid_target &
        (group["future_return"].between(
            -NEUTRAL_THRESHOLD,
            NEUTRAL_THRESHOLD
        )),
        "target"
    ] = "NEUTRAL"

    return group


def main():

    print("=" * 65)
    print("ECL FEATURE ENGINEERING")
    print("=" * 65)

    df = load_data()

    print("\nLoaded cleaned candles:", len(df))

    # Ensure valid price records
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Missing OHLC values found.")

    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Data is not chronologically sorted.")

    if df["datetime"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    # Create continuous segments
    df = create_segments(df)

    print("Continuous segments:", df["segment_id"].nunique())

    # Calculate features and labels independently per segment
    processed_groups = []

    for _, group in df.groupby("segment_id", sort=False):

        group = calculate_features(group)

        group = create_labels(group)

        processed_groups.append(group)

    result = pd.concat(
        processed_groups,
        ignore_index=True
    )

    # Keep only valid rows
    required_columns = (
        FEATURE_COLUMNS +
        ["future_return", "target", "future_datetime"]
    )

    before = len(result)

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    result = result.dropna(
        subset=required_columns
    ).copy()

    removed = before - len(result)

    # Verify exact target horizon
    target_difference = (
        pd.to_datetime(result["future_datetime"]) -
        pd.to_datetime(result["datetime"])
    )

    if not (
        target_difference ==
        pd.Timedelta(minutes=PREDICTION_HORIZON_MINUTES)
    ).all():

        raise ValueError(
            "Invalid prediction horizon detected."
        )

    # Verify valid labels
    valid_classes = {"UP", "DOWN", "NEUTRAL"}

    if not set(result["target"].unique()).issubset(valid_classes):
        raise ValueError("Unexpected target class found.")

    # Output columns
    output_columns = [
        "datetime",
        "future_datetime",
        "close",
        "future_return",
        "target"
    ] + FEATURE_COLUMNS

    result = result[output_columns].copy()

    # Save feature dataset
    feature_output_dir = BASE_DIR / "data" / "alpaca_features_v1"
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = feature_output_dir / "ECL_5min_features.csv"

    result.to_csv(
        output_file,
        index=False
    )

    # Class distribution
    class_summary = (
        result["target"]
        .value_counts()
        .reindex(["UP", "DOWN", "NEUTRAL"], fill_value=0)
        .rename("count")
        .to_frame()
    )

    class_summary["percentage"] = (
        class_summary["count"] /
        len(result) * 100
    ).round(2)

    report_dir = BASE_DIR / "reports" / "alpaca_feature_engineering_v1"
    report_dir.mkdir(parents=True, exist_ok=True)

    class_summary.to_csv(
        report_dir / "target_class_distribution.csv"
    )

    # Final report
    print("\n" + "-" * 65)
    print("FEATURE ENGINEERING SUMMARY")
    print("-" * 65)

    print("Input candles:", len(df))
    print("Removed rows:", removed)
    print("Final feature rows:", len(result))
    print("Number of features:", len(FEATURE_COLUMNS))

    print("\nClass distribution:")
    print(class_summary.to_string())

    print("\nMissing values:")
    print(result.isna().sum().to_string())

    print("\nInfinite values:",
          np.isinf(
              result[FEATURE_COLUMNS].to_numpy()
          ).sum())

    print("\nFirst 5 rows:")
    print(result.head().to_string(index=False))

    print("\nFeature dataset saved:")
    print(output_file)

    print("\nClass report saved:")
    print(report_dir / "target_class_distribution.csv")

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
