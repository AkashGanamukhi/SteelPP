import pandas as pd
import numpy as np

from config import DATE_COLUMN, PRICE_COLUMN


def create_supervised_from_weekly(df: pd.DataFrame, n_lags: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """
    Given weekly price df, create lag features and rolling stats:
    - price_lag_1..n_lags
    - rolling_mean
    - rolling_std
    """
    df = df.copy()
    df = df.sort_values(DATE_COLUMN)

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df[PRICE_COLUMN].shift(lag)

    df["roll_mean_4"] = df[PRICE_COLUMN].rolling(window=4).mean()
    df["roll_std_4"] = df[PRICE_COLUMN].rolling(window=4).std().fillna(0.0)
    df["pct_change_1"] = df[PRICE_COLUMN].pct_change().fillna(0.0)

    # Drop rows without full history
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        f"lag_{lag}" for lag in range(1, n_lags + 1)
    ] + ["roll_mean_4", "roll_std_4", "pct_change_1"]

    X = df[feature_cols]
    y = df[PRICE_COLUMN]
    return X, y


def prepare_latest_feature_vector(
    df: pd.DataFrame,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build the latest feature row from the last n weeks,
    using the same logic as create_supervised_from_weekly.
    """
    df = df.sort_values(DATE_COLUMN).copy()
    # Reuse logic by building features and taking last row
    X, y = create_supervised_from_weekly(df, n_lags=n_lags)
    latest_X = X.iloc[[-1]].copy()
    latest_date = df[DATE_COLUMN].max()
    return latest_X, latest_date
