from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    N_LAGS,
    TEST_SIZE_FRACTION,
    RANDOM_STATE,
    ANN_HIDDEN_LAYER_SIZES,
    ANN_MAX_ITER,
)
from src.features import create_supervised_from_weekly, prepare_latest_feature_vector
from src.utils.date_utils import add_days


class TimeSeriesANN:
    """
    ANN-based regressor on weekly data, as per the reference paper's
    conclusion that ANN beats RF for closing price prediction.

    Training:
      - Input: lag features + rolling stats from weekly prices
      - Output: next-week price

    Forecast:
      - Autoregressive: iteratively predicts future weekly prices
      - Then interpolates to daily for the next N days from current_date.
    """

    def __init__(
        self,
        n_lags: int = N_LAGS,
        hidden_layer_sizes=ANN_HIDDEN_LAYER_SIZES,
        max_iter: int = ANN_MAX_ITER,
        random_state: int = RANDOM_STATE,
    ):
        self.n_lags = n_lags
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

        self.model: MLPRegressor | None = None
        self.scaler_X: StandardScaler | None = None
        self.scaler_y: StandardScaler | None = None
        self.feature_cols: list[str] | None = None

    def fit(self, df_weekly: pd.DataFrame) -> None:
        X, y = create_supervised_from_weekly(df_weekly, n_lags=self.n_lags)
        self.feature_cols = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE_FRACTION,
            shuffle=False,  # keep temporal order
        )

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_train_scaled, y_train_scaled)

    def _predict_next_week_price(self, feature_row: pd.DataFrame) -> float:
        assert self.model is not None, "Model not fitted."
        assert self.scaler_X is not None and self.scaler_y is not None

        X_scaled = self.scaler_X.transform(feature_row)
        y_scaled_pred = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
        return float(y_pred[0])

    def forecast_daily(
        self,
        df_weekly: pd.DataFrame,
        current_date: date,
        horizon_days: int,
    ) -> pd.DataFrame:
        """
        1. Predict sufficient weekly steps beyond current_date + horizon_days.
        2. Interpolate weekly to daily.
        3. Return daily prices from current_date (inclusive) for horizon_days.
        """
        df_weekly = df_weekly.sort_values(DATE_COLUMN).copy()

        # Make sure model is fitted
        if self.model is None:
            self.fit(df_weekly)

        # Build initial feature vector
        latest_X, latest_date = prepare_latest_feature_vector(df_weekly, n_lags=self.n_lags)
        current_week_date = latest_date

        # We will simulate a copy of the weekly series including predictions
        sim_df = df_weekly[[DATE_COLUMN, PRICE_COLUMN]].copy()

        # Predict weekly prices until we cover horizon
        horizon_end_date = current_date + timedelta(days=horizon_days)
        # Use pandas Timestamp for robust comparisons with pandas Timestamps
        horizon_end_ts = pd.Timestamp(horizon_end_date)

        while pd.Timestamp(current_week_date) < horizon_end_ts:
            next_price = self._predict_next_week_price(latest_X)

            next_date = current_week_date + timedelta(weeks=1)
            sim_df = pd.concat(
                [
                    sim_df,
                    pd.DataFrame({DATE_COLUMN: [next_date], PRICE_COLUMN: [next_price]}),
                ],
                ignore_index=True,
            )

            # Recompute latest_X from extended series
            latest_X, current_week_date = prepare_latest_feature_vector(
                sim_df, n_lags=self.n_lags
            )

        sim_df = sim_df.drop_duplicates(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)

        # Interpolate to daily
        daily_index = pd.date_range(
            start=current_date,
            end=horizon_end_date,
            freq="D",
        )

        interpolated = (
            sim_df.set_index(DATE_COLUMN)[PRICE_COLUMN]
            .asfreq("D")
            .interpolate(method="linear")
        )

        daily_forecast = interpolated.reindex(daily_index).rename("base_price").reset_index()
        daily_forecast = daily_forecast.rename(columns={"index": "date"})
        # If some early days NaN (gap before last weekly), forward-fill
        daily_forecast["base_price"] = daily_forecast["base_price"].ffill().bfill()

        return daily_forecast
