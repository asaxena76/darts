from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Sequence, List
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel

class DisneyModel(ABC):
    """
    Interface for models used by the harness.

    Conventions:
      - `train_df` and `infer_history_df` are pandas.DataFrames indexed by datetime
        at HOURLY frequency (local park time), with at least the target column.
      - Required columns for now: ["mrtd"] (target). Others optional for baseline.

    """

    def __init__(self, target_col: str = "mrtd"):
        self.target_col = target_col

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fit the model on (roughly) last 12 months up to a 3am cutoff of the same day."""
        ...

    @abstractmethod
    def predict_until_close(
            self,
            infer_history_df: pd.DataFrame,
            park_close_ts: pd.Timestamp,
    ) -> pd.Series:
        """
        Predict per-MINUTE target from the *next minute after infer_history_df.index.max()*
        up to `park_close_ts` (exclusive). Return a pd.Series (DatetimeIndex, minutely).
        """
        ...


class NaiveLastHourCarryForward(DisneyModel):
    """
    Dummy baseline:
      - Use the last observed HOURLY mrtd value in infer_history_df.
      - Expand it to per-minute for the rest of the day.
      - If no last value exists (NaN or empty), predict 0.

    This gives us a trivial, stable baseline for harness wiring & metrics.
    """

    def fit(self, train_df: pd.DataFrame) -> None:
        # No-op for naive baseline.
        return

    def predict_until_close(
            self,
            infer_history_df: pd.DataFrame,
            park_close_ts: pd.Timestamp,
    ) -> pd.Series:
        if infer_history_df.empty:
            last_val = 0.0
        else:
            last_val = infer_history_df[self.target_col].dropna().iloc[-1] if not infer_history_df[self.target_col].dropna().empty else 0.0

        start = infer_history_df.index.max()
        # start at the next hour after the latest history timestamp
        start_minute = (start + pd.Timedelta(hours=1)).ceil("min")

        if start_minute >= park_close_ts:
            return pd.Series(dtype=float)

        idx = pd.date_range(start=start_minute, end=park_close_ts - pd.Timedelta(minutes=1), freq="min")
        return pd.Series(last_val, index=idx, name=self.target_col)


class DartsLGBMDisneyModel(DisneyModel):
    """
    LightGBMModel (darts) implementation that trains once per attraction (on hourly data),
    and predicts hourly to park-close at inference; results are expanded to *per-minute*
    by forward-filling within each hour (to match the harness interface).
    """

    def __init__(
        self,
        target_col: str = "mrtd",
        covariate_cols: Optional[Sequence[str]] = None,
        lags: int = 24,
        lags_past_covariates: int = 2,
        output_chunk_length: int = 24,
        random_state: int = 42,
    ):
        super().__init__(target_col=target_col)
        self.covariate_cols: List[str] = list(covariate_cols or [])
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.output_chunk_length = output_chunk_length
        self.random_state = random_state

        self.model: Optional[LightGBMModel] = None
        self.ts_scaler: Optional[Scaler] = None
        self.cov_scaler: Optional[Scaler] = None
        self._trained = False

    def _build_series(self, df: pd.DataFrame) -> tuple[TimeSeries, Optional[TimeSeries]]:
        # df is indexed by datetime at hourly freq (as ensured by harness)
        # Fill NaNs as 0 (per project cleaning rule)
        d = df.copy()
        d[self.target_col] = d[self.target_col].fillna(0.0)

        y_ts = TimeSeries.from_dataframe(
            d.reset_index().rename(columns={d.index.name or "index": "ts"}),
            time_col="ts",
            value_cols=self.target_col,
            freq="h",
            fill_missing_dates=False,
            fillna_value=0.0,
        )

        cov_ts = None
        if self.covariate_cols:
            present = [c for c in self.covariate_cols if c in d.columns]
            if present:
                cov_ts = TimeSeries.from_dataframe(
                    d.reset_index().rename(columns={d.index.name or "index": "ts"}),
                    time_col="ts",
                    value_cols=present,
                    freq="h",
                    fill_missing_dates=False,
                    fillna_value=0.0,
                )
        return y_ts, cov_ts

    def fit(self, train_df: pd.DataFrame) -> None:
        # train_df: single-attraction, index is hourly datetime
        if train_df.empty:
            self._trained = False
            return

        y_ts, cov_ts = self._build_series(train_df)

        # Scale target and covariates
        self.ts_scaler = Scaler()
        y_s = self.ts_scaler.fit_transform(y_ts)

        cov_s = None
        if cov_ts is not None:
            self.cov_scaler = Scaler()
            cov_s = self.cov_scaler.fit_transform(cov_ts)

        self.model = LightGBMModel(
            lags=self.lags,
            output_chunk_length=self.output_chunk_length,
            lags_past_covariates=self.lags_past_covariates if cov_s is not None else None,
            random_state=self.random_state,
        )
        self.model.fit(y_s, past_covariates=cov_s)
        self._trained = True

    def predict_until_close(
        self,
        infer_history_df: pd.DataFrame,
        park_close_ts: pd.Timestamp,
    ) -> pd.Series:
        if not self._trained or self.model is None or self.ts_scaler is None:
            return pd.Series(dtype=float)
        if infer_history_df.empty:
            return pd.Series(dtype=float)

        # Determine start minute (next minute after last observed hour)
        last_obs: pd.Timestamp = infer_history_df.index.max()
        start_minute = (last_obs + pd.Timedelta(hours=1)).ceil("min")
        if start_minute >= park_close_ts:
            return pd.Series(dtype=float)

        # Build hourly history series (up to last observed hour)
        y_hist_ts, cov_hist_ts = self._build_series(infer_history_df)

        # Transform with fitted scalers
        y_hist_s = self.ts_scaler.transform(y_hist_ts)
        cov_hist_s = None
        if cov_hist_ts is not None and self.cov_scaler is not None:
            cov_hist_s = self.cov_scaler.transform(cov_hist_ts)

        # How many *hourly* steps until close? (start at next full hour)
        start_hour = (last_obs + pd.Timedelta(hours=1)).floor("h")
        # create an hourly range [start_hour, park_close_ts) with step 1h
        if start_hour >= park_close_ts:
            return pd.Series(dtype=float)
        hourly_targets = pd.date_range(start=start_hour, end=park_close_ts - pd.Timedelta(hours=1), freq="h")
        n_steps = len(hourly_targets)
        if n_steps <= 0:
            return pd.Series(dtype=float)

        # Predict hourly steps
        pred_s = self.model.predict(n=n_steps, series=y_hist_s, past_covariates=cov_hist_s)
        pred_y = self.ts_scaler.inverse_transform(pred_s)

        # Convert hourly predictions to per-minute series by ffill within each hour
        pred_df = pred_y.to_dataframe(time_as_index=True)  # index is hourly timestamps
        # Up-sample to minutes and forward-fill
        minute_index = pd.date_range(start=start_minute, end=park_close_ts - pd.Timedelta(minutes=1), freq="min")

        # Build a minutely series by reindexing the hourly preds over a dense minutely range then ffill
        # First map hourly index -> value, then resample
        hourly_series = pred_df.iloc[:, 0]  # first/only column is the target
        # Ensure we include the first minute: align to the hour grid and then ffill
        dense_minute = pd.date_range(
            start=hourly_series.index.min(),
            end=park_close_ts - pd.Timedelta(minutes=1),
            freq="min",
        )
        minute_filled = hourly_series.reindex(dense_minute).ffill()
        minute_filled = minute_filled.reindex(minute_index)
        minute_filled.name = self.target_col
        return minute_filled
