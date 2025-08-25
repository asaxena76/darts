from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

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
        # start at the next minute after the latest history timestamp
        start_minute = (start + pd.Timedelta(minutes=1)).ceil("min")

        if start_minute >= park_close_ts:
            return pd.Series(dtype=float)

        idx = pd.date_range(start=start_minute, end=park_close_ts - pd.Timedelta(minutes=1), freq="min")
        return pd.Series(last_val, index=idx, name=self.target_col)
