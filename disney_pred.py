# fix python path if working locally
#from utils import fix_pythonpath_if_working_locally
#fix_pythonpath_if_working_locally()
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)
import matplotlib.pyplot as plt
import numpy as np
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel
from darts.metrics import mae
import pandas as pd
from darts import TimeSeries
df = pd.read_csv("tf_ll.csv", parse_dates=["pred_hour_pdt"])
df = df.fillna(0.0)


# --- Pretty printing helpers ---
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

def _ts_df(ts):
    df_ = ts.to_dataframe(time_as_index=False)
    if "pred_hour_pdt" in df_.columns:
        df_ = df_.rename(columns={"pred_hour_pdt": "ts"})
    return df_

def show_window(title, y_ts, cov_ts=None, limit=None):
    y_df = _ts_df(y_ts).copy()
    y_cols = [c for c in y_df.columns if c != "ts"]
    y_name = y_cols[0]
    y_df = y_df.rename(columns={y_name: f"target[{y_name}]"})
    if cov_ts is not None:
        cdf = _ts_df(cov_ts)
        df = y_df.merge(cdf, on="ts", how="left")
    else:
        df = y_df
    df = df.sort_values("ts")
    if limit is not None and len(df) > limit:
        df = df.tail(limit)
    print(f"\n=== {title} (rows={len(y_df)}) ===")
    print(df.round(3).to_string(index=False))

def show_forecast(title, actual_ts, pred_ts):
    a = _ts_df(actual_ts).rename(columns={c: "actual" for c in _ts_df(actual_ts).columns if c != "ts"})
    p = _ts_df(pred_ts).rename(columns={c: "pred"   for c in _ts_df(pred_ts).columns if c != "ts"})
    df = a.merge(p, on="ts", how="outer").sort_values("ts")
    df["abs_err"] = (df["actual"] - df["pred"]).abs()
    print(f"\n=== {title} (rows={len(df)}) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


PAST_LAGS = 2  # uses covariate lags -1 and -2

# ADD: pick covariate columns you want to use (must exist & be numeric)
covariate_cols = [c for c in [
    "pre_hour", "mwt", "mrtd_prev", "mwt_prev", "am_prev", "om_prev",
    "mrtd_sum_prev", "mwt_sum_prev", "am_sum_prev", "om_sum_prev",
] if c in df.columns]
value_col = "mrtd"  # change to another metric if you prefer (e.g., "mwt")
series_by_ride = {
    name: TimeSeries.from_dataframe(
        g.sort_values("pred_hour_pdt"),
        time_col="pred_hour_pdt",
        value_cols=value_col,
        freq="h",
        fillna_value=0.0,
        fill_missing_dates=False,
    )
    for name, g in df.groupby("gw_name", sort=False)
}
# ADD: build past covariates per ride (same time index as target)
past_covs_by_ride = {
    name: TimeSeries.from_dataframe(
        g.sort_values("pred_hour_pdt"),
        time_col="pred_hour_pdt",
        value_cols=covariate_cols,
        freq="h",
        fill_missing_dates=False,
        fillna_value=0.0,
    )
    for name, g in df.groupby("gw_name", sort=False)
}

ride = "INDIANA_JONES_ADVENTURE" if "INDIANA_JONES_ADVENTURE" in series_by_ride else next(iter(series_by_ride))
ts = series_by_ride[ride]
past_covs = past_covs_by_ride[ride]
ts_scaler = Scaler()
cov_scaler = Scaler()
# Setup up training window
CUTOFF = pd.Timestamp("2025-01-01 12:00:00")
back = pd.Timedelta(hours=PAST_LAGS)

ts_train = ts.slice(ts.start_time(), CUTOFF)
ts_train_s = ts_scaler.fit_transform(ts_train)
covars_train = past_covs.slice(ts_train.start_time() - back, ts_train.end_time())
covars_train_s = cov_scaler.fit_transform(covars_train)

model = LightGBMModel(
    lags=24,                # use last 24 hours as features
    output_chunk_length=24, # predict up to 24 steps per call
    lags_past_covariates=PAST_LAGS,
    random_state=42
)

model.fit(ts_train_s, past_covariates=covars_train_s)

# --- Predictions ---
# -- Setup prediction window --
pred_day = pd.Timestamp("2025-01-02 00:00:00")  # day to forecast
pred_hour = 13
hour_delta = pd.Timedelta(hours = pred_hour)# first hour to forecast (24-hour clock)
PRED_START = pred_day + hour_delta         # first hour to forecast
DAY_END = pred_day + pd.Timedelta(hours=23)  # last hour to forecast
pred_back = pd.Timedelta(hours=PAST_LAGS + 1)

ts_pred   = ts.slice(ts.start_time(),PRED_START - pd.Timedelta(hours=1)) # full history up to pred start-1
covars_pred   = past_covs.slice(PRED_START - back, PRED_START - pd.Timedelta(hours=1))
covars_pred_s   = cov_scaler.transform(covars_pred)
n_steps = 23 - pred_hour

# Actual prediction
pred_s = model.predict(n=n_steps, series=ts_pred, past_covariates=covars_pred_s)
pred   = ts_scaler.inverse_transform(pred_s)


# Validation window
ts_val_pred   = ts.slice(PRED_START, DAY_END)
show_forecast("Forecast vs Actuals ", ts_val_pred, pred)
# Compute MAE on validation window
val_mae    = mae(ts_val_pred, pred)
# Show target + covariates for the TRAIN window (last 48 rows for readability)
show_window("TRAIN window (target + covariates)",
            ts_train,
            covars_train,
            limit=6)

show_window("Past covariates for first forecast step (should end at pred_hour-1)",
            covars_pred,
            None,
            limit=PAST_LAGS + 2)
print(f"Validation MAE {pred.start_time()}: {val_mae:.3f}")

