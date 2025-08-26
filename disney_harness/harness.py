from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Dict, Any
import logging
from .model import DisneyModel
from .metrics import mae, mape

# Basic park hours (you can parameterize later)
PARK_OPEN_HOUR = 8   # 08:00
PARK_CLOSE_HOUR = 24 # 24:00 == midnight end-of-day
log = logging.getLogger(__name__)
DEBUG_PRINTED= False

@dataclass
class HarnessConfig:
    csv_path: str
    target_col: str = "mrtd"
    attraction_col: str = "gw_name"
    ts_col: str = "pred_hour_pdt"  # HOURLY timestamp in local time
    test_days: int = 30            # last N days for evaluation
    daily_train_cutoff_hour: int = 3  # train up to 03:00 of the same day
    # --- debug peek
    debug_infer_hour: int = 12
    debug_sample: bool = False
    debug_rows: int = 20

def _load_hourly_df(cfg: HarnessConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path, parse_dates=[cfg.ts_col])
    # keep only necessary columns for baseline
    keep = [cfg.attraction_col, cfg.ts_col, cfg.target_col]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # ensure hourly index
    df = df.sort_values([cfg.attraction_col, cfg.ts_col])
    # logging summary
    if not df.empty:
        n_gw = df[cfg.attraction_col].nunique()
        tmin, tmax = df[cfg.ts_col].min(), df[cfg.ts_col].max()
        log.info("Loaded %s rows | attractions=%s | time range: %s → %s", f"{len(df):,}", n_gw, tmin, tmax)
    else:
        log.warning("Loaded 0 rows from %s", cfg.csv_path)
    return df

def _hourly_truth_for_day(day_hourly: pd.DataFrame, cfg: HarnessConfig) -> pd.Series:
    """
    Use the hourly target as-is, restricted to park hours [open, close).
    """
    if day_hourly.empty:
        return pd.Series(dtype=float)

    day = day_hourly[cfg.ts_col].dt.normalize().iloc[0]
    start_hour = day + pd.Timedelta(hours=PARK_OPEN_HOUR)
    end_hour   = day + pd.Timedelta(hours=PARK_CLOSE_HOUR) - pd.Timedelta(hours=1)

    hourly = (
        day_hourly.set_index(cfg.ts_col)[cfg.target_col]
        .asfreq("h")
        .sort_index()
    )
    hourly = hourly.loc[(hourly.index >= start_hour) & (hourly.index <= end_hour)]
    return hourly

def evaluate_model(cfg: HarnessConfig, model: DisneyModel) -> Dict[str, Any]:
    df = _load_hourly_df(cfg)
    results = []
    raw_det__rows = []
    debug_printed = False

    # pick last N distinct dates as test set
    all_days = df[cfg.ts_col].dt.normalize().drop_duplicates().sort_values()
    test_days = all_days.iloc[-cfg.test_days:] if len(all_days) >= cfg.test_days else all_days
    hours_per_day = PARK_CLOSE_HOUR - PARK_OPEN_HOUR
    # Precompute total inference windows for progress
    total_jobs = 0
    for day in test_days:
        day_mask = df[cfg.ts_col].dt.normalize() == day
        total_jobs += df[day_mask][cfg.attraction_col].nunique() * hours_per_day
    done_jobs = 0
    log.info("Evaluation window: days=%d | hours/day=%d | total inference windows=%d", len(test_days), hours_per_day, total_jobs)

    for day in test_days:
        day_mask = df[cfg.ts_col].dt.normalize() == day
        day_df = df[day_mask]
        train_cutoff = day + pd.Timedelta(hours=cfg.daily_train_cutoff_hour)
        park_open = day + pd.Timedelta(hours=PARK_OPEN_HOUR)
        park_close = day + pd.Timedelta(hours=PARK_CLOSE_HOUR)
        log.info("▶︎ Day %s | attractions=%d | train_cutoff=%s | open=%s | close=%s",
                 day.date(), day_df[cfg.attraction_col].nunique(), train_cutoff, park_open, park_close)

        # For each attraction, train once (3am cutoff), then loop by infer_hour
        for gw, gdf in day_df.groupby(cfg.attraction_col):
            # History up to cutoff is allowed for "fit"
            hist_upto_cutoff = df[(df[cfg.attraction_col] == gw) & (df[cfg.ts_col] < train_cutoff)]
            log.info("  • Training model for '%s' on %s rows (<= %s)", gw, len(hist_upto_cutoff), train_cutoff)
            model.fit(hist_upto_cutoff.set_index(cfg.ts_col))

            # Build hourly TRUTH for the full day window
            truth_hourly = _hourly_truth_for_day(day_df[day_df[cfg.attraction_col] == gw], cfg)
            # Fill na to 0
            truth_hourly = truth_hourly.fillna(0.0)

            # infer from each hour (open..23)
            for infer_hour in range(PARK_OPEN_HOUR, PARK_CLOSE_HOUR):
                infer_ts = day + pd.Timedelta(hours=infer_hour)
                # history available up to infer_ts
                hist_to_now = df[(df[cfg.attraction_col] == gw) & (df[cfg.ts_col] < infer_ts)].set_index(cfg.ts_col)

                # predict minute-level from (infer_ts+1min) to close
                pred = model.predict_until_close(hist_to_now, park_close)

                # restrict truth to same prediction window
                truth_window = truth_hourly.loc[pred.index.min(): pred.index.max()] if not pred.empty else pd.Series(dtype=float)

                print_debug(cfg, day, df, gw, hist_to_now, infer_hour, infer_ts, pred, test_days,
                            train_cutoff, truth_hourly, truth_window)

                aligned_full = pd.concat([truth_window.rename("truth"), pred.rename("pred")],axis=1)

                for ts, row in aligned_full.iterrows():
                    truth_val = row["truth"]
                    pred_val = round(row["pred"])  # nearest integer
                    abs_err = abs(truth_val - pred_val)
                    mape_val = abs_err / (abs(truth_val)) if truth_val != 0 else 0.0
                    raw_det__rows.append({
                        "date": day.date(),
                        "gw_name": gw,
                        "infer_hour": infer_hour,
                        "pred_hour": ts.hour,   # 8–23
                        "truth": truth_val,
                        "pred": pred_val,
                        "mae": abs_err,
                        "mape": mape_val,
                    })

                res = {
                    "date": day.date(),
                    "gw_name": gw,
                    "infer_hour": infer_hour,
                    "mae": mae(truth_window, pred),
                    "mape": mape(truth_window, pred),
                }
                results.append(res)
            done_jobs += hours_per_day
            pct = (done_jobs / max(total_jobs, 1)) * 100.0
            log.info("  ✓ Completed '%s' for %s (%d/%d windows, %.1f%%)",
                     gw, day.date(), done_jobs, total_jobs, pct)

    out = pd.DataFrame(results)
    if out.empty:
        log.warning("No results produced — check input coverage and filters.")
        return {"overall": {}, "by_attraction": pd.DataFrame(), "by_infer_hour": pd.DataFrame(), "raw": out}

    raw_det_out = pd.DataFrame(raw_det__rows)
    summary_overall = {
        "mae": float(out["mae"].mean()),
        "mape": float(out["mape"].mean()),
        "n_rows": int(len(out)),
    }
    log.info("Computed metrics: rows=%d | overall MAE=%.4f | MAPE=%.4f",
             summary_overall["n_rows"], summary_overall["mae"], summary_overall["mape"])
    by_attraction = out.groupby("gw_name")[["mae", "mape"]].mean().reset_index()
    by_infer_hour = out.groupby("infer_hour")[["mae", "mape"]].mean().reset_index()

    return {"overall": summary_overall, "by_attraction": by_attraction, "by_infer_hour": by_infer_hour, "raw": out, "raw_detailed": raw_det_out}

def print_debug(cfg, day, df, gw, hist_to_now, infer_hour, infer_ts, pred, test_days, train_cutoff,
                truth_hourly, truth_window):
    # optional debug: for first test day & chosen hour, print aligned rows + tails of train/history
    global DEBUG_PRINTED
    if (
            cfg.debug_sample
            and (not DEBUG_PRINTED)
            and (not pred.empty)
            and (not truth_window.empty)
            and infer_hour == cfg.debug_infer_hour
            # "first day" in the selected test set
            and day == test_days.iloc[0]
    ):
        # print predictions
        print(
            f"\n[DEBUG] Predictions for '{gw}' on {day.date()} at infer_hour={infer_hour} (from {pred.index.min()} to {pred.index.max()}):")
        print(pred.head(cfg.debug_rows).rename("pred").reset_index().to_string(index=False))
        # print truth
        print(f"\n[DEBUG] Truth for '{gw}' on {day.date()} (park hours {PARK_OPEN_HOUR}:00 to {PARK_CLOSE_HOUR}:00):")
        print(truth_hourly.head(cfg.debug_rows).rename("truth").reset_index().to_string(index=False))

        # Build aligned slice for manual verification
        aligned = pd.concat(
            [truth_window.rename("truth"), pred.rename("pred")], axis=1
        ).dropna()
        if not aligned.empty:
            aligned = aligned.assign(abs_err=(aligned["truth"] - aligned["pred"]).abs())
            manual_mae = float(aligned["abs_err"].mean())
            head_n = aligned.head(cfg.debug_rows)
            print("\n========== DEBUG SAMPLE (hourly truth vs hourly pred) ==========")
            print(f"Day: {day.date()} | Attraction: {gw} | infer_hour: {infer_hour}")
            print(head_n.reset_index().rename(columns={"index": "time"}).to_string(index=False))
            print(f"\nmanual MAE over these {len(head_n)} rows: {manual_mae:.4f}")
            print("==================================================")

            # Print last few rows of TRAINING data used (<= train_cutoff - strictly earlier)
            train_tail = (
                df[(df[cfg.attraction_col] == gw) & (df[cfg.ts_col] < train_cutoff)]
                .sort_values(cfg.ts_col)
                .tail(cfg.debug_rows)
            )[[cfg.ts_col, cfg.target_col]]
            print("\n---------- TRAIN TAIL (<= train_cutoff) ----------")
            print(f"train_cutoff: {train_cutoff}")
            if not train_tail.empty:
                print(train_tail.to_string(index=False))
                print(f"last_train_ts: {train_tail[cfg.ts_col].max()}")
            else:
                print("(empty)")

            # Print last few rows of HISTORY available to inference (<= infer_ts)
            hist_tail = (
                hist_to_now.reset_index()
                .sort_values(cfg.ts_col)
                .tail(cfg.debug_rows)
            )[[cfg.ts_col, cfg.target_col]]
            print("\n---------- HISTORY TAIL (<= infer_ts) ------------")
            print(f"infer_ts: {infer_ts}")
            if not hist_tail.empty:
                print(hist_tail.to_string(index=False))
                print(f"last_hist_ts: {hist_tail[cfg.ts_col].max()}")
            else:
                print("(empty)")
            print("==================================================\n")
            DEBUG_PRINTED = True
