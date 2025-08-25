#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import shutil
import logging

# ensure repo root is on sys.path so 'disney_harness' can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from disney_harness.harness import HarnessConfig, evaluate_model
from disney_harness.model import NaiveLastHourCarryForward

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to hourly features CSV")
    ap.add_argument("--test-days", type=int, default=30)
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Base directory to write experiment outputs (default: results/)",
    )
    cfg = HarnessConfig(
        csv_path=args.csv,
        test_days=args.test_days,
        debug_sample=args.debug_sample,
        debug_rows=args.debug_rows,
        debug_infer_hour=args.debug_infer_hour,
    )
    ap.add_argument(
        "--debug-infer-hour",
        type=int,
        default=12,
        help="Hour-of-day (0-23) to trigger the debug sample on the FIRST test day (default: 12).",
    )
    ap.add_argument(
        "--debug-sample",
        action="store_true",
        help="Print a few aligned rows of truth vs pred for the first day to verify MAE.",
    )
    ap.add_argument(
        "--debug-rows",
        type=int,
        default=20,
        help="How many rows to print in the debug sample (default: 20).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    args = ap.parse_args()

    # Configure logging early
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("run_harness")
    log.info("Starting harness")
    log.info("Args: %s", {k: v for k, v in vars(args).items() if k != "csv"} )
    log.info("CSV: %s", args.csv)

    cfg = HarnessConfig(
        csv_path=args.csv,
        test_days=args.test_days,
        debug_sample=args.debug_sample,
        debug_rows=args.debug_rows,
    )
    model = NaiveLastHourCarryForward(target_col=cfg.target_col)

    # --- prepare run output dir: results/{timestamp}__{args_slug}/
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_name = Path(args.csv).stem
    args_slug = f"csv-{csv_name}__testdays-{args.test_days}"
    run_dir = Path(args.results_dir) / f"{ts}__{args_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Run directory: %s", run_dir)

    # Save args & config snapshot for reproducibility
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        "target_col": cfg.target_col,
        "attraction_col": cfg.attraction_col,
        "ts_col": cfg.ts_col,
        "daily_train_cutoff_hour": cfg.daily_train_cutoff_hour,
        "park_hours": {"open": 8, "close": 24},
        "debug_sample": cfg.debug_sample,
        "debug_infer_hour": cfg.debug_infer_hour,
        "debug_rows": cfg.debug_rows,
    }, indent=2))
    (run_dir / "INPUT_CSV_PATH.txt").write_text(str(Path(args.csv).resolve()))

    results = evaluate_model(cfg, model)

    print(json.dumps({"overall": results["overall"]}, indent=2))
    print(f"[run_dir] {run_dir}")
    log.info("Overall summary: %s", results["overall"])

    # write detailed outputs into run_dir
    results["by_attraction"].to_csv(run_dir / "by_attraction.csv", index=False)
    results["by_infer_hour"].to_csv(run_dir / "by_infer_hour.csv", index=False)
    results["raw"].to_csv(run_dir / "raw_results.csv", index=False)
    (run_dir / "overall.json").write_text(json.dumps(results["overall"], indent=2))
    log.info("Wrote: by_attraction.csv, by_infer_hour.csv, raw_results.csv, overall.json")

if __name__ == "__main__":
    main()
