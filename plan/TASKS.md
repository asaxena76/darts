# Build the test harness and a simple baseline model
Build the test harness and using a simple baseline model using the darts library.
I have checked out the darts

## Implementation notes (initial milestone)
- Harness lives in `disney_harness/` with:
  - `model.py` (interface + naive baseline)
  - `metrics.py` (MAE/MAPE)
  - `harness.py` (loader, loop over days/attractions, per-hour inference windows)
- CLI: `scripts/run_harness.py --csv <hourly_csv> --test-days 30`
- Outputs: overall summary (stdout) + CSVs for by-attraction, by-infer-hour, and raw rows.
