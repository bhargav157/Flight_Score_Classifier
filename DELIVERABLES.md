Deliverables and artifacts

This folder contains the code, outputs, and a concise presentation for the Flight Difficulty case study.

Files added/updated
- `hello.py` : main Python script that computes EDA, per-day difficulty score, rankings and outputs CSV + plots (already present and updated).
- `requirements.txt` : Python dependencies (pandas, numpy, scikit-learn, matplotlib).
- `plots/` : directory with generated plots (png files).
- `data/test_chaitanyakumar.csv` : sample output produced by running `hello.py` (written into `data/`).
- `DELIVERABLES.md` : this file summarizing outputs and instructions.
- `PRESENTATION.md` : 7 slide presentation in markdown covering approach, key results and recommendations.
- `scripts/column_summary.py` : small helper to print column summaries and sample rows for each input CSV.

Quick summary of results (from a run on the provided data):

- Average departure delay (minutes): 21.18
- % flights departed later than scheduled: 49.6%
- Flights with scheduled ground time <= minimum_turn + 5min: 780
- Average transfer/checked bag ratio (per flight, avg): NaN (many flights had zero checked bags in groups or missing bag types)
- Correlation between load_factor and departure_delay: -0.150 (weak negative correlation in this dataset)
- SSR correlation with delay residuals (controlling for load): NaN (likely due to missing/empty SSR values)

Important output files (after running `hello.py`):

- `./data/test_chaitanyakumar.csv` - scored flights with difficulty_score, rank and class.
- `flight_difficulty_full_debug.csv` - full debugging CSV with all intermediate features and z-scores.
- Plots: `plots/difficulty_score_hist.png`, `plots/top_destinations_difficult_count.png`, `plots/load_vs_delay_scatter.png`, `plots/daily_mean_difficulty.png`.

How to reproduce

1. (Optional) create a virtual environment and activate it (PowerShell):
```powershell
python3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
python -m pip install -r requirements.txt
```
3. Run the main script:
```powershell
python .\hello.py
```

4. Inspect outputs in `./data/` and `./plots/`.

If you want a quick CSV inspection before running, use:
```powershell
python .\scripts\column_summary.py
```

Or run the provided automation script (PowerShell) which sets up a venv, installs deps and runs the pipeline:
```powershell
.\run_all.ps1
```

Notes and caveats
- The scripts add sensible defaults for missing columns; please verify columns and datatypes in your real data.
- Some metrics came out NaN due to missing data in bag / SSR columns; see `flight_difficulty_full_debug.csv` for detailed diagnostics.
