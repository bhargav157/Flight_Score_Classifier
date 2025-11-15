# ✈️ Flight_Score_Classifier

This project builds a daily, per-flight "difficulty score" to help airline operations prioritize focus and resources. It analyzes multiple data sources to identify flights that are most likely to face challenges, such as tight ground times, high passenger loads, and special handling needs.

## 🎯 Objective

To build a daily, per-flight difficulty score to prioritize operational focus.

## 📊 Data Used

The score is derived from five main data sources:
* Flight-level data
* PNR (passenger) data
* PNR remarks (SSRs)
* Bag-level data
* Airports reference data

## ⚙️ Approach

The project follows a three-step process:

1.  **EDA (Exploratory Data Analysis):** Compute delays, ground time surplus, bag aggregates, load factors, and SSR counts.
2.  **Feature Engineering:** Create features for `load_factor`, `transfer_share`, `ground_pressure` (negative ground time surplus), `n_ssr`, bag counts, and passenger composition.
3.  **Scoring:**
    * Features are standardized within each day using robust scaling (median/MAD) to reduce outlier effects.
    * A weighted linear combination is used (e.g., ground_pressure 30%, load 20%, transfer_share 15%, n_ssr 15%).
    * The score is normalized from 0 to 1 each day and classified into three categories: **Difficult** (Top 20%), **Medium**, and **Easy** (Bottom 30%).

## 📈 Key Results & Findings

* **Average Departure Delay:** 21.18 minutes.
* **Delayed Flights:** 49.6% of flights departed later than scheduled.
* **Ground Time Pressure:** 780 flights had a scheduled ground time less than or equal to the minimum turn time + 5 minutes.
* **Load vs. Delay:** A weak negative correlation (-0.150) was found between load factor and departure delay.
* **Top Difficult Destinations:** The destinations with the most "Difficult" flights include IAH, SFO, DEN, LAX, and SEA.
* **Common Drivers:** Difficulty is often driven by short ground time, a high share of transfer passengers, and high SSR counts.

## 🚀 Getting Started

Follow these instructions to reproduce the analysis and results.

### 1. Requirements

This project requires the following Python dependencies:
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* python-pptx

### 2. Installation

1.  (Optional) Create and activate a virtual environment:
    ```powershell
    python3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
2.  Install the required dependencies from `requirements.txt`:
    ```powershell
    python -m pip install -r requirements.txt
    ```

### 3. How to Run

1.  **Run the Main Script:**
    Execute the main script (`hello.py`) to run the full pipeline (EDA, scoring, and output generation).
    ```powershell
    python .\hello.py
    ```
2.  **Run the Automation Script:**
    Alternatively, you can use the provided PowerShell script (`run_all.ps1`) to set up the environment, install dependencies, and run the pipeline in one step.
    ```powershell
    .\run_all.ps1
    ```
3.  **Inspect Data (Optional):**
    To get a quick summary of the input CSVs before running, use the helper script:
    ```powershell
    python .\scripts\column_summary.py
    ```

## 📂 Project Structure & Outputs

### Key Files
* `hello.py`: Main Python script for EDA, scoring, and output generation.
* `requirements.txt`: Python dependencies.
* `DELIVERABLES.md`: Summary of outputs and instructions.
* `PRESENTATION.md`: A markdown-based presentation of the project.
* `scripts/column_summary.py`: Helper script for inspecting data.

### Generated Outputs
After running `hello.py`, the following files will be generated:

* **Data:**
    * `./data/test_chaitanyakumar.csv`: The final output file with scored flights, `difficulty_score`, rank, and class.
    * `flight_difficulty_full_debug.csv`: A detailed debugging CSV with all intermediate features and z-scores.
* **Plots (in `plots/` directory):**
    * `difficulty_score_hist.png`
    * `top_destinations_difficult_count.png`
    * `load_vs_delay_scatter.png`
    * `daily_mean_difficulty.png`

## 🔮 Recommendations & Next Steps

### Operational Recommendations
1.  **Prioritize Resources:** Increase buffers or prioritize staffing/bridging for flights with tight ground times (<= min turn + 5 min) or those ranked as "Difficult".
2.  **Improve Transfer Handling:** Review transfer baggage processes at frequently difficult destinations.
3.  **Manage SSRs:** Monitor SSR-heavy flights to pre-assign assistance staff.
4.  **Real-time Decisions:** Use the daily score to guide real-time dispatch, pre-clear transfer bags, and adjust pushback targets.

### Next Steps
* **Validate:** Validate the score with more historical data and compare it against actual delay incidents.
* **Enhance Scoring:** Add a model-based scoring method (e.g., XGBoost) to predict minute-level delay risk and compare its performance to the current rule-based score.
