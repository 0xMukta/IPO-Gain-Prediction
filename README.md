# IPO Listing-Day Gain Prediction (India – Mainboard & SME)

**Course:** Introduction to Machine Learning (PCCAIML502)  
**Objective:** Predict Day-1 listing-day percentage gains (`target_open_gain_pct`, `target_typical_gain_pct`, `target_close_gain_pct`) for Indian equity IPOs using pre-listing fundamentals, subscription demand, anchor quality, grey market sentiment, and broad market macro regimes.

---

## Problem Statement

Indian IPOs exhibit wide Day-1 listing-day dispersion across Mainboard and SME segments. Using only information available up to the IPO subscription close date (`close_date`), this project builds a leakage-free machine learning pipeline to estimate expected listing gains and compare distinct model families under strict chronological validation.

---

## Dataset Summary

- **Sample Size:** 1,060 listed equity IPOs (Nov 2023 – Oct 15, 2026; capped at October 15, 2026 for regulatory compliance) across BSE and NSE
- **Feature Space:** 66 columns (identifiers/dates for chronological splitting, engineered financial ratios, `log1p`-transformed subscription & issue size features, sentiment metrics, macro indicators, and 3 Day-1 gain targets)
- **Sources:** Personally compiled and curated by me from exchange disclosures (BSE, NSE) for issue, subscription, listing OHLC, and index data (Nifty 50, Nifty Smallcap 250, India VIX), offer documents (RHP) for financials, peer P/E, and anchor allocations, and third-party IPO aggregators for GMP and broker consensus
- **Details:** See [Dataset.md](Dataset.md) and [Preprocessing.md](Preprocessing.md) for the complete schema and cleaning rules.

---

## Key EDA Findings (`EDA.ipynb`)

- **Overall Typical Gain (`target_typical_gain_pct`):** Median **`+9.71%`**, Mean **`+23.02%`**, Positive Listing Rate **`71.23%`** (`324` Mainboard / `736` SME).
- **Subscription Demand Gradient:**
  - `< 2x` total subscription (`173` IPOs): median **`-3.00%`** (`37.6%` positive)
  - `50x–150x` (`230` IPOs): median **`+20.53%`** (`89.6%` positive)
  - `> 150x` (`247` IPOs): median **`+71.03%`** (`96.0%` positive)
- **Regime Shift:** 2023–2024 averaged higher listing gains (`+20.24%` / `+31.84%` median) compared to 2025–2026 (`+3.70%` / `+4.15%` median), motivating strict chronological evaluation rather than random splits.

---

## Modeling Methodology (`train.py`)

To evaluate diverse inductive biases on ~1,060 tabular rows, three distinct algorithm families (plus a Naive GMP heuristic and a CV-tuned weighted blend) are trained and compared:
1. **Ridge Regression (`RidgeCV`)** – Linear baseline inside a scikit-learn `Pipeline` (`SimpleImputer(strategy="median")` + `StandardScaler` + `OneHotEncoder(min_frequency=10)`).
2. **Random Forest Regressor** – Bagging ensemble benchmark for non-linear interactions (`max_depth=10`, `min_samples_leaf=5`, `n_estimators=400`).
3. **CatBoost Regressor (`MultiQuantile` & `RMSE`)** – Primary gradient-boosted model with native categorical handling (`exchange`, `issue_type`, `sector`, `lead_manager_grouped`), Bernoulli `0.8` subsampling, and native `NaN` routing, outputting **P10 / P50 / P90** prediction intervals (`np.sort` post-processed and SME circuit-clipped).
4. **Weighted Ensemble** – Out-of-fold CV-optimized blend across `CatBoost MultiQuantile (P50)`, `CatBoost RMSE`, and `Random Forest`.

All models are evaluated chronologically using `TimeSeriesSplit(n_splits=5)` on the first `980` IPOs sorted by `listing_date`, with an untouched chronological holdout set of the `80` most recent IPOs (up to `2026-10-15`).

---

## Experimental Results (`target_typical_gain_pct`)

Models were evaluated across both **With GMP** and **Without GMP** feature sets on the untouched chronological holdout (`n = 80`), with SME predictions clipped to `[-24.0%, +99.5%]`:

### Chronological Holdout Results (`n = 80`)

| Mode | Model | MAE (%) | RMSE (%) | $R^2$ | Dir. Acc (%) | Spearman $\rho$ | Mainboard MAE | SME MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **With GMP** | Naive GMP Baseline (`n=79`) | 12.08 | 17.01 | 0.689 | 69.62 | 0.697 | 10.69 | 12.97 |
| **With GMP** | RidgeCV | 12.67 | 19.05 | 0.606 | 77.50 | 0.707 | 14.34 | 11.55 |
| **With GMP** | Random Forest | 11.85 | **16.89** | **0.690** | **77.50** | 0.718 | **11.43** | 12.12 |
| **With GMP** | CatBoost RMSE | 12.65 | 18.40 | 0.632 | 71.25 | 0.684 | 12.85 | 12.52 |
| **With GMP** | **CatBoost MultiQuantile (P50)** | **11.48** | 17.04 | 0.685 | 75.00 | **0.730** | 11.83 | **11.24** |
| **With GMP** | Weighted Blend (`0.20*P50 + 0.30*RMSE + 0.50*RF`) | 11.90 | 17.14 | 0.681 | 73.75 | 0.718 | 11.84 | 11.93 |
| **Without GMP** | RidgeCV | 17.11 | 21.87 | 0.481 | 71.25 | 0.612 | 17.68 | 16.73 |
| **Without GMP** | Random Forest | 13.24 | 18.89 | 0.613 | **73.75** | 0.594 | 12.12 | 13.99 |
| **Without GMP** | CatBoost RMSE | 13.48 | 18.55 | 0.626 | 60.00 | 0.549 | 12.41 | 14.19 |
| **Without GMP** | **CatBoost MultiQuantile (P50)** | **12.45** | **18.29** | **0.637** | 70.00 | **0.622** | **11.63** | **13.00** |

- **P10–P90 Interval Coverage:** `73.75%` (With GMP) and `81.25%` (Without GMP).
- **GMP Ablation Finding:** CatBoost P50 beats the Naive GMP heuristic (`11.48%` vs. `12.08%` MAE, `75.0%` vs. `69.6%` directional accuracy). Removing GMP increases holdout MAE by only `+0.97%` overall (`11.48%` $\to$ `12.45%`, while Mainboard actually improves from `11.83%` $\to$ `11.63%`), so both modes are retained for reproducibility and live toggling in the UI.

---

## Repository Structure

```text
IPO_Gain_Prediction/
├── Preprocessing.ipynb          # Field-by-field validation, cleaning & feature engineering
├── EDA.ipynb                    # Exploratory data analysis & summary stats export
├── feature_pipeline.py          # Shared raw-to-clean preprocessing module for inference
├── train.py                     # Chronological CV, GMP ablation, blending & SHAP training
├── app.py                       # Streamlit UI (raw entry, CSV upload, clean export & SHAP)
├── sample_ipo_input.csv         # Sample raw CSV template for batch upload
├── ipo_listing_gain_clean.csv   # Baseline cleaned dataset (1,060 rows x 66 columns)
├── artifacts/                   # Trained models, metrics JSON, EDA JSON & plots
├── Dataset.md                   # Cleaned dataset dictionary & missingness guide
├── Preprocessing.md             # Preprocessing & feature engineering documentation
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview & results
```

---

## Streamlit UI Features (`app.py`)

- **Sidebar Controls & Sample Template:**
  - Toggle **With GMP** vs. **Without GMP** models live.
  - Toggle between **CatBoost MultiQuantile (P50)** (default, lowest holdout MAE) and the **CV-Optimized Weighted Ensemble**.
  - Download `sample_ipo_input.csv` directly from the sidebar to see the exact raw input schema.
- **Tab 1 – Predict & Preprocess Raw Data:**
  - Enter raw IPO details (dropdowns for `is_sme`, `exchange`, `issue_type`, `priced_at_upper_band`, `industry`/`sector`, and `lead_manager`; leave any optional numeric field blank or select `"Leave Empty (NaN)"` to pass `NaN` natively).
  - Upload a batch CSV (`sample_ipo_input.csv` format) – any missing columns are automatically supplemented from the UI form (with an option to also fill `NaN` cells from the UI).
  - Inspect and download the transformed **Cleaned Model-Ready Features CSV** (applying all [Preprocessing.md](Preprocessing.md) rules).
  - View **Expected Gain**, **Worst-Case Downside (P10)**, **Expected Median (P50)**, **Upside Potential (P90)**, and the **Top 10 local SHAP feature drivers** for each IPO.
- **Tab 2 – EDA & Historical Gain Stats:** Historical segment, subscription-bucket, sector, and yearly listing gain summaries with `300` DPI charts.
- **Tab 3 – Model Benchmarks & GMP Ablation:** Side-by-side With-GMP vs. Without-GMP holdout tables, Naive GMP comparison, and global SHAP summary charts.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **(Optional) Regenerate cleaned dataset from raw data:**  
   Place `raw_ipo_dataset.csv` in the project root and execute all cells in `Preprocessing.ipynb` to produce `ipo_listing_gain_clean.csv`.
3. **Run Exploratory Data Analysis:**
   ```bash
   python -m nbconvert --to notebook --execute --inplace EDA.ipynb
   ```
   Generates `artifacts/eda_stats.json` and `300` DPI EDA plots in `artifacts/plots/`.
4. **Train & evaluate all models (`train.py`):**
   ```bash
   python train.py --target target_typical_gain_pct --holdout 80
   ```
   **CLI Flags:**
   - `--target`: Day-1 target column to predict – `target_typical_gain_pct` *(default)*, `target_open_gain_pct`, or `target_close_gain_pct`.
   - `--holdout`: Number of most recent chronological IPOs reserved for the untouched holdout test set (default `80`; recommended `50–100`).

   Saves trained models (`catboost_mq_*.cbm`, `catboost_rmse_*.cbm`, `rf_*.joblib`, `ridge_*.joblib`), `artifacts/model_config.json`, `artifacts/model_metrics.json`, and `300` DPI comparison/SHAP plots in `artifacts/plots/`.
5. **Launch the Streamlit Demo App:**
   ```bash
   python -m streamlit run app.py
   ```

---

## Disclaimer

This project was built solely for academic coursework (PCCAIML502 – Introduction to Machine Learning) and educational research using historical IPO data. I am not a SEBI-registered Research Analyst or Investment Adviser. The model outputs are statistical estimates for demonstration purposes only and do not constitute financial advice, investment recommendations, or guaranteed returns.

---

## License

This project is licensed under the **MIT License** – free to use, modify, and distribute for any purpose with attribution.
