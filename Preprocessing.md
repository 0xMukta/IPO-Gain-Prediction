# Data Preprocessing & Feature Engineering (`Preprocessing.md`)

This document summarizes the validation checks, cleaning rules, and feature engineering transformations implemented in `Preprocessing.ipynb` to produce `ipo_listing_gain_clean.csv`.

## 1. Scope & Target Selection

- **Day-1 Gain Focus:** This baseline targets Day-1 listing performance (`target_open_gain_pct`, `target_typical_gain_pct`, `target_close_gain_pct`).
- **Universal Typical Price Target:** `target_typical_gain_pct` uses the 5-point weighted price `(list_open + list_high + list_low + 2 * list_close) / 5`, which weights the pre-open call auction and closing auction consistently across all 1,060 IPOs without relying on short-retention intraday bars.

## 2. Data Validation & Cleaning Findings

Walking through the 1,064 compiled records field by field revealed the following issues, which were resolved systematically:

1. **Duplicate & Identifier Check:** `0` duplicate company names and `0` duplicate non-null ISINs.
2. **Unlisted / Withdrawn Issues (4 rows dropped):**
   - *Trafiksol ITS Technologies Ltd.* (BSE SME, Sep 2024)
   - *Phytochem Remedies (India) Ltd.* (BSE SME, Dec 2025)
   - *Sri Priyanka Geo Commex Ltd.* (NSE, Jun 2026)
   - *Twinkle Papers Ltd.* (BSE SME, Jun 2026)
   All 4 lacked a valid `listing_date` and listing targets, leaving **1,060 listed equity IPOs**.
3. **Chronological Consistency:** `open_date`, `close_date`, and `listing_date` parsed with `0` invalid dates and `0` ordering violations (`open_date <= close_date <= listing_date`). Rows are sorted chronologically by `listing_date` (covering Nov 2023 – Oct 15, 2026 for regulatory compliance).
4. **Structural Zeros vs Missingness:**
   - Pure OFS issues (`55` rows) had `NaN` in `fresh_issue_cr` – filled with `0.0`.
   - Pure fresh issues (`647` rows) had `NaN` in `ofs_size_cr` – filled with `0.0`.
   - Non-anchor issues have `0.0` / `0` across anchor fields (`anchor_portion_cr`, `tier1_anchor_share_pct`, `tier1_unique_houses`).
5. **Financial Anomalies:**
   - **Negative Borrowing:** *Trust Fintech Ltd.* had `borrowing_fy1 = -0.04` Cr (rounding artifact) – clipped to `0.0`.
   - **Negative Pre-IPO Net Worth (4 rows):** *Premier Energies Ltd.*, *Indiqube Spaces Ltd.*, *Pine Labs Ltd.*, and *Purple Style Labs Ltd.* had `net_worth_fy1 <= 0`. `debt_to_equity` is set to `NaN` for these rows rather than producing misleading negative leverage.
   - **Missing FY1 Revenue (1 row):** *Harshdeep Hortico Ltd.* had `NaN` `rev_fy1`, so margin and `price_to_sales` ratios evaluate cleanly to `NaN`.
   - **Extreme Ratios:** Outliers in `roe_pct`, `roce_pct` (`[-100, 150]`), `price_to_book` (`[0, 50]`), and `pe_post_ipo` (upper `150.0`) are domain-clipped to protect linear models.
6. **Macro & Demand Integrity:** All 6 macro indicators (`nifty50_*`, `smallcap250_*`, `india_vix_*`) are 100% populated as of `close_date`. `sub_total_x` has `0` nulls. Sparse `sub_bnii_x` and `sub_snii_x` are dropped in favor of universal `sub_nii_total_x`.

## 3. Feature Engineering & Transformations

### Scale-Invariant Ratios
- **Issue Composition:** `ofs_pct = ofs_size_cr / total_issue_size_cr` (clipped to `[0, 1]`).
- **GMP Standardization:** Converted raw ₹ GMP into percentages relative to `issue_price`:
  - `gmp_close_day_pct = (gmp_close_day_inr / issue_price) * 100`
  - `gmp_momentum_pct = gmp_close_day_pct - gmp_avg_pct` (collinear `gmp_avg_pct` is not retained in the cleaned CSV).
- **Growth & Margins:** Derived `rev_yoy_pct`, `pat_yoy_pct`, `rev_2yr_cagr`, `ebitda_yoy_pct`, `pat_margin_pct`, and `ebitda_margin_pct` with safe domain clipping.
- **Valuation & Leverage:** Derived `debt_to_equity`, `price_to_sales`, and fallback `price_to_book` using post-issue capitalization (`net_worth_fy1 + fresh_issue_cr`).
- **Loss-Making & Relative P/E (Strict Order):**
  1. Compute `is_loss_making` from `pat_fy1 <= 0` (or `pe_post_ipo < 0` if `pat_fy1` is missing) **before** modifying `pe_post_ipo`.
  2. Convert `pe_post_ipo <= 0` to `NaN` and upper-clip at `150.0` so loss-makers do not appear artificially cheap to linear models.
  3. Compute `relative_pe = pe_post_ipo / peer_avg_pe` (upper-clipped at `10.0`).
- **Broker Consensus:** `broker_total_reviews` and `broker_sub_ratio = broker_sub / broker_total_reviews` (`NaN` when total reviews is `0`).
- **Binary Indicator Flags:** `has_listed_peers`, `has_multiyear_financials`, `has_qib_quota`, `has_anchor_book`, and `gmp_available`.

### High-Cardinality Categorical Normalization
- **`sector`:** Mapped 153 fine-grained `industry` strings into 14 standardized macro sector buckets.
- **`lead_manager_grouped`:** Reduced 124 unique lead managers to 63 groups by bucketing managers with `< 5` historical IPOs into `"Other"`.

### Skewness & Target Circuit Clipping
- **`log1p` Transforms:** Applied `np.log1p(x.clip(lower=0))` in-place to 17 right-skewed size, market cap, anchor, and subscription/application columns while preserving `NaN`s.
- **SME Circuit Limits:** Clipped SME `target_open_gain_pct` to `[-20.0, 90.0]` (43 rows) and `target_typical_gain_pct` / `target_close_gain_pct` to `[-24.0, 99.5]` (39 close rows) to match exchange listing-day bands.

### Missing Value & Leakage Rules
- **No Global Median Imputation:** Columns such as `app_qib_*`, `sub_qib_*`, and multi-year growth rates retain `NaN` in the exported CSV. Tree models (CatBoost, LightGBM) route `NaN` natively, while linear/bagging models (`Ridge`, `RandomForest`) must fit `SimpleImputer(strategy="median")` strictly inside `X_train` within cross-validation.
- **Precision Rounding:** `log1p` and fraction columns (`ofs_pct`, `broker_sub_ratio`) are rounded to 4 decimal places; percentage and ratio columns are rounded to 2 decimal places.
- **Dropped Columns:** Raw ₹ Cr sums, raw share prices, raw INR GMP, collinear `gmp_avg_pct`, sparse NII sub-tiers, `anchor_investor_count`, `peer_avg_pe`, unnormalized `industry`/`lead_manager`, and Day-1 execution OHLC prices (`list_open`, `list_high`, `list_low`, `list_close`, `list_typical_price`) are excluded from the cleaned CSV.
