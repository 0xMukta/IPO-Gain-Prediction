# Cleaned IPO Listing-Day Dataset (`ipo_listing_gain_clean.csv`)

This document describes the baseline cleaned dataset (`ipo_listing_gain_clean.csv`) prepared for predicting Day-1 IPO listing gains across Indian Mainboard and SME equity IPOs.

## Dataset Overview

- **Rows:** 1,060 listed equity IPOs (after removing 4 withdrawn/unlisted issues from the 1,064 compiled records)
- **Columns:** 66 features, identifiers, and targets
- **Time Span:** November 6, 2023 – October 15, 2026 (sorted chronologically by `listing_date`; capped at October 15, 2026 for regulatory compliance)
- **Market Segments:** Mainboard and SME IPOs across BSE and NSE
- **Scope:** Day-1 listing-day gain prediction

## Data Sources

The dataset was personally compiled and curated by me from public primary and secondary market disclosures:
- **Stock Exchanges (BSE & NSE):** Official IPO issue details, subscription figures, listing-day OHLC prices, corporate identifiers (`ISIN`, `symbol`, `bse_code`), and benchmark/volatility index series (Nifty 50, Nifty Smallcap 250, India VIX).
- **Offer Documents (RHP):** Historical prospectus financials, peer valuation multiples, and anchor investor allocations.
- **Third-Party IPO Aggregators:** Grey market premium (GMP) quotes and broker consensus review counts.

---

## Feature & Column Dictionary (66 Columns)

### 1. Identifiers & Dates (`F_META` – Excluded from Model Input `X`)
Retained in the CSV strictly for chronological train/test splitting and error auditing.

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `isin` | `str` | 0 | Unique ISIN identifier |
| `symbol` | `str` | 0 | Primary trading symbol (NSE symbol or BSE scrip code) |
| `bse_code` | `str` | 381 | Numeric BSE scrip code (`NaN` for NSE-only issues) |
| `company_name` | `str` | 0 | Registered company name |
| `open_date` | `datetime` | 0 | IPO subscription open date |
| `close_date` | `datetime` | 0 | IPO subscription close date (anchor date for macro & GMP) |
| `listing_date` | `datetime` | 0 | Exchange listing date (used for chronological splits) |

---

### 2. Market Segment, Issue Structure & Anchor Book (`F_STATIC` – Retained in `X`)

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `is_sme` | `int64` | 0 | `1` = SME IPO, `0` = Mainboard IPO |
| `exchange` | `str` | 0 | Listing exchange (`NSE_EMERGE`, `BSE_SME`, `NSE_BSE`, `BSE_ONLY`) |
| `issue_type` | `str` | 0 | `Book_Built` or `Fixed_Price` |
| `priced_at_upper_band` | `int64` | 0 | `1` if `issue_price == price_band_high` (or `Fixed_Price`), else `0` |
| `total_issue_size_cr` | `float64` | 0 | `log1p`-transformed total issue size in ₹ Cr |
| `post_issue_mcap_cr` | `float64` | 0 | `log1p`-transformed post-issue market capitalization in ₹ Cr |
| `ofs_pct` | `float64` | 0 | Offer-for-Sale fraction of total issue (`0.0` for pure fresh issues, `1.0` for pure OFS) |
| `anchor_portion_cr` | `float64` | 0 | `log1p`-transformed anchor allocation in ₹ Cr (`0.0` if no anchor book) |
| `tier1_anchor_share_pct` | `float64` | 0 | Percentage of anchor book allocated to Tier-1 institutional houses (`0.0`–`100.0`) |
| `tier1_unique_houses` | `int64` | 0 | Count of distinct Tier-1 parent AMCs/FIIs in anchor book |
| `promoter_hold_post_pct` | `float64` | 14 | Post-issue promoter shareholding percentage (`0.0`–`100.0`) |
| `has_anchor_book` | `int64` | 0 | `1` if `anchor_portion_cr > 0`, else `0` |
| `sector` | `str` | 0 | Normalized macro sector (14 standardized buckets) |
| `lead_manager_grouped` | `str` | 0 | Primary BRLM / lead manager (rare managers with `< 5` IPOs grouped as `"Other"`) |

---

### 3. Financials & Valuation Ratios (`F_FINANCIALS`)
Raw ₹ Cr sums were replaced with scale-invariant growth, margin, leverage, and valuation ratios.

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `pe_post_ipo` | `float64` | 30 | Post-IPO P/E ratio (`NaN` for loss-making firms; upper-clipped at `150.0`) |
| `relative_pe` | `float64` | 56 | `pe_post_ipo / peer_avg_pe` (upper-clipped at `10.0`; `NaN` if no listed peers or loss-making) |
| `price_to_book` | `float64` | 1 | Post-IPO Price-to-Book ratio (clipped to `[0.0, 50.0]`) |
| `price_to_sales` | `float64` | 1 | `post_issue_mcap_cr / rev_fy1` (clipped to `[0.0, 50.0]`) |
| `roe_pct` | `float64` | 1 | Return on Equity % for FY1 (clipped to `[-100.0, 150.0]`) |
| `roce_pct` | `float64` | 75 | Return on Capital Employed % for FY1 (clipped to `[-100.0, 150.0]`) |
| `debt_to_equity` | `float64` | 66 | `borrowing_fy1 / net_worth_fy1` (`NaN` if `net_worth_fy1 <= 0`; clipped to `[0.0, 20.0]`) |
| `rev_yoy_pct` | `float64` | 18 | 1-year revenue growth % (clipped to `[-100.0, 500.0]`) |
| `rev_2yr_cagr` | `float64` | 51 | 2-year revenue CAGR % (clipped to `[-100.0, 300.0]`) |
| `pat_yoy_pct` | `float64` | 67 | 1-year PAT growth % (clipped to `[-200.0, 500.0]`) |
| `ebitda_yoy_pct` | `float64` | 60 | 1-year EBITDA growth % (clipped to `[-100.0, 500.0]`) |
| `pat_margin_pct` | `float64` | 1 | FY1 PAT margin % (clipped to `[-100.0, 100.0]`) |
| `ebitda_margin_pct` | `float64` | 9 | FY1 EBITDA margin % (clipped to `[-100.0, 100.0]`) |
| `is_loss_making` | `float64` | 0 | `1.0` if `pat_fy1 <= 0` (or negative raw P/E), else `0.0` |
| `has_listed_peers` | `int64` | 0 | `1` if listed peer P/E data was available, else `0` |
| `has_multiyear_financials` | `int64` | 0 | `1` if multi-year (`FY2`) revenue is available, else `0` |

---

### 4. Grey Market & Broker Sentiment (`F_SENTIMENT`)

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `gmp_close_day_pct` | `float64` | 4 | Close-day Grey Market Premium as % of `issue_price` |
| `gmp_momentum_pct` | `float64` | 4 | `gmp_close_day_pct - gmp_avg_pct` (captures late GMP surge or drop) |
| `gmp_available` | `int64` | 0 | `1` if GMP quote was tracked, else `0` |
| `broker_total_reviews` | `float64` | 0 | Total count of published broker recommendations |
| `broker_sub_ratio` | `float64` | 246 | Share of `"Subscribe"` calls among total reviews (`NaN` when `broker_total_reviews == 0`) |

---

### 5. Subscription & Application Demand (`F_SUBS`)
All subscription multiples and application counts are `log1p`-transformed (`np.log1p(x.clip(lower=0))`). `NaN` is preserved where a quota/category did not exist, whereas `0.0` represents zero demand in an existing category.

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `sub_qib_fii_x` | `float64` | 353 | `log1p` FII / FPI QIB subscription multiple |
| `sub_qib_dfi_x` | `float64` | 353 | `log1p` DFI / Bank / Insurance QIB subscription multiple |
| `sub_qib_mf_x` | `float64` | 353 | `log1p` Domestic Mutual Fund QIB subscription multiple |
| `sub_qib_other_x` | `float64` | 353 | `log1p` Other QIB subscription multiple |
| `sub_qib_total_x` | `float64` | 190 | `log1p` overall QIB subscription multiple |
| `sub_nii_total_x` | `float64` | 105 | `log1p` overall NII (HNI) subscription multiple |
| `sub_rii_x` | `float64` | 22 | `log1p` Retail (RII) subscription multiple |
| `sub_total_x` | `float64` | 0 | `log1p` overall total subscription multiple |
| `app_qib_fii` | `float64` | 445 | `log1p` FII QIB application count |
| `app_qib_dfi` | `float64` | 445 | `log1p` DFI QIB application count |
| `app_qib_mf` | `float64` | 445 | `log1p` Mutual Fund QIB application count |
| `app_qib_other` | `float64` | 445 | `log1p` Other QIB application count |
| `app_nii_x` | `float64` | 522 | `log1p` NII application-wise subscription |
| `app_rii_x` | `float64` | 452 | `log1p` Retail application-wise subscription |
| `has_qib_quota` | `float64` | 16 | `1.0` if QIB quota present, `0.0` if no QIB quota, `NaN` if category breakdown unavailable |

---

### 6. Broad Market Macro Regime (`F_MACRO`)
Anchored strictly to `close_date` to prevent future leakage.

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `nifty50_ret_5d_pct` | `float64` | 0 | Nifty 50 5-trading-day return % up to `close_date` |
| `nifty50_ret_22d_pct` | `float64` | 0 | Nifty 50 22-trading-day return % up to `close_date` |
| `smallcap250_ret_5d_pct` | `float64` | 0 | Nifty Smallcap 250 5-trading-day return % up to `close_date` |
| `smallcap250_ret_22d_pct` | `float64` | 0 | Nifty Smallcap 250 22-trading-day return % up to `close_date` |
| `india_vix_close` | `float64` | 0 | India VIX closing level on `close_date` |
| `india_vix_5d_chg_pct` | `float64` | 0 | 5-trading-day % change in India VIX |

---

### 7. Day-1 Listing Targets (`F_TARGETS`)
Only one target should be selected per model run; the other two must be excluded from `X`.

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `target_open_gain_pct` | `float64` | 0 | Listing open gain % vs `issue_price` (SME clipped to `[-20.0, 90.0]`) |
| `target_typical_gain_pct` | `float64` | 0 | Listing typical-price `(O + H + L + 2C) / 5` gain % vs `issue_price` (SME clipped to `[-24.0, 99.5]`) |
| `target_close_gain_pct` | `float64` | 0 | Listing close gain % vs `issue_price` (SME clipped to `[-24.0, 99.5]`) |
