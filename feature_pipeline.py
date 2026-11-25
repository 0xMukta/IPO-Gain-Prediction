"""Shared preprocessing & feature transformation pipeline for training and UI inference.

Applies the exact rules from Preprocessing.ipynb / Preprocessing.md to raw IPO inputs
(single-row UI entry or uploaded CSV) and returns the clean, model-ready feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 14 standardized macro sectors
MACRO_SECTORS = [
    "Agriculture",
    "Auto & Ancillaries",
    "Chemicals & Petrochemicals",
    "Consumer Goods & Retail",
    "Energy & Utilities",
    "Financials",
    "Healthcare & Pharma",
    "IT & Tech",
    "Industrials & Capital Goods",
    "Infrastructure & Construction",
    "Logistics & Transport",
    "Media & Entertainment",
    "Other/Diversified",
    "Real Estate",
]

EXCHANGES = ["NSE_EMERGE", "BSE_SME", "NSE_BSE", "BSE_ONLY"]
ISSUE_TYPES = ["Book_Built", "Fixed_Price"]

RAW_LOG1P_COLS = [
    "total_issue_size_cr",
    "post_issue_mcap_cr",
    "anchor_portion_cr",
    "sub_total_x",
    "sub_rii_x",
    "sub_nii_total_x",
    "sub_qib_total_x",
    "sub_qib_fii_x",
    "sub_qib_dfi_x",
    "sub_qib_mf_x",
    "sub_qib_other_x",
]

# Columns excluded from X per Predictor_Pipeline.md
META_EXCLUDE_COLS = [
    "isin",
    "symbol",
    "bse_code",
    "company_name",
    "open_date",
    "close_date",
    "listing_date",
]
TARGET_COLS = [
    "target_open_gain_pct",
    "target_typical_gain_pct",
    "target_close_gain_pct",
]
TEMPORAL_PROXY_COLS = [
    "app_qib_fii",
    "app_qib_dfi",
    "app_qib_mf",
    "app_qib_other",
    "app_nii_x",
    "app_rii_x",
]
GMP_COLS = ["gmp_close_day_pct", "gmp_momentum_pct", "gmp_available"]
CATEGORICAL_COLS = ["exchange", "issue_type", "sector", "lead_manager_grouped"]


def map_industry_to_sector(ind: object) -> str:
    """Map raw industry text (or pre-selected macro sector) into 14 macro buckets.

    Returns 'Other/Diversified' when missing/NaN so categorical encoders stay clean.
    """
    if pd.isna(ind) or not str(ind).strip():
        return "Other/Diversified"
    raw_str = str(ind).strip()
    if raw_str in MACRO_SECTORS:
        return raw_str
    s = raw_str.lower()
    if any(k in s for k in ["bank", "nbfc", "financial", "fintech", "insurance", "stockbroking", "asset management", "microfinance", "capital market", "finance"]):
        return "Financials"
    if any(k in s for k in ["pharma", "hospital", "healthcare", "medical", "diagnostic", "biotech", "path lab"]):
        return "Healthcare & Pharma"
    if any(k in s for k in ["software", "it ", "it-", "information technology", "e-learning", "bpo", "kpo", "computers", "cyber", "saas", "telecom"]):
        return "IT & Tech"
    if any(k in s for k in ["auto", "wheeler", "tractor", "passenger car", "commercial vehicle"]):
        return "Auto & Ancillaries"
    if any(k in s for k in ["chemical", "petrochemical", "pesticide", "agrochemical", "fertilizer", "dyes", "pigment"]):
        return "Chemicals & Petrochemicals"
    if any(k in s for k in ["real estate", "residential", "commercial project", "housing", "reit"]):
        return "Real Estate"
    if any(k in s for k in ["construction", "civil", "infrastructure", "road asset", "cement", "granite", "marble", "building material", "sanitary", "ceramics", "water supply"]):
        return "Infrastructure & Construction"
    if any(k in s for k in ["logistics", "transport", "shipping", "airline", "courier", "supply chain", "port", "warehousing"]):
        return "Logistics & Transport"
    if any(k in s for k in ["power", "energy", "solar", "oil", "gas", "lpg", "cng", "coal", "petroleum", "utility", "utilities"]):
        return "Energy & Utilities"
    if any(k in s for k in ["media", "entertainment", "advertising", "film", "printing", "publication", "event", "exhibitions"]):
        return "Media & Entertainment"
    if any(k in s for k in ["agricult", "seed", "dairy", "seafood", "tea", "coffee", "plantation"]):
        return "Agriculture"
    if any(k in s for k in ["garment", "apparel", "textile", "jewellery", "gems", "watch", "food", "fmcg", "retail", "e-retail", "e-commerce", "furniture", "houseware", "footwear", "personal care", "hotel", "restaurant", "tour", "travel", "leisure", "appliance", "breweries", "consumer"]):
        return "Consumer Goods & Retail"
    if any(k in s for k in ["industrial", "electrical", "steel", "metal", "aluminium", "copper", "zinc", "castings", "forging", "cable", "packaging", "paper", "plastic", "aerospace", "defense", "defence", "engineering", "glass", "rubber", "waste", "compressor", "pump", "machinery"]):
        return "Industrials & Capital Goods"
    return "Other/Diversified"


def get_feature_columns(include_gmp: bool = True) -> list[str]:
    """Return the exact ordered list of model input columns X."""
    base_cols = [
        "is_sme",
        "exchange",
        "issue_type",
        "priced_at_upper_band",
        "total_issue_size_cr",
        "post_issue_mcap_cr",
        "anchor_portion_cr",
        "tier1_anchor_share_pct",
        "tier1_unique_houses",
        "promoter_hold_post_pct",
        "pe_post_ipo",
        "price_to_book",
        "roe_pct",
        "roce_pct",
        "sub_qib_fii_x",
        "sub_qib_dfi_x",
        "sub_qib_mf_x",
        "sub_qib_other_x",
        "sub_qib_total_x",
        "sub_nii_total_x",
        "sub_rii_x",
        "sub_total_x",
        "nifty50_ret_5d_pct",
        "nifty50_ret_22d_pct",
        "smallcap250_ret_5d_pct",
        "smallcap250_ret_22d_pct",
        "india_vix_close",
        "india_vix_5d_chg_pct",
        "ofs_pct",
        "gmp_close_day_pct",
        "gmp_momentum_pct",
        "rev_yoy_pct",
        "pat_yoy_pct",
        "rev_2yr_cagr",
        "ebitda_yoy_pct",
        "pat_margin_pct",
        "ebitda_margin_pct",
        "debt_to_equity",
        "price_to_sales",
        "is_loss_making",
        "relative_pe",
        "broker_total_reviews",
        "broker_sub_ratio",
        "has_listed_peers",
        "has_multiyear_financials",
        "has_qib_quota",
        "has_anchor_book",
        "gmp_available",
        "sector",
        "lead_manager_grouped",
    ]
    if not include_gmp:
        base_cols = [c for c in base_cols if c not in GMP_COLS]
    return base_cols


def preprocess_raw_records(
    raw_df: pd.DataFrame,
    frequent_lead_managers: set[str] | None = None,
) -> pd.DataFrame:
    """Transform raw IPO rows into the clean feature table matching ipo_listing_gain_clean.csv."""
    df = raw_df.copy()

    expected_numeric = [
        "is_sme", "price_band_high", "issue_price", "priced_at_upper_band",
        "total_issue_size_cr", "post_issue_mcap_cr", "fresh_issue_cr", "ofs_size_cr",
        "anchor_portion_cr", "tier1_anchor_share_pct", "tier1_unique_houses",
        "promoter_hold_post_pct", "rev_fy1", "rev_fy2", "rev_fy3",
        "pat_fy1", "pat_fy2", "pat_fy3", "ebitda_fy1", "ebitda_fy2",
        "net_worth_fy1", "borrowing_fy1", "pe_post_ipo", "price_to_book",
        "peer_avg_pe", "roe_pct", "roce_pct",
        "broker_sub", "broker_may", "broker_neutral", "broker_avoid",
        "gmp_close_day_inr", "gmp_avg_inr",
        "sub_qib_fii_x", "sub_qib_dfi_x", "sub_qib_mf_x", "sub_qib_other_x",
        "sub_qib_total_x", "sub_nii_total_x", "sub_rii_x", "sub_total_x",
        "nifty50_ret_5d_pct", "nifty50_ret_22d_pct",
        "smallcap250_ret_5d_pct", "smallcap250_ret_22d_pct",
        "india_vix_close", "india_vix_5d_chg_pct",
    ]
    for col in expected_numeric:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, default in [("exchange", "NSE_BSE"), ("issue_type", "Book_Built"), ("industry", np.nan), ("lead_manager", np.nan)]:
        if col not in df.columns:
            df[col] = default

    df["is_sme"] = df["is_sme"].fillna(0).astype(int)
    df["exchange"] = df["exchange"].fillna("NSE_BSE").astype(str).str.strip()
    df["issue_type"] = df["issue_type"].fillna("Book_Built").astype(str).str.strip()
    df.loc[(df["is_sme"] == 1) & (df["exchange"] == "NSE_ONLY"), "exchange"] = "NSE_EMERGE"

    # Derive priced_at_upper_band if missing
    inferred_upper = np.where(
        df["issue_type"] == "Fixed_Price",
        1,
        np.where(
            df["price_band_high"].notna() & df["issue_price"].notna(),
            (df["issue_price"] >= df["price_band_high"]).astype(int),
            1,
        ),
    )
    df["priced_at_upper_band"] = df["priced_at_upper_band"].fillna(pd.Series(inferred_upper, index=df.index)).astype(int)

    # Structural zeros for issue composition and anchor book
    df["fresh_issue_cr"] = df["fresh_issue_cr"].fillna(0.0).clip(lower=0.0)
    df["ofs_size_cr"] = df["ofs_size_cr"].fillna(0.0).clip(lower=0.0)
    df["anchor_portion_cr"] = df["anchor_portion_cr"].fillna(0.0).clip(lower=0.0)
    df["tier1_anchor_share_pct"] = df["tier1_anchor_share_pct"].fillna(0.0).clip(0.0, 100.0)
    df["tier1_unique_houses"] = df["tier1_unique_houses"].fillna(0).clip(lower=0).astype(int)
    df.loc[df["borrowing_fy1"] < 0, "borrowing_fy1"] = 0.0

    # Derived ratios
    df["ofs_pct"] = np.where(df["total_issue_size_cr"] > 0, df["ofs_size_cr"] / df["total_issue_size_cr"], 0.0).clip(0.0, 1.0)

    valid_price = df["issue_price"] > 0
    df["gmp_close_day_pct"] = np.where(valid_price, (df["gmp_close_day_inr"] / df["issue_price"]) * 100.0, np.nan)
    gmp_avg_pct = np.where(valid_price, (df["gmp_avg_inr"] / df["issue_price"]) * 100.0, np.nan)
    df["gmp_momentum_pct"] = df["gmp_close_day_pct"] - gmp_avg_pct

    df["rev_yoy_pct"] = np.where(df["rev_fy2"] > 0, ((df["rev_fy1"] - df["rev_fy2"]) / df["rev_fy2"]) * 100.0, np.nan).clip(-100.0, 500.0)
    df["pat_yoy_pct"] = np.where(df["pat_fy2"] > 0, ((df["pat_fy1"] - df["pat_fy2"]) / df["pat_fy2"]) * 100.0, np.nan).clip(-200.0, 500.0)
    df["rev_2yr_cagr"] = np.where(
        (df["rev_fy1"] > 0) & (df["rev_fy3"] > 0),
        (np.sqrt(df["rev_fy1"] / df["rev_fy3"]) - 1.0) * 100.0,
        np.nan,
    ).clip(-100.0, 300.0)
    df["ebitda_yoy_pct"] = np.where(df["ebitda_fy2"] > 0, ((df["ebitda_fy1"] - df["ebitda_fy2"]) / df["ebitda_fy2"]) * 100.0, np.nan).clip(-100.0, 500.0)
    df["pat_margin_pct"] = np.where(df["rev_fy1"] > 0, (df["pat_fy1"] / df["rev_fy1"]) * 100.0, np.nan).clip(-100.0, 100.0)
    df["ebitda_margin_pct"] = np.where(df["rev_fy1"] > 0, (df["ebitda_fy1"] / df["rev_fy1"]) * 100.0, np.nan).clip(-100.0, 100.0)

    df["debt_to_equity"] = np.where(df["net_worth_fy1"] > 0, df["borrowing_fy1"] / df["net_worth_fy1"], np.nan).clip(0.0, 20.0)
    df["price_to_sales"] = np.where(df["rev_fy1"] > 0, df["post_issue_mcap_cr"] / df["rev_fy1"], np.nan).clip(0.0, 50.0)

    book_val_post = df["net_worth_fy1"] + df["fresh_issue_cr"]
    derived_pb = np.where(book_val_post > 0, df["post_issue_mcap_cr"] / book_val_post, np.nan)
    df["price_to_book"] = df["price_to_book"].fillna(pd.Series(derived_pb, index=df.index)).clip(0.0, 50.0)
    df["roe_pct"] = df["roe_pct"].clip(-100.0, 150.0)
    df["roce_pct"] = df["roce_pct"].clip(-100.0, 150.0)

    # Derive is_loss_making BEFORE setting negative P/E to NaN so loss-makers aren't treated as cheap
    df["is_loss_making"] = np.where(
        df["pat_fy1"].notnull(),
        (df["pat_fy1"] <= 0).astype(float),
        np.where(df["pe_post_ipo"] < 0, 1.0, np.where(df["pe_post_ipo"] > 0, 0.0, np.nan)),
    )
    df.loc[df["pe_post_ipo"] <= 0, "pe_post_ipo"] = np.nan
    df["pe_post_ipo"] = df["pe_post_ipo"].clip(upper=150.0)

    valid_pe = (df["pe_post_ipo"] > 0) & (df["peer_avg_pe"] > 0)
    df["relative_pe"] = np.clip(np.where(valid_pe, df["pe_post_ipo"] / df["peer_avg_pe"], np.nan), None, 10.0)

    broker_cols = ["broker_sub", "broker_may", "broker_neutral", "broker_avoid"]
    df["broker_total_reviews"] = df[broker_cols].fillna(0).sum(axis=1)
    df["broker_sub_ratio"] = np.where(
        df["broker_total_reviews"] > 0,
        df["broker_sub"].fillna(0) / df["broker_total_reviews"],
        np.nan,
    )

    df["has_listed_peers"] = df["peer_avg_pe"].notnull().astype(int)
    df["has_multiyear_financials"] = df["rev_fy2"].notnull().astype(int)
    missing_cat_breakdown = df["sub_qib_total_x"].isna() & df["sub_nii_total_x"].isna() & df["sub_rii_x"].isna()
    df["has_qib_quota"] = np.where(missing_cat_breakdown, np.nan, df["sub_qib_total_x"].notnull().astype(float))
    df["has_anchor_book"] = (df["anchor_portion_cr"] > 0).astype(int)
    df["gmp_available"] = df["gmp_close_day_pct"].notnull().astype(int)

    # Categoricals
    if "sector" in df.columns and df["sector"].notna().any():
        df["sector"] = np.where(
            df["sector"].notna() & (df["sector"].astype(str).str.strip() != ""),
            df["sector"].apply(map_industry_to_sector),
            df["industry"].apply(map_industry_to_sector),
        )
    else:
        df["sector"] = df["industry"].apply(map_industry_to_sector)

    lm_clean = df["lead_manager"].fillna("Other").astype(str).str.strip().replace("", "Other")
    if frequent_lead_managers:
        df["lead_manager_grouped"] = np.where(lm_clean.isin(frequent_lead_managers), lm_clean, "Other")
    else:
        df["lead_manager_grouped"] = lm_clean

    # log1p compresses right-skewed multiples while preserving NaNs (no global median imputation)
    for col in RAW_LOG1P_COLS:
        df[col] = np.log1p(df[col].clip(lower=0))

    feature_cols = get_feature_columns(include_gmp=True)
    out = df[feature_cols].copy()

    four_dec_cols = [c for c in RAW_LOG1P_COLS + ["ofs_pct", "broker_sub_ratio"] if c in out.columns]
    two_dec_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c not in four_dec_cols and out[c].dtype != "int64"]
    out[four_dec_cols] = out[four_dec_cols].round(4)
    out[two_dec_cols] = out[two_dec_cols].round(2)

    return out
