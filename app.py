"""Streamlit Demo App – IPO Listing-Day Gain Prediction & Raw Data Preprocessor.

Supports:
1. Interactive raw IPO entry (with dropdowns for coded/categorical fields and explicit NaN support)
2. CSV batch upload + downloadable sample_ipo_input.csv + UI fallback for missing CSV fields
3. Live raw-to-clean feature transformation matching Preprocessing.md + CSV download
4. CatBoost MultiQuantile (P10 / P50 / P90) & Weighted Ensemble predictions + per-IPO SHAP drivers
5. Stored EDA insights & With-GMP vs. Without-GMP model benchmark dashboards
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostRegressor

from feature_pipeline import (
    CATEGORICAL_COLS,
    EXCHANGES,
    ISSUE_TYPES,
    MACRO_SECTORS,
    get_feature_columns,
    preprocess_raw_records,
)
from train import SME_CLIP_BOUNDS, clip_predictions, prepare_catboost_df

ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

st.set_page_config(
    page_title="IPO Listing-Day Gain Predictor",
    page_icon="📈",
    layout="wide",
)


def parse_optional_float(val: str) -> float:
    """Convert text input to float, returning np.nan when empty."""
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na"}:
        return float(np.nan)
    return float(s)


def parse_optional_int(val: str) -> float:
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "na"}:
        return float(np.nan)
    return float(int(float(s)))


@st.cache_resource
def load_artifacts() -> dict[str, Any]:
    with open(os.path.join(ARTIFACTS_DIR, "model_config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "model_metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    eda_path = os.path.join(ARTIFACTS_DIR, "eda_stats.json")
    eda_stats = json.load(open(eda_path, encoding="utf-8")) if os.path.exists(eda_path) else {}

    models = {}
    for mode in ("with_gmp", "no_gmp"):
        mq = CatBoostRegressor()
        mq.load_model(os.path.join(ARTIFACTS_DIR, f"catboost_mq_{mode}.cbm"))
        rmse = CatBoostRegressor()
        rmse.load_model(os.path.join(ARTIFACTS_DIR, f"catboost_rmse_{mode}.cbm"))
        weights = config["with_gmp_blend_weights"] if mode == "with_gmp" else config["no_gmp_blend_weights"]
        rf_path = os.path.join(ARTIFACTS_DIR, f"rf_{mode}.joblib")
        rf = joblib.load(rf_path) if weights.get("w_rf", 0) > 0 and os.path.exists(rf_path) else None
        models[mode] = {"mq": mq, "rmse": rmse, "rf": rf}

    return {"config": config, "metrics": metrics, "eda": eda_stats, "models": models}


def run_predictions(
    clean_df: pd.DataFrame,
    include_gmp: bool,
    use_ensemble: bool,
    artifacts: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    mode = "with_gmp" if include_gmp else "no_gmp"
    config = artifacts["config"]
    target_col = config["target_col"]
    feature_cols = get_feature_columns(include_gmp=include_gmp)
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    X = clean_df[feature_cols].copy()
    X_cb = prepare_catboost_df(X, cat_cols)
    sme_arr = clean_df["is_sme"].to_numpy(dtype=int)

    m_bundle = artifacts["models"][mode]
    mq_preds = np.sort(m_bundle["mq"].predict(X_cb), axis=1)
    mq_preds = clip_predictions(mq_preds, sme_arr, target_col)

    rmse_pred = clip_predictions(m_bundle["rmse"].predict(X_cb), sme_arr, target_col)
    weights = config["with_gmp_blend_weights"] if include_gmp else config["no_gmp_blend_weights"]
    if use_ensemble:
        rf_pred = (
            clip_predictions(m_bundle["rf"].predict(X), sme_arr, target_col)
            if m_bundle["rf"] is not None
            else 0.0
        )
        point_pred = (
            weights["w_p50"] * mq_preds[:, 1]
            + weights["w_rmse"] * rmse_pred
            + weights["w_rf"] * rf_pred
        )
        point_pred = clip_predictions(point_pred, sme_arr, target_col)
    else:
        point_pred = mq_preds[:, 1]

    explainer = shap.TreeExplainer(m_bundle["rmse"])
    shap_vals = explainer.shap_values(X_cb)

    out = pd.DataFrame(
        {
            "Pred_P10_Worst_pct": np.round(mq_preds[:, 0], 2),
            "Pred_Expected_pct": np.round(point_pred, 2),
            "Pred_P50_Median_pct": np.round(mq_preds[:, 1], 2),
            "Pred_P90_Upside_pct": np.round(mq_preds[:, 2], 2),
        },
        index=clean_df.index,
    )
    return out, shap_vals, X_cb


def main() -> None:
    st.title("📈 Indian IPO Listing-Day Gain Predictor")
    st.caption(
        "Course: Introduction to Machine Learning (PCCAIML502) – Day-1 Gain Prediction across Mainboard & SME IPOs (Nov 2023 – Oct 15, 2026)"
    )

    if not os.path.exists(os.path.join(ARTIFACTS_DIR, "model_config.json")):
        st.error("Trained model artifacts not found. Run `python train.py` first.")
        return

    artifacts = load_artifacts()
    config = artifacts["config"]
    metrics = artifacts["metrics"]
    eda = artifacts["eda"]

    # Sidebar Controls
    with st.sidebar:
        st.header("Model Configuration")
        include_gmp = st.toggle("Use GMP Features (With GMP)", value=True)
        use_ensemble = st.toggle(
            "Use CV-Optimized Weighted Ensemble",
            value=False,
            help="Off = CatBoost MultiQuantile P50 (lowest holdout MAE); On = blends CatBoost P50, CatBoost RMSE, and Random Forest using CV-tuned weights.",
        )
        st.markdown("---")
        st.subheader("Sample Input Template")
        if os.path.exists("sample_ipo_input.csv"):
            with open("sample_ipo_input.csv", "rb") as f:
                st.download_button(
                    label="⬇️ Download sample_ipo_input.csv",
                    data=f.read(),
                    file_name="sample_ipo_input.csv",
                    mime="text/csv",
                )
        st.markdown("---")
        st.caption(
            "**Disclaimer:** Academic coursework project (PCCAIML502). Not SEBI-registered financial advice."
        )

    tab_predict, tab_eda, tab_bench = st.tabs(
        ["🔮 Predict & Preprocess Raw Data", "📊 EDA & Historical Gain Stats", "🧪 Model Benchmarks & GMP Ablation"]
    )

    # =========================================================================
    # TAB 1: RAW DATA INPUT, CSV UPLOAD, CLEANING & PREDICTION
    # =========================================================================
    with tab_predict:
        st.subheader("1. Enter Raw IPO Fields (or Provide Fallbacks for CSV Upload)")
        st.info(
            "Leave optional numeric fields blank or choose **Leave Empty (NaN)** in dropdowns to pass missing values natively without fake zero/median imputation."
        )

        frequent_lms = config.get("frequent_lead_managers", [])
        lm_options = ["Leave Empty (NaN)", "Other"] + frequent_lms
        sector_options = ["Leave Empty (NaN / Other)"] + MACRO_SECTORS

        with st.expander("🧾 Raw IPO Input Form (Single IPO / CSV Missing-Field Fallback)", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            company_name = c1.text_input("company_name", value="Demo Equity IPO Ltd.")
            is_sme_label = c2.selectbox("is_sme (Market Segment)", ["0 – Mainboard", "1 – SME"])
            is_sme_val = int(is_sme_label.split(" ")[0])
            exchange_val = c3.selectbox("exchange", EXCHANGES, index=2 if is_sme_val == 0 else 0)
            issue_type_val = c4.selectbox("issue_type", ISSUE_TYPES, index=0)

            c1, c2, c3, c4 = st.columns(4)
            sector_choice = c1.selectbox("industry / sector", sector_options, index=0)
            custom_industry = c2.text_input("Optional raw industry text (overrides dropdown)", value="")
            lm_choice = c3.selectbox("lead_manager", lm_options, index=0)
            upper_choice = c4.selectbox("priced_at_upper_band", ["Auto-derive", "1 – Yes", "0 – No"], index=0)

            st.markdown("**Issue Structure & Anchor Book**")
            c1, c2, c3, c4 = st.columns(4)
            price_band_high = c1.text_input("price_band_high (₹)", value="250")
            issue_price = c2.text_input("issue_price (₹)", value="250")
            total_issue_size_cr = c3.text_input("total_issue_size_cr (₹ Cr)", value="450")
            post_issue_mcap_cr = c4.text_input("post_issue_mcap_cr (₹ Cr)", value="1800")

            c1, c2, c3, c4 = st.columns(4)
            fresh_issue_cr = c1.text_input("fresh_issue_cr (₹ Cr, blank=0)", value="300")
            ofs_size_cr = c2.text_input("ofs_size_cr (₹ Cr, blank=0)", value="150")
            anchor_portion_cr = c3.text_input("anchor_portion_cr (₹ Cr, blank=0)", value="135")
            tier1_anchor_share_pct = c4.text_input("tier1_anchor_share_pct (%)", value="65")

            c1, c2, c3, c4 = st.columns(4)
            tier1_unique_houses = c1.text_input("tier1_unique_houses", value="8")
            promoter_hold_post_pct = c2.text_input("promoter_hold_post_pct (%)", value="64.5")
            gmp_close_day_inr = c3.text_input("gmp_close_day_inr (₹, blank=NaN)", value="60")
            gmp_avg_inr = c4.text_input("gmp_avg_inr (₹, blank=NaN)", value="52")

            st.markdown("**Financials & Valuation (Leave blank for NaN)**")
            c1, c2, c3, c4, c5 = st.columns(5)
            rev_fy1 = c1.text_input("rev_fy1 (₹ Cr)", value="420")
            rev_fy2 = c2.text_input("rev_fy2 (₹ Cr)", value="310")
            rev_fy3 = c3.text_input("rev_fy3 (₹ Cr)", value="225")
            pat_fy1 = c4.text_input("pat_fy1 (₹ Cr)", value="52")
            pat_fy2 = c5.text_input("pat_fy2 (₹ Cr)", value="34")

            c1, c2, c3, c4, c5 = st.columns(5)
            ebitda_fy1 = c1.text_input("ebitda_fy1 (₹ Cr)", value="84")
            ebitda_fy2 = c2.text_input("ebitda_fy2 (₹ Cr)", value="58")
            net_worth_fy1 = c3.text_input("net_worth_fy1 (₹ Cr)", value="210")
            borrowing_fy1 = c4.text_input("borrowing_fy1 (₹ Cr)", value="40")
            pe_post_ipo = c5.text_input("pe_post_ipo", value="34.6")

            c1, c2, c3, c4, c5 = st.columns(5)
            price_to_book = c1.text_input("price_to_book", value="4.2")
            peer_avg_pe = c2.text_input("peer_avg_pe", value="42.0")
            roe_pct = c3.text_input("roe_pct (%)", value="24.8")
            roce_pct = c4.text_input("roce_pct (%)", value="27.2")
            pat_fy3 = c5.text_input("pat_fy3 (₹ Cr)", value="22")

            st.markdown("**Subscription Multiples (Raw x), Broker Calls & Macro Regime**")
            c1, c2, c3, c4, c5 = st.columns(5)
            sub_qib_fii_x = c1.text_input("sub_qib_fii_x", value="18.5")
            sub_qib_dfi_x = c2.text_input("sub_qib_dfi_x", value="12.0")
            sub_qib_mf_x = c3.text_input("sub_qib_mf_x", value="6.5")
            sub_qib_other_x = c4.text_input("sub_qib_other_x", value="4.2")
            sub_qib_total_x = c5.text_input("sub_qib_total_x", value="41.2")

            c1, c2, c3, c4, c5 = st.columns(5)
            sub_nii_total_x = c1.text_input("sub_nii_total_x", value="55.0")
            sub_rii_x = c2.text_input("sub_rii_x", value="18.4")
            sub_total_x = c3.text_input("sub_total_x", value="34.5")
            broker_sub = c4.text_input("broker_sub", value="6")
            broker_may = c5.text_input("broker_may", value="1")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            broker_neutral = c1.text_input("broker_neutral", value="1")
            broker_avoid = c2.text_input("broker_avoid", value="0")
            nifty50_ret_5d_pct = c3.text_input("nifty50_ret_5d_pct", value="1.1")
            nifty50_ret_22d_pct = c4.text_input("nifty50_ret_22d_pct", value="3.2")
            smallcap250_ret_5d_pct = c5.text_input("smallcap250_ret_5d_pct", value="1.6")
            smallcap250_ret_22d_pct = c6.text_input("smallcap250_ret_22d_pct", value="4.5")

            c1, c2 = st.columns(2)
            india_vix_close = c1.text_input("india_vix_close", value="13.4")
            india_vix_5d_chg_pct = c2.text_input("india_vix_5d_chg_pct", value="-2.1")

        ui_industry = (
            custom_industry.strip()
            if custom_industry.strip()
            else (np.nan if sector_choice.startswith("Leave Empty") else sector_choice)
        )
        ui_lm = np.nan if lm_choice.startswith("Leave Empty") else lm_choice
        ui_upper = np.nan if upper_choice == "Auto-derive" else int(upper_choice.split(" ")[0])

        ui_row = {
            "company_name": company_name,
            "is_sme": is_sme_val,
            "exchange": exchange_val,
            "issue_type": issue_type_val,
            "industry": ui_industry,
            "lead_manager": ui_lm,
            "price_band_high": parse_optional_float(price_band_high),
            "issue_price": parse_optional_float(issue_price),
            "priced_at_upper_band": ui_upper,
            "total_issue_size_cr": parse_optional_float(total_issue_size_cr),
            "post_issue_mcap_cr": parse_optional_float(post_issue_mcap_cr),
            "fresh_issue_cr": parse_optional_float(fresh_issue_cr),
            "ofs_size_cr": parse_optional_float(ofs_size_cr),
            "anchor_portion_cr": parse_optional_float(anchor_portion_cr),
            "tier1_anchor_share_pct": parse_optional_float(tier1_anchor_share_pct),
            "tier1_unique_houses": parse_optional_int(tier1_unique_houses),
            "promoter_hold_post_pct": parse_optional_float(promoter_hold_post_pct),
            "rev_fy1": parse_optional_float(rev_fy1),
            "rev_fy2": parse_optional_float(rev_fy2),
            "rev_fy3": parse_optional_float(rev_fy3),
            "pat_fy1": parse_optional_float(pat_fy1),
            "pat_fy2": parse_optional_float(pat_fy2),
            "pat_fy3": parse_optional_float(pat_fy3),
            "ebitda_fy1": parse_optional_float(ebitda_fy1),
            "ebitda_fy2": parse_optional_float(ebitda_fy2),
            "net_worth_fy1": parse_optional_float(net_worth_fy1),
            "borrowing_fy1": parse_optional_float(borrowing_fy1),
            "pe_post_ipo": parse_optional_float(pe_post_ipo),
            "price_to_book": parse_optional_float(price_to_book),
            "peer_avg_pe": parse_optional_float(peer_avg_pe),
            "roe_pct": parse_optional_float(roe_pct),
            "roce_pct": parse_optional_float(roce_pct),
            "broker_sub": parse_optional_int(broker_sub),
            "broker_may": parse_optional_int(broker_may),
            "broker_neutral": parse_optional_int(broker_neutral),
            "broker_avoid": parse_optional_int(broker_avoid),
            "gmp_close_day_inr": parse_optional_float(gmp_close_day_inr),
            "gmp_avg_inr": parse_optional_float(gmp_avg_inr),
            "sub_qib_fii_x": parse_optional_float(sub_qib_fii_x),
            "sub_qib_dfi_x": parse_optional_float(sub_qib_dfi_x),
            "sub_qib_mf_x": parse_optional_float(sub_qib_mf_x),
            "sub_qib_other_x": parse_optional_float(sub_qib_other_x),
            "sub_qib_total_x": parse_optional_float(sub_qib_total_x),
            "sub_nii_total_x": parse_optional_float(sub_nii_total_x),
            "sub_rii_x": parse_optional_float(sub_rii_x),
            "sub_total_x": parse_optional_float(sub_total_x),
            "nifty50_ret_5d_pct": parse_optional_float(nifty50_ret_5d_pct),
            "nifty50_ret_22d_pct": parse_optional_float(nifty50_ret_22d_pct),
            "smallcap250_ret_5d_pct": parse_optional_float(smallcap250_ret_5d_pct),
            "smallcap250_ret_22d_pct": parse_optional_float(smallcap250_ret_22d_pct),
            "india_vix_close": parse_optional_float(india_vix_close),
            "india_vix_5d_chg_pct": parse_optional_float(india_vix_5d_chg_pct),
        }

        st.markdown("---")
        st.subheader("2. Optional: Upload Raw CSV (Missing Columns Filled From UI)")
        uploaded_file = st.file_uploader("Upload raw IPO CSV (see sample_ipo_input.csv in sidebar)", type=["csv"])
        fill_null_cells = st.checkbox(
            "Also fill NaN cells inside uploaded CSV columns using UI form values (uncheck to preserve CSV NaNs)",
            value=False,
        )

        if uploaded_file is not None:
            raw_input_df = pd.read_csv(uploaded_file)
            for k, v in ui_row.items():
                if k not in raw_input_df.columns:
                    raw_input_df[k] = v
                elif fill_null_cells:
                    raw_input_df[k] = raw_input_df[k].fillna(v)
            st.success(f"Loaded {len(raw_input_df)} row(s) from uploaded CSV; missing columns supplemented from UI.")
        else:
            raw_input_df = pd.DataFrame([ui_row])

        if st.button("🚀 Preprocess & Predict Listing Gain", type="primary"):
            clean_df = preprocess_raw_records(raw_input_df, frequent_lead_managers=set(frequent_lms))
            preds_df, shap_vals, X_cb = run_predictions(clean_df, include_gmp, use_ensemble, artifacts)

            display_names = (
                raw_input_df["company_name"].fillna("Unnamed IPO").astype(str)
                if "company_name" in raw_input_df.columns
                else pd.Series([f"Row {i+1}" for i in range(len(clean_df))])
            )
            results_df = pd.concat([display_names.rename("company_name"), preds_df], axis=1)

            st.subheader("Prediction Results")
            if len(results_df) == 1:
                r = results_df.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Expected Gain (Model)", f"{r['Pred_Expected_pct']:+.2f}%")
                m2.metric("Worst-Case Downside (P10)", f"{r['Pred_P10_Worst_pct']:+.2f}%")
                m3.metric("Expected Median (P50)", f"{r['Pred_P50_Median_pct']:+.2f}%")
                m4.metric("Upside Potential (P90)", f"{r['Pred_P90_Upside_pct']:+.2f}%")
            else:
                st.dataframe(results_df, width="stretch")

            # Per-IPO SHAP explanation for first row (or selected row)
            row_idx = 0
            if len(results_df) > 1:
                row_idx = st.selectbox(
                    "Inspect SHAP Drivers for Row:",
                    options=list(range(len(results_df))),
                    format_func=lambda i: f"{i+1}. {results_df.iloc[i]['company_name']}",
                )
            st.markdown(f"**Top SHAP Feature Drivers for `{results_df.iloc[row_idx]['company_name']}`**")
            shap_row = pd.DataFrame(
                {
                    "feature": X_cb.columns,
                    "cleaned_value": [X_cb.iloc[row_idx][c] for c in X_cb.columns],
                    "shap_impact_pct": np.round(shap_vals[row_idx], 2),
                }
            )
            shap_row["abs_impact"] = shap_row["shap_impact_pct"].abs()
            top_local = shap_row.sort_values("abs_impact", ascending=False).head(10).drop(columns=["abs_impact"])
            st.dataframe(top_local, width="stretch")

            st.subheader("Cleaned Model-Ready Features (Per Preprocessing.md)")
            st.dataframe(clean_df, width="stretch")
            st.download_button(
                label="⬇️ Download Cleaned Features CSV",
                data=clean_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="cleaned_ipo_features.csv",
                mime="text/csv",
            )

    # =========================================================================
    # TAB 2: EDA & HISTORICAL STATS
    # =========================================================================
    with tab_eda:
        st.subheader("Historical IPO Listing-Day Gain Insights (1,060 Listed IPOs)")
        if eda:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Listed IPOs", eda.get("total_ipos", 1060))
            k2.metric("Mainboard / SME", f"{eda.get('mainboard_count', 324)} / {eda.get('sme_count', 736)}")
            k3.metric("Median Typical Gain", f"{eda.get('overall_typical_median_pct', 0):+.2f}%")
            k4.metric("Positive Listing Rate", f"{eda.get('overall_positive_rate_pct', 0):.1f}%")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Segment Summary (Mainboard vs. SME)**")
                st.dataframe(pd.DataFrame(eda.get("segment_stats", [])), width="stretch")
                st.markdown("**Subscription Bucket vs. Listing Gain**")
                st.dataframe(pd.DataFrame(eda.get("subscription_buckets", [])), width="stretch")
            with c2:
                st.markdown("**Sector-Wise Typical Listing Gain (%)**")
                st.dataframe(pd.DataFrame(eda.get("sector_stats", [])), width="stretch")

            st.markdown("**Year-Wise Listing Gain Trend**")
            st.dataframe(pd.DataFrame(eda.get("yearly_stats", [])), width="stretch")

        for img_name, caption in [
            ("eda_target_distribution.png", "Listing-Day Gain Distributions by Segment & Exchange"),
            ("eda_sub_and_gmp.png", "Subscription Demand & GMP vs. Listing-Day Gain"),
            ("eda_sectors_and_years.png", "Sector-Wise Median Gains & Yearly Trends"),
        ]:
            img_path = os.path.join(PLOTS_DIR, img_name)
            if os.path.exists(img_path):
                st.image(img_path, caption=caption, width="stretch")

    # =========================================================================
    # TAB 3: MODEL BENCHMARKS & GMP ABLATION
    # =========================================================================
    with tab_bench:
        st.subheader(
            f"Chronological Holdout Benchmarks (`{metrics['target']}`, n={metrics['holdout_rows']}: {metrics['holdout_date_range'][0]} to {metrics['holdout_date_range'][1]})"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**With GMP Features (Holdout Test Set)**")
            st.dataframe(pd.DataFrame(metrics["with_gmp"]["holdout_metrics"]).T, width="stretch")
            st.caption(f"CV Blend Weights: {metrics['with_gmp']['blend_weights']}")
        with c2:
            st.markdown("**Without GMP Features (Holdout Test Set)**")
            st.dataframe(pd.DataFrame(metrics["without_gmp"]["holdout_metrics"]).T, width="stretch")
            st.caption(f"CV Blend Weights: {metrics['without_gmp']['blend_weights']}")

        comp_img = os.path.join(PLOTS_DIR, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="With GMP vs. Without GMP Holdout MAE Comparison", width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            p = os.path.join(PLOTS_DIR, "shap_summary_with_gmp.png")
            if os.path.exists(p):
                st.image(p, caption="Global SHAP Summary (With GMP)", width="stretch")
        with c2:
            p = os.path.join(PLOTS_DIR, "shap_summary_no_gmp.png")
            if os.path.exists(p):
                st.image(p, caption="Global SHAP Summary (Without GMP)", width="stretch")

    st.markdown("---")
    st.caption(
        "**Disclaimer:** This project was built solely for academic coursework (PCCAIML502 – Introduction to Machine Learning) and educational research using historical IPO data. I am not a SEBI-registered Research Analyst or Investment Adviser. The model outputs are statistical estimates for demonstration purposes only and do not constitute financial advice, investment recommendations, or guaranteed returns."
    )


if __name__ == "__main__":
    main()
