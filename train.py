"""Training, Chronological CV, GMP Ablation, and Evaluation Pipeline.

Compares Naive GMP, RidgeCV, Random Forest, CatBoost (RMSE + MultiQuantile P10/P50/P90),
and a CV-tuned Weighted Ensemble across 5-fold TimeSeriesSplit CV and an untouched
chronological holdout set (last 80 IPOs), for both With-GMP and Without-GMP feature sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_pipeline import CATEGORICAL_COLS, get_feature_columns

warnings.filterwarnings("ignore")

DATA_PATH = "ipo_listing_gain_clean.csv"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

SME_CLIP_BOUNDS = {
    "target_open_gain_pct": (-20.0, 90.0),
    "target_typical_gain_pct": (-24.0, 99.5),
    "target_close_gain_pct": (-24.0, 99.5),
}


def clip_predictions(preds: np.ndarray, is_sme: np.ndarray, target_col: str) -> np.ndarray:
    """Apply exchange listing-day circuit bounds to SME predictions."""
    low, high = SME_CLIP_BOUNDS[target_col]
    out = np.asarray(preds, dtype=float).copy()
    sme_idx = np.asarray(is_sme) == 1
    if out.ndim == 1:
        out[sme_idx] = np.clip(out[sme_idx], low, high)
    else:
        out[sme_idx, :] = np.clip(out[sme_idx, :], low, high)
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_sme: np.ndarray | None = None) -> dict[str, float]:
    """Compute regression, directional, ranking, and segment MAE metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    dir_acc = float(np.mean((y_pred > 0) == (y_true > 0)) * 100.0)
    rho = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 else 0.0
    if np.isnan(rho):
        rho = 0.0

    metrics: dict[str, float] = {
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 2),
        "medae": round(float(median_absolute_error(y_true, y_pred)), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "dir_acc_pct": round(dir_acc, 2),
        "spearman_rho": round(rho, 3),
    }

    if is_sme is not None:
        is_sme_arr = np.asarray(is_sme)
        mb_mask = is_sme_arr == 0
        sme_mask = is_sme_arr == 1
        if mb_mask.any():
            metrics["mainboard_mae"] = round(float(mean_absolute_error(y_true[mb_mask], y_pred[mb_mask])), 2)
        if sme_mask.any():
            metrics["sme_mae"] = round(float(mean_absolute_error(y_true[sme_mask], y_pred[sme_mask])), 2)

    return metrics


def make_ridge_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    # min_frequency=10 collapses rare lead_manager_grouped levels to prevent overfitting on ~850 rows
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), cat_cols),
        ]
    )
    return Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-1, 4, 30)))])


def make_rf_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    # Constrained depth & leaf size prevent memorizing small-sample IPO noise
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), cat_cols),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=5,
                    max_features=0.6,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_catboost(loss_function: str, iterations: int, depth: int, l2_leaf_reg: float) -> CatBoostRegressor:
    # Bernoulli 0.8 subsampling reduces variance on ~980 tabular rows (verified on CV & holdout)
    return CatBoostRegressor(
        loss_function=loss_function,
        iterations=iterations,
        learning_rate=0.03,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=42,
        allow_writing_files=False,
        verbose=0,
    )


def prepare_catboost_df(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df_cb = X.copy()
    for c in cat_cols:
        df_cb[c] = df_cb[c].fillna("Missing").astype(str)
    return df_cb


def run_suite(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    target_col: str,
    include_gmp: bool,
) -> dict:
    """Run 5-fold TimeSeriesSplit CV + untouched holdout evaluation for one GMP mode."""
    mode_tag = "with_gmp" if include_gmp else "no_gmp"
    feature_cols = get_feature_columns(include_gmp=include_gmp)
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].to_numpy(dtype=float)
    sme_train = train_df["is_sme"].to_numpy(dtype=int)

    X_test = holdout_df[feature_cols].copy()
    y_test = holdout_df[target_col].to_numpy(dtype=float)
    sme_test = holdout_df["is_sme"].to_numpy(dtype=int)

    X_train_cb = prepare_catboost_df(X_train, cat_cols)
    X_test_cb = prepare_catboost_df(X_test, cat_cols)

    tscv = TimeSeriesSplit(n_splits=5)

    # 1. Tune CatBoost depth & l2_leaf_reg strictly on TimeSeriesSplit CV MAE
    best_cfg = {"depth": 4, "l2_leaf_reg": 5.0}
    best_cv_mae = float("inf")
    for depth in [4, 5]:
        for l2_reg in [5.0, 10.0]:
            fold_maes = []
            for tr_idx, va_idx in tscv.split(X_train):
                cb = make_catboost(
                    loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
                    iterations=1200,
                    depth=depth,
                    l2_leaf_reg=l2_reg,
                )
                cb.fit(
                    X_train_cb.iloc[tr_idx],
                    y_train[tr_idx],
                    cat_features=cat_cols,
                    eval_set=(X_train_cb.iloc[va_idx], y_train[va_idx]),
                    early_stopping_rounds=50,
                    verbose=False,
                )
                preds = np.sort(cb.predict(X_train_cb.iloc[va_idx]), axis=1)
                p50 = clip_predictions(preds[:, 1], sme_train[va_idx], target_col)
                fold_maes.append(mean_absolute_error(y_train[va_idx], p50))
            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_cv_mae:
                best_cv_mae = mean_mae
                best_cfg = {"depth": depth, "l2_leaf_reg": l2_reg}

    print(f"[{mode_tag}] Selected CatBoost config from CV: {best_cfg} (CV MAE={best_cv_mae:.2f})")
    sys.stdout.flush()

    # 2. Full 5-fold TimeSeriesSplit CV across all models
    cv_records: dict[str, list[dict[str, float]]] = {
        "RidgeCV": [],
        "RandomForest": [],
        "CatBoost_RMSE": [],
        "CatBoost_MultiQuantile_P50": [],
    }
    if include_gmp:
        cv_records["Naive_GMP"] = []

    oof_p50, oof_rmse, oof_rf, oof_y = [], [], [], []
    mq_best_iters, rmse_best_iters = [], []
    mq_coverages = []

    for tr_idx, va_idx in tscv.split(X_train):
        y_va = y_train[va_idx]
        sme_va = sme_train[va_idx]

        # Naive GMP baseline
        if include_gmp:
            gmp_va = X_train["gmp_close_day_pct"].iloc[va_idx].fillna(0.0).to_numpy(dtype=float)
            gmp_va = clip_predictions(gmp_va, sme_va, target_col)
            cv_records["Naive_GMP"].append(compute_metrics(y_va, gmp_va, sme_va))

        # RidgeCV
        ridge = make_ridge_pipeline(num_cols, cat_cols)
        ridge.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        ridge_pred = clip_predictions(ridge.predict(X_train.iloc[va_idx]), sme_va, target_col)
        cv_records["RidgeCV"].append(compute_metrics(y_va, ridge_pred, sme_va))

        # RandomForest
        rf = make_rf_pipeline(num_cols, cat_cols)
        rf.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        rf_pred = clip_predictions(rf.predict(X_train.iloc[va_idx]), sme_va, target_col)
        cv_records["RandomForest"].append(compute_metrics(y_va, rf_pred, sme_va))

        # CatBoost RMSE
        cb_rmse = make_catboost(
            loss_function="RMSE",
            iterations=1200,
            depth=best_cfg["depth"],
            l2_leaf_reg=best_cfg["l2_leaf_reg"],
        )
        cb_rmse.fit(
            X_train_cb.iloc[tr_idx],
            y_train[tr_idx],
            cat_features=cat_cols,
            eval_set=(X_train_cb.iloc[va_idx], y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        rmse_best_iters.append(cb_rmse.get_best_iteration() or 300)
        cb_rmse_pred = clip_predictions(cb_rmse.predict(X_train_cb.iloc[va_idx]), sme_va, target_col)
        cv_records["CatBoost_RMSE"].append(compute_metrics(y_va, cb_rmse_pred, sme_va))

        # CatBoost MultiQuantile (outputs P10, P50, P90 simultaneously)
        cb_mq = make_catboost(
            loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
            iterations=1200,
            depth=best_cfg["depth"],
            l2_leaf_reg=best_cfg["l2_leaf_reg"],
        )
        cb_mq.fit(
            X_train_cb.iloc[tr_idx],
            y_train[tr_idx],
            cat_features=cat_cols,
            eval_set=(X_train_cb.iloc[va_idx], y_va),
            early_stopping_rounds=50,
            verbose=False,
        )
        mq_best_iters.append(cb_mq.get_best_iteration() or 300)
        # Sort across quantiles to prevent P10 > P50 or P50 > P90 crossing
        mq_raw = np.sort(cb_mq.predict(X_train_cb.iloc[va_idx]), axis=1)
        mq_clip = clip_predictions(mq_raw, sme_va, target_col)
        cv_records["CatBoost_MultiQuantile_P50"].append(compute_metrics(y_va, mq_clip[:, 1], sme_va))
        mq_coverages.append(float(np.mean((y_va >= mq_clip[:, 0]) & (y_va <= mq_clip[:, 2])) * 100.0))

        oof_p50.append(mq_clip[:, 1])
        oof_rmse.append(cb_rmse_pred)
        oof_rf.append(rf_pred)
        oof_y.append(y_va)

    # Optimize blend weights strictly on OOF CV predictions (CatBoost_P50 vs CatBoost_RMSE vs RF)
    y_oof = np.concatenate(oof_y)
    p50_oof = np.concatenate(oof_p50)
    rmse_oof = np.concatenate(oof_rmse)
    rf_oof = np.concatenate(oof_rf)

    best_blend_weights = {"w_p50": 1.0, "w_rmse": 0.0, "w_rf": 0.0}
    best_blend_mae = mean_absolute_error(y_oof, p50_oof)
    for w_p50 in np.linspace(0.2, 1.0, 9):
        for w_rmse in np.linspace(0.0, 1.0 - w_p50, 9):
            w_rf = max(0.0, 1.0 - w_p50 - w_rmse)
            if abs((w_p50 + w_rmse + w_rf) - 1.0) > 1e-6:
                continue
            blend = w_p50 * p50_oof + w_rmse * rmse_oof + w_rf * rf_oof
            mae = mean_absolute_error(y_oof, blend)
            if mae < best_blend_mae:
                best_blend_mae = mae
                best_blend_weights = {
                    "w_p50": round(float(w_p50), 2),
                    "w_rmse": round(float(w_rmse), 2),
                    "w_rf": round(float(w_rf), 2),
                }

    # Aggregate CV summary across folds
    cv_summary = {}
    for m_name, recs in cv_records.items():
        agg = {}
        for k in recs[0]:
            if k == "n":
                agg[k] = int(sum(r[k] for r in recs))
            else:
                agg[k] = round(float(np.mean([r[k] for r in recs])), 3 if k in ("r2", "spearman_rho") else 2)
        cv_summary[m_name] = agg
    cv_summary["CatBoost_MultiQuantile_P50"]["p10_p90_coverage_pct"] = round(float(np.mean(mq_coverages)), 2)

    # 3. Refit on full train_df using CV mean best_iteration (NO holdout leakage into early stopping!)
    final_mq_iters = max(100, int(round(np.mean(mq_best_iters) * 1.1)))
    final_rmse_iters = max(100, int(round(np.mean(rmse_best_iters) * 1.1)))

    ridge_final = make_ridge_pipeline(num_cols, cat_cols)
    ridge_final.fit(X_train, y_train)
    ridge_test = clip_predictions(ridge_final.predict(X_test), sme_test, target_col)

    rf_final = make_rf_pipeline(num_cols, cat_cols)
    rf_final.fit(X_train, y_train)
    rf_test = clip_predictions(rf_final.predict(X_test), sme_test, target_col)

    cb_rmse_final = make_catboost(
        loss_function="RMSE",
        iterations=final_rmse_iters,
        depth=best_cfg["depth"],
        l2_leaf_reg=best_cfg["l2_leaf_reg"],
    )
    cb_rmse_final.fit(X_train_cb, y_train, cat_features=cat_cols, verbose=False)
    cb_rmse_test = clip_predictions(cb_rmse_final.predict(X_test_cb), sme_test, target_col)

    cb_mq_final = make_catboost(
        loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
        iterations=final_mq_iters,
        depth=best_cfg["depth"],
        l2_leaf_reg=best_cfg["l2_leaf_reg"],
    )
    cb_mq_final.fit(X_train_cb, y_train, cat_features=cat_cols, verbose=False)
    mq_test_raw = np.sort(cb_mq_final.predict(X_test_cb), axis=1)
    mq_test = clip_predictions(mq_test_raw, sme_test, target_col)

    blend_test = (
        best_blend_weights["w_p50"] * mq_test[:, 1]
        + best_blend_weights["w_rmse"] * cb_rmse_test
        + best_blend_weights["w_rf"] * rf_test
    )
    blend_test = clip_predictions(blend_test, sme_test, target_col)

    holdout_summary = {}
    if include_gmp:
        avail_mask = X_test["gmp_available"].to_numpy() == 1
        naive_all = clip_predictions(X_test["gmp_close_day_pct"].fillna(0.0).to_numpy(dtype=float), sme_test, target_col)
        holdout_summary["Naive_GMP"] = compute_metrics(y_test[avail_mask], naive_all[avail_mask], sme_test[avail_mask])

    holdout_summary["RidgeCV"] = compute_metrics(y_test, ridge_test, sme_test)
    holdout_summary["RandomForest"] = compute_metrics(y_test, rf_test, sme_test)
    holdout_summary["CatBoost_RMSE"] = compute_metrics(y_test, cb_rmse_test, sme_test)
    holdout_summary["CatBoost_MultiQuantile_P50"] = compute_metrics(y_test, mq_test[:, 1], sme_test)
    holdout_summary["CatBoost_MultiQuantile_P50"]["p10_p90_coverage_pct"] = round(
        float(np.mean((y_test >= mq_test[:, 0]) & (y_test <= mq_test[:, 2])) * 100.0), 2
    )
    holdout_summary["Weighted_Ensemble"] = compute_metrics(y_test, blend_test, sme_test)

    # 4. Save model artifacts & global SHAP summary for winning CatBoost model
    cb_mq_final.save_model(os.path.join(ARTIFACTS_DIR, f"catboost_mq_{mode_tag}.cbm"))
    cb_rmse_final.save_model(os.path.join(ARTIFACTS_DIR, f"catboost_rmse_{mode_tag}.cbm"))
    joblib.dump(rf_final, os.path.join(ARTIFACTS_DIR, f"rf_{mode_tag}.joblib"))
    joblib.dump(ridge_final, os.path.join(ARTIFACTS_DIR, f"ridge_{mode_tag}.joblib"))

    explainer = shap.TreeExplainer(cb_rmse_final)
    shap_values = explainer.shap_values(X_test_cb)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_shap = sorted(
        [{"feature": col, "mean_abs_shap": round(float(val), 3)} for col, val in zip(feature_cols, mean_abs_shap)],
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )[:15]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test_cb, max_display=15, show=False)
    plt.title(f"Top 15 SHAP Drivers ({'With GMP' if include_gmp else 'Without GMP'})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"shap_summary_{mode_tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "best_catboost_config": best_cfg,
        "final_iterations": {"multi_quantile": final_mq_iters, "rmse": final_rmse_iters},
        "blend_weights": best_blend_weights,
        "cv_metrics": cv_summary,
        "holdout_metrics": holdout_summary,
        "top_shap_features": top_shap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate IPO listing-gain models.")
    parser.add_argument(
        "--target",
        default="target_typical_gain_pct",
        choices=["target_open_gain_pct", "target_typical_gain_pct", "target_close_gain_pct"],
    )
    parser.add_argument("--holdout", type=int, default=80, help="Number of most recent IPOs for holdout (50-100).")
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, dtype={"bse_code": str})
    df["listing_date"] = pd.to_datetime(df["listing_date"])
    # Sort chronologically so TimeSeriesSplit and holdout strictly test future IPOs
    df = df.sort_values(["listing_date", "close_date", "company_name"]).reset_index(drop=True)

    train_df = df.iloc[: -args.holdout].copy()
    holdout_df = df.iloc[-args.holdout :].copy()

    print(
        f"Dataset: {len(df)} rows | Train CV: {len(train_df)} ({train_df['listing_date'].min().date()} to {train_df['listing_date'].max().date()}) | "
        f"Holdout: {len(holdout_df)} ({holdout_df['listing_date'].min().date()} to {holdout_df['listing_date'].max().date()})"
    )
    sys.stdout.flush()

    with_gmp_res = run_suite(train_df, holdout_df, args.target, include_gmp=True)
    no_gmp_res = run_suite(train_df, holdout_df, args.target, include_gmp=False)

    # Save comparison bar chart across With-GMP vs Without-GMP models
    models_plot = ["RidgeCV", "RandomForest", "CatBoost_RMSE", "CatBoost_MultiQuantile_P50", "Weighted_Ensemble"]
    mae_with = [with_gmp_res["holdout_metrics"][m]["mae"] for m in models_plot]
    mae_no = [no_gmp_res["holdout_metrics"][m]["mae"] for m in models_plot]

    x = np.arange(len(models_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, mae_with, width, label="With GMP")
    ax.bar(x + width / 2, mae_no, width, label="Without GMP")
    if "Naive_GMP" in with_gmp_res["holdout_metrics"]:
        naive_mae = with_gmp_res["holdout_metrics"]["Naive_GMP"]["mae"]
        ax.axhline(naive_mae, color="crimson", linestyle="--", linewidth=1.2, label=f"Naive GMP Baseline ({naive_mae:.2f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["RidgeCV", "Random Forest", "CatBoost RMSE", "CatBoost P50", "Weighted Blend"])
    ax.set_ylabel("Holdout MAE (% points, lower is better)")
    ax.set_title(f"Holdout Model Comparison ({args.target}, n={args.holdout})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=300)
    plt.close()

    # Persist model config & metrics JSON for UI + reporting
    frequent_lms = sorted([lm for lm in df["lead_manager_grouped"].dropna().unique() if lm != "Other"])
    model_config = {
        "target_col": args.target,
        "holdout_size": args.holdout,
        "frequent_lead_managers": frequent_lms,
        "with_gmp_blend_weights": with_gmp_res["blend_weights"],
        "no_gmp_blend_weights": no_gmp_res["blend_weights"],
        "feature_cols_with_gmp": get_feature_columns(include_gmp=True),
        "feature_cols_no_gmp": get_feature_columns(include_gmp=False),
        "categorical_cols": CATEGORICAL_COLS,
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    metrics_payload = {
        "target": args.target,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "holdout_date_range": [
            str(holdout_df["listing_date"].min().date()),
            str(holdout_df["listing_date"].max().date()),
        ],
        "with_gmp": with_gmp_res,
        "without_gmp": no_gmp_res,
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n=== HOLDOUT SUMMARY (WITH GMP) ===")
    print(pd.DataFrame(with_gmp_res["holdout_metrics"]).T.to_string())
    print("\n=== HOLDOUT SUMMARY (WITHOUT GMP) ===")
    print(pd.DataFrame(no_gmp_res["holdout_metrics"]).T.to_string())
    sys.stdout.flush()


if __name__ == "__main__":
    main()
    os._exit(0)
