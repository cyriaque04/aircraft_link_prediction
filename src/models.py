"""
Leakage-aware temporal grid search.

The single public entry point is `grid_search_temporal_leakfree`. It
accepts a model factory, a parameter grid, the chronologically sorted
train pairs, and a `fit_and_score_fn` callback that handles the model
family's specific fit/predict idiom. Three callbacks are provided:

  • _fit_score_proba         — sklearn-style binary classifier
  • _fit_score_proba_scaled  — same, with per-fold StandardScaler
  • _fit_score_ranker        — LGBMRanker (group=, raw scores)

Each fold REFITS the historical features on its own training portion
only, preventing fold-validation statistics from contaminating
candidate hyperparameter scores.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from features import apply_historical_features, fit_historical_features


# ────────────────────────────────────────────────────────────────────
# Per-fold scoring helpers
# ────────────────────────────────────────────────────────────────────

def _aucs_and_top1(
    y_true: pd.Series,
    scores: np.ndarray,
    arr_ids: np.ndarray,
) -> tuple[float, float]:
    """Pair-level AUC and link-level Top-1 from one fold's predictions."""
    auc = float(roc_auc_score(y_true, scores))
    fold_df = pd.DataFrame({
        'arr_flight_id': arr_ids,
        'label':         np.asarray(y_true),
        'score':         np.asarray(scores),
    })
    idx_best = fold_df.groupby('arr_flight_id')['score'].idxmax()
    top1 = float(fold_df.loc[idx_best, 'label'].mean())
    return auc, top1


def _fit_score_proba(model_factory, params, tr_X, va_X, feature_cols):
    """Standard binary classifier with predict_proba."""
    Xtr = tr_X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xva = va_X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr, yva = tr_X['label'], va_X['label']
    model = model_factory(**params)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]
    return _aucs_and_top1(yva, p, va_X['arr_flight_id'].values)


def _fit_score_proba_scaled(model_factory, params, tr_X, va_X, feature_cols):
    """Classifier with per-fold StandardScaler (used for LogReg)."""
    Xtr = tr_X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xva = va_X[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr, yva = tr_X['label'], va_X['label']
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    model = model_factory(**params)
    model.fit(Xtr_sc, ytr)
    p = model.predict_proba(Xva_sc)[:, 1]
    return _aucs_and_top1(yva, p, va_X['arr_flight_id'].values)


def _fit_score_ranker(model_factory, params, tr_X, va_X, feature_cols):
    """LGBMRanker: sort by arr_flight_id, pass group=, return raw scores."""
    tr_sorted = tr_X.sort_values('arr_flight_id')
    va_sorted = va_X.sort_values('arr_flight_id')
    Xtr = tr_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xva = va_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    ytr, yva = tr_sorted['label'], va_sorted['label']
    tr_groups = tr_sorted.groupby('arr_flight_id', sort=False).size().values
    model = model_factory(**params)
    model.fit(Xtr, ytr, group=tr_groups)
    p = model.predict(Xva)
    return _aucs_and_top1(yva, p, va_sorted['arr_flight_id'].values)


# ────────────────────────────────────────────────────────────────────
# Main grid search
# ────────────────────────────────────────────────────────────────────

FitScoreFn = Callable[
    [Callable[..., Any], dict[str, Any], pd.DataFrame, pd.DataFrame, list[str]],
    tuple[float, float],
]


def grid_search_temporal_leakfree(
    model_factory: Callable[..., Any],
    param_grid: dict[str, list[Any]],
    train_pairs_sorted: pd.DataFrame,
    train_mc: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 3,
    name: str = "model",
    fit_and_score_fn: FitScoreFn = _fit_score_proba,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Manual grid search with TimeSeriesSplit and per-fold feature refit.

    Parameters
    ----------
    model_factory      callable: factory(**params) → estimator
    param_grid         dict of hyperparameter name → list of values
    train_pairs_sorted within-row-feature pairs, sorted by arr_datetime
    train_mc           train slice of true (arrival, departure) links;
                       used to fit fold-level historical features
    feature_cols       final feature list (safe + historical)
    n_splits           TimeSeriesSplit folds (default 3)
    name               pretty name for logs
    fit_and_score_fn   one of _fit_score_proba / _proba_scaled / _ranker

    Returns
    -------
    best_params  dict (Python types preserved from the grid input)
    results_df   one row per config, sorted by mean_Top1 desc
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    combos = list(ParameterGrid(param_grid))
    n      = len(combos)
    print(f"\n  Grid: {n} configs × {n_splits} folds = {n*n_splits} fits "
          f"(features refit per fold)")

    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = (-np.inf, -np.inf)

    folds = list(tscv.split(train_pairs_sorted))

    for i, params in enumerate(combos, 1):
        fold_auc, fold_top1 = [], []
        for tr_idx, va_idx in folds:
            fold_tr = train_pairs_sorted.iloc[tr_idx]
            fold_va = train_pairs_sorted.iloc[va_idx]

            fold_va_min_date = fold_va['arr_datetime'].min().date()
            fold_train_mc = train_mc[
                train_mc['arr_datetime'].dt.date < fold_va_min_date
            ]
            if len(fold_train_mc) == 0:
                continue

            T_fold = fit_historical_features(fold_train_mc, fold_tr)
            fold_tr_X = apply_historical_features(fold_tr, T_fold)
            fold_va_X = apply_historical_features(fold_va, T_fold)

            auc, top1 = fit_and_score_fn(
                model_factory, params, fold_tr_X, fold_va_X, feature_cols
            )
            fold_auc.append(auc)
            fold_top1.append(top1)

        if not fold_auc:
            print(f"   [{i:>2d}/{n}] {params}  →  SKIPPED")
            continue

        mean_auc  = float(np.mean(fold_auc))
        mean_top1 = float(np.mean(fold_top1))
        std_top1  = float(np.std(fold_top1))
        rows.append({
            **params,
            'mean_AUC':  round(mean_auc,  4),
            'mean_Top1': round(mean_top1, 4),
            'std_Top1':  round(std_top1,  4),
        })

        score = (mean_top1, mean_auc)
        if score > best_score:
            best_score  = score
            best_params = dict(params)

        print(f"   [{i:>2d}/{n}] {params}  →  "
              f"Top1 = {mean_top1:.4f} (±{std_top1:.4f})  AUC = {mean_auc:.4f}")

    res_df = pd.DataFrame(rows).sort_values(
        ['mean_Top1', 'mean_AUC'], ascending=False
    ).reset_index(drop=True)

    print(f"\n  Best for {name}: {best_params}")
    print(f"  → mean_Top1 = {best_score[0]:.4f}   mean_AUC = {best_score[1]:.4f}")
    return best_params, res_df
