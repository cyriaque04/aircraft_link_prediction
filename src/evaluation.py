"""
Evaluation metrics and stability diagnostics.

Provides:
  • Pair-level classification metrics (AUC, P, R, F1)
  • Link-level operational metrics (Top-1, Top-3 per arrival)
  • Population Stability Index (PSI) for train-vs-val drift checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score
)


# ════════════════════════════════════════════════════════════════════
# Classification & ranking metrics
# ════════════════════════════════════════════════════════════════════

def pair_level_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """AUC, Precision, Recall, F1 on the per-pair binary task."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        'AUC':       float(roc_auc_score(y_true, y_prob)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'F1':        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def link_level_accuracy(
    val_ids_df: pd.DataFrame,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Top-1 and Top-3 accuracy at the arrival level.

    val_ids_df must contain columns: arr_flight_id, label.
    y_prob is aligned with val_ids_df rows. Per arrival we pick the
    candidate with the highest score and check whether its label is 1.
    """
    tmp = val_ids_df.copy()
    tmp['score'] = np.asarray(y_prob)

    idx_best = tmp.groupby('arr_flight_id')['score'].idxmax()
    top1 = float(tmp.loc[idx_best, 'label'].mean())

    def _in_topk(group: pd.DataFrame, k: int = 3) -> int:
        return int(group.nlargest(k, 'score')['label'].sum() > 0)

    top3 = float(tmp.groupby('arr_flight_id').apply(lambda g: _in_topk(g, 3)).mean())
    return {'Top1_Acc': top1, 'Top3_Acc': top3}


def evaluate_model(
    name: str,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    val_ids_df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, float]:
    """Combine pair-level and link-level metrics; optionally print."""
    metrics = {**pair_level_metrics(y_true, y_prob),
               **link_level_accuracy(val_ids_df, y_prob)}
    if verbose:
        print(f"\n  [{name}]")
        for k, v in metrics.items():
            print(f"    {k:<12s}: {v:.4f}")
    return metrics


# ════════════════════════════════════════════════════════════════════
# Population Stability Index (PSI)
# ════════════════════════════════════════════════════════════════════
#  PSI = Σ_i (p_actual_i − p_expected_i) · ln(p_actual_i / p_expected_i)
#  Conventional thresholds:
#      PSI < 0.10  ─ no significant shift
#      0.10–0.25   ─ moderate shift, monitor
#      ≥ 0.25      ─ significant shift, model degradation likely
# ════════════════════════════════════════════════════════════════════

def compute_psi(
    expected: np.ndarray | pd.Series,
    actual:   np.ndarray | pd.Series,
    n_buckets: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    PSI between two 1-D distributions.

    Quantile-bucketing is performed on `expected`. Buckets with zero
    mass in either distribution are floored at `eps` to keep the log
    finite. Constant or empty inputs return 0.0.
    """
    e = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.Series(actual  ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) == 0 or len(a) == 0:
        return float('nan')
    if e.nunique() <= 1:
        return 0.0

    qs    = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(e, qs))
    if len(edges) < 3:
        return 0.0
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))

    e_counts = np.histogram(e, bins=edges)[0]
    a_counts = np.histogram(a, bins=edges)[0]
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    e_pct = np.where(e_pct == 0, eps, e_pct)
    a_pct = np.where(a_pct == 0, eps, a_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_report(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    cols:     list[str],
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Per-feature PSI report between train and val, sorted by drift desc.
    Returns columns: feature, PSI, severity ∈ {OK, MODERATE, SIGNIFICANT, N/A}.
    """
    rows = []
    for c in cols:
        if c not in train_df.columns or c not in val_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            continue
        rows.append({'feature': c, 'PSI': compute_psi(train_df[c], val_df[c], n_buckets)})

    df = pd.DataFrame(rows).sort_values('PSI', ascending=False).reset_index(drop=True)

    def _label(p: float) -> str:
        if pd.isna(p):  return 'N/A'
        if p < 0.10:    return 'OK'
        if p < 0.25:    return 'MODERATE'
        return 'SIGNIFICANT'

    df['severity'] = df['PSI'].apply(_label)
    df['PSI']      = df['PSI'].round(4)
    return df
