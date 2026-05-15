"""
Train baselines + grid-tuned models, score on val, persist artefacts.

Steps
-----
1. Load raw pairs + merged_clean + split date.
2. Fit historical features on train slice; apply to train & val.
3. Compute PSI on the full feature set (train vs val).
4. Train five baselines: LogReg, RF, XGBoost, LightGBM Clf, LightGBM Ranker.
5. (Optional) Grid search all five with temporal CV + per-fold feature refit.
6. Save trained models, transforms, scaler, feature list, comparison CSV.

Run from `src/`:
    python train.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from features import (
    apply_historical_features,
    fit_historical_features,
)
from evaluation import (
    evaluate_model,
    link_level_accuracy,
    pair_level_metrics,
    psi_report,
)
from models import (
    _fit_score_proba_scaled,
    _fit_score_ranker,
    grid_search_temporal_leakfree,
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

ROOT      = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS   = ROOT / "outputs"
MODELS    = ROOT / "models"

RUN_GRID_SEARCH = True
GRID_CV_SPLITS  = 3
GRID_CV_NEST    = 300
GRID_REFIT_NEST = 1500

# Optional GBM imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost not installed — skipping XGBoost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ lightgbm not installed — skipping LightGBM")


# ════════════════════════════════════════════════════════════════════
# 1. LOAD
# ════════════════════════════════════════════════════════════════════

print("Loading data...")
pairs        = pd.read_parquet(DATA_PROC / "candidate_pairs_raw.parquet")
merged_clean = pd.read_parquet(DATA_PROC / "merged_clean.parquet")
with open(DATA_PROC / "safe_feature_cols.json") as f:
    safe_feature_cols = json.load(f)
with open(DATA_PROC / "split_date.txt") as f:
    SPLIT_DATE = pd.to_datetime(f.read().strip()).date()

pairs['arr_datetime'] = pd.to_datetime(pairs['arr_datetime'])
pairs['dep_datetime'] = pd.to_datetime(pairs['dep_datetime'])
merged_clean['arr_datetime'] = pd.to_datetime(merged_clean['arr_datetime'])
merged_clean['dep_datetime'] = pd.to_datetime(merged_clean['dep_datetime'])

print(f"  Rows: {len(pairs):,}   Split: {SPLIT_DATE}")


# ════════════════════════════════════════════════════════════════════
# 2. TEMPORAL SPLIT + FIT/TRANSFORM HISTORICAL FEATURES
# ════════════════════════════════════════════════════════════════════

pairs['arr_date'] = pairs['arr_datetime'].dt.date
train_mask = pairs['arr_date'] < SPLIT_DATE
val_mask   = pairs['arr_date'] >= SPLIT_DATE

train_pairs = pairs.loc[train_mask].copy()
val_pairs   = pairs.loc[val_mask].copy()
train_mc    = merged_clean[merged_clean['arr_datetime'].dt.date < SPLIT_DATE].copy()

print(f"\n  Train pairs: {len(train_pairs):,}   "
      f"Val pairs: {len(val_pairs):,}")

print("\nFitting historical features on train slice...")
transforms  = fit_historical_features(train_mc, train_pairs)
train_pairs = apply_historical_features(train_pairs, transforms)
val_pairs   = apply_historical_features(val_pairs,   transforms)

id_cols  = ['arr_flight_id', 'dep_flight_id', 'arr_date']
raw_cats = [
    'arr_datetime', 'dep_datetime',
    'arr_pax_cargo', 'dep_pax_cargo',
    'arr_origin_dest', 'dep_origin_dest',
    'arr_airline', 'dep_airline',
    'arr_aircraft_type', 'dep_aircraft_type',
]
historical_cols = [
    c for c in train_pairs.columns
    if c not in id_cols + raw_cats + ['label']
    and c not in safe_feature_cols
    and pd.api.types.is_numeric_dtype(train_pairs[c])
]
feature_cols = safe_feature_cols + historical_cols
print(f"  Total features: {len(feature_cols)}  "
      f"(safe: {len(safe_feature_cols)}, historical: {len(historical_cols)})")


# ════════════════════════════════════════════════════════════════════
# 3. PSI AFTER FE
# ════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print(" PSI after feature engineering (train vs val)")
print("="*65)

psi_after = psi_report(train_pairs, val_pairs, feature_cols)
print(f"  All: OK={(psi_after['severity']=='OK').sum()}  "
      f"MOD={(psi_after['severity']=='MODERATE').sum()}  "
      f"SIG={(psi_after['severity']=='SIGNIFICANT').sum()}")

psi_safe = psi_after[psi_after['feature'].isin(safe_feature_cols)]
psi_hist = psi_after[psi_after['feature'].isin(historical_cols)]
print(f"  Within-row: OK={(psi_safe['severity']=='OK').sum()}  "
      f"MOD={(psi_safe['severity']=='MODERATE').sum()}  "
      f"SIG={(psi_safe['severity']=='SIGNIFICANT').sum()}")
print(f"  Historical: OK={(psi_hist['severity']=='OK').sum()}  "
      f"MOD={(psi_hist['severity']=='MODERATE').sum()}  "
      f"SIG={(psi_hist['severity']=='SIGNIFICANT').sum()}")

print(f"\n  Top 10 by drift:")
for _, r in psi_after.head(10).iterrows():
    print(f"    {r['feature']:<28s}  PSI={r['PSI']:.4f}  [{r['severity']}]")

OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
psi_after.to_csv(OUTPUTS / "psi_after_fe.csv", index=False)


# ════════════════════════════════════════════════════════════════════
# 4. ASSEMBLE X / y
# ════════════════════════════════════════════════════════════════════

X_train = train_pairs[feature_cols].copy()
y_train = train_pairs['label'].copy()
X_val   = val_pairs[feature_cols].copy()
y_val   = val_pairs['label'].copy()
val_ids = val_pairs[['arr_flight_id', 'dep_flight_id', 'label']].copy()

for X in [X_train, X_val]:
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)


# ════════════════════════════════════════════════════════════════════
# 5. BASELINES
# ════════════════════════════════════════════════════════════════════

results: dict[str, dict[str, float]] = {}
trained_models: dict[str, object] = {}

# ── LogReg ──────────────────────────────────────────────────────
print("\n" + "="*65, "\n Training Logistic Regression...\n", "="*65, sep="")
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
t0 = time.time()
lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                        C=1.0, solver='lbfgs', n_jobs=-1)
lr.fit(X_train_sc, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")
results['LogReg'] = evaluate_model('LogReg', y_val, lr.predict_proba(X_val_sc)[:, 1], val_ids)
trained_models['LogReg'] = lr

# ── Random Forest ───────────────────────────────────────────────
print("\n" + "="*65, "\n Training Random Forest...\n", "="*65, sep="")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                            class_weight='balanced_subsample', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")
results['RF'] = evaluate_model('RF', y_val, rf.predict_proba(X_val)[:, 1], val_ids)
trained_models['RF'] = rf

# ── XGBoost ─────────────────────────────────────────────────────
if HAS_XGB:
    print("\n" + "="*65, "\n Training XGBoost...\n", "="*65, sep="")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        scale_pos_weight=neg_pos_ratio, eval_metric='auc',
        early_stopping_rounds=30, random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Trained in {time.time()-t0:.1f}s  (best iter: {xgb_model.best_iteration})")
    results['XGBoost'] = evaluate_model(
        'XGBoost', y_val, xgb_model.predict_proba(X_val)[:, 1], val_ids
    )
    trained_models['XGBoost'] = xgb_model

# ── LightGBM Classifier ────────────────────────────────────────
if HAS_LGB:
    print("\n" + "="*65, "\n Training LightGBM Classifier...\n", "="*65, sep="")
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        scale_pos_weight=neg_pos_ratio, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_clf.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    print(f"  Trained in {time.time()-t0:.1f}s  (best iter: {lgb_clf.best_iteration_})")
    results['LGBM_Clf'] = evaluate_model(
        'LGBM_Clf', y_val, lgb_clf.predict_proba(X_val)[:, 1], val_ids
    )
    trained_models['LGBM_Clf'] = lgb_clf

# ── LightGBM Ranker ─────────────────────────────────────────────
if HAS_LGB:
    print("\n" + "="*65, "\n Training LightGBM LambdaMART (Ranker)...\n", "="*65, sep="")
    train_order = train_pairs.sort_values('arr_flight_id').index
    val_order   = val_pairs.sort_values('arr_flight_id').index

    X_train_rank = X_train.loc[train_order]
    y_train_rank = y_train.loc[train_order]
    X_val_rank   = X_val.loc[val_order]
    y_val_rank   = y_val.loc[val_order]

    train_groups = train_pairs.loc[train_order].groupby('arr_flight_id', sort=False).size().values
    val_groups   = val_pairs.loc[val_order].groupby('arr_flight_id', sort=False).size().values
    val_ids_rank = val_pairs.loc[val_order, ['arr_flight_id', 'dep_flight_id', 'label']].copy()

    t0 = time.time()
    lgb_rank = lgb.LGBMRanker(
        objective='lambdarank', n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_rank.fit(
        X_train_rank, y_train_rank, group=train_groups,
        eval_set=[(X_val_rank, y_val_rank)], eval_group=[val_groups],
        eval_metric='ndcg', eval_at=[1, 3],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    print(f"  Trained in {time.time()-t0:.1f}s  (best iter: {lgb_rank.best_iteration_})")
    rank_scores = lgb_rank.predict(X_val_rank)
    results['LGBM_Rank'] = {
        **pair_level_metrics(y_val_rank, rank_scores),
        **link_level_accuracy(val_ids_rank, rank_scores),
    }
    print(f"\n  [LGBM_Rank]")
    for k, v in results['LGBM_Rank'].items():
        print(f"    {k:<12s}: {v:.4f}")
    trained_models['LGBM_Rank'] = lgb_rank


# ════════════════════════════════════════════════════════════════════
# 6. GRID SEARCH (all five models, leak-free)
# ════════════════════════════════════════════════════════════════════

if RUN_GRID_SEARCH:
    print("\n" + "="*65, "\n Grid Search (TimeSeriesSplit, per-fold feature refit)\n", "="*65, sep="")

    raw_train_pairs    = pairs.loc[train_mask].copy()
    train_pairs_sorted = raw_train_pairs.sort_values('arr_datetime').reset_index(drop=True)

    # LogReg
    print("\n" + "-"*65, "\n LogReg grid\n", "-"*65, sep="")
    best_lr, lr_grid_df = grid_search_temporal_leakfree(
        lambda **p: LogisticRegression(max_iter=1000, class_weight='balanced',
                                        solver='lbfgs', n_jobs=-1, **p),
        {'C': [0.01, 0.1, 1.0, 10.0]},
        train_pairs_sorted, train_mc, feature_cols,
        n_splits=GRID_CV_SPLITS, name='LogReg',
        fit_and_score_fn=_fit_score_proba_scaled,
    )
    lr_grid_df.to_csv(OUTPUTS / "grid_search_lr.csv", index=False)
    lr_tuned = LogisticRegression(max_iter=1000, class_weight='balanced',
                                  solver='lbfgs', n_jobs=-1, **best_lr)
    lr_tuned.fit(X_train_sc, y_train)
    results['LogReg_tuned'] = evaluate_model(
        'LogReg_tuned', y_val, lr_tuned.predict_proba(X_val_sc)[:, 1], val_ids
    )
    trained_models['LogReg_tuned'] = lr_tuned

    # Random Forest
    print("\n" + "-"*65, "\n RF grid\n", "-"*65, sep="")
    best_rf, rf_grid_df = grid_search_temporal_leakfree(
        lambda **p: RandomForestClassifier(n_estimators=300,
                                            class_weight='balanced_subsample',
                                            random_state=42, n_jobs=-1, **p),
        {'max_depth': [10, 14], 'min_samples_leaf': [10, 20, 30]},
        train_pairs_sorted, train_mc, feature_cols,
        n_splits=GRID_CV_SPLITS, name='RF',
    )
    rf_grid_df.to_csv(OUTPUTS / "grid_search_rf.csv", index=False)
    rf_tuned = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                       random_state=42, n_jobs=-1, **best_rf)
    rf_tuned.fit(X_train, y_train)
    results['RF_tuned'] = evaluate_model(
        'RF_tuned', y_val, rf_tuned.predict_proba(X_val)[:, 1], val_ids
    )
    trained_models['RF_tuned'] = rf_tuned

    # XGBoost
    if HAS_XGB:
        print("\n" + "-"*65, "\n XGBoost grid\n", "-"*65, sep="")
        best_xgb, xgb_grid_df = grid_search_temporal_leakfree(
            lambda **p: xgb.XGBClassifier(
                n_estimators=GRID_CV_NEST,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=neg_pos_ratio, eval_metric='auc',
                random_state=42, n_jobs=-1, verbosity=0, **p,
            ),
            {'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1],
             'min_child_weight': [5, 10]},
            train_pairs_sorted, train_mc, feature_cols,
            n_splits=GRID_CV_SPLITS, name='XGBoost',
        )
        xgb_grid_df.to_csv(OUTPUTS / "grid_search_xgb.csv", index=False)
        xgb_tuned = xgb.XGBClassifier(
            n_estimators=GRID_REFIT_NEST,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio, eval_metric='auc',
            early_stopping_rounds=30, random_state=42, n_jobs=-1, verbosity=0,
            **best_xgb,
        )
        xgb_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        results['XGBoost_tuned'] = evaluate_model(
            'XGBoost_tuned', y_val, xgb_tuned.predict_proba(X_val)[:, 1], val_ids
        )
        trained_models['XGBoost_tuned'] = xgb_tuned

    # LightGBM Classifier
    if HAS_LGB:
        print("\n" + "-"*65, "\n LightGBM Classifier grid\n", "-"*65, sep="")
        best_lgb, lgb_grid_df = grid_search_temporal_leakfree(
            lambda **p: lgb.LGBMClassifier(
                n_estimators=GRID_CV_NEST,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=neg_pos_ratio,
                random_state=42, n_jobs=-1, verbose=-1, **p,
            ),
            {'num_leaves': [31, 63], 'learning_rate': [0.05, 0.1],
             'min_child_samples': [10, 20]},
            train_pairs_sorted, train_mc, feature_cols,
            n_splits=GRID_CV_SPLITS, name='LightGBM_Clf',
        )
        lgb_grid_df.to_csv(OUTPUTS / "grid_search_lgb.csv", index=False)
        lgb_tuned = lgb.LGBMClassifier(
            n_estimators=GRID_REFIT_NEST,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg_pos_ratio,
            random_state=42, n_jobs=-1, verbose=-1, **best_lgb,
        )
        lgb_tuned.fit(
            X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        results['LGBM_tuned'] = evaluate_model(
            'LGBM_tuned', y_val, lgb_tuned.predict_proba(X_val)[:, 1], val_ids
        )
        trained_models['LGBM_tuned'] = lgb_tuned

    # LightGBM Ranker
    if HAS_LGB:
        print("\n" + "-"*65, "\n LightGBM Ranker grid\n", "-"*65, sep="")
        best_lgb_rank, lgb_rank_grid_df = grid_search_temporal_leakfree(
            lambda **p: lgb.LGBMRanker(
                objective='lambdarank',
                n_estimators=GRID_CV_NEST,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1, **p,
            ),
            {'num_leaves': [31, 63], 'learning_rate': [0.05, 0.1],
             'min_child_samples': [10, 20]},
            train_pairs_sorted, train_mc, feature_cols,
            n_splits=GRID_CV_SPLITS, name='LightGBM_Rank',
            fit_and_score_fn=_fit_score_ranker,
        )
        lgb_rank_grid_df.to_csv(OUTPUTS / "grid_search_lgb_rank.csv", index=False)
        lgb_rank_tuned = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=GRID_REFIT_NEST,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1, **best_lgb_rank,
        )
        lgb_rank_tuned.fit(
            X_train_rank, y_train_rank, group=train_groups,
            eval_set=[(X_val_rank, y_val_rank)], eval_group=[val_groups],
            eval_metric='ndcg', eval_at=[1, 3],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        rt_scores = lgb_rank_tuned.predict(X_val_rank)
        results['LGBM_Rank_tuned'] = {
            **pair_level_metrics(y_val_rank, rt_scores),
            **link_level_accuracy(val_ids_rank, rt_scores),
        }
        print(f"\n  [LGBM_Rank_tuned]")
        for k, v in results['LGBM_Rank_tuned'].items():
            print(f"    {k:<12s}: {v:.4f}")
        trained_models['LGBM_Rank_tuned'] = lgb_rank_tuned


# ════════════════════════════════════════════════════════════════════
# 7. COMPARISON + PERSIST
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print(" MODEL COMPARISON (leakage-free pipeline)")
print("="*80)

comp = (
    pd.DataFrame(results).T
    [['AUC', 'Precision', 'Recall', 'F1', 'Top1_Acc', 'Top3_Acc']]
    .round(4)
)
print(comp.to_string())

best_name = comp['Top1_Acc'].idxmax()
print(f"\n  Best by Top-1: {best_name}  ({comp.loc[best_name, 'Top1_Acc']:.4f})")
comp.to_csv(OUTPUTS / "model_comparison.csv")

# Persist artefacts for predict.py
joblib.dump({
    'model':           trained_models[best_name],
    'model_name':      best_name,
    'transforms':      transforms,
    'scaler':          scaler,           # only used if best_name in {'LogReg','LogReg_tuned'}
    'feature_cols':    feature_cols,
    'safe_feature_cols': safe_feature_cols,
    'historical_cols': historical_cols,
}, MODELS / "best_model.pkl")

# Persist all individual models too
for name, m in trained_models.items():
    joblib.dump(m, MODELS / f"model_{name}.pkl")

print(f"\n  Saved → {OUTPUTS / 'model_comparison.csv'}")
print(f"          {MODELS / 'best_model.pkl'}")
print(f"          {MODELS} / model_*.pkl   ({len(trained_models)} models)")
