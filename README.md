# Aircraft Arrival–Departure Link Prediction

Pairwise machine-learning pipeline that infers which scheduled departure each arriving aircraft will become — the "turnaround link" — from historical schedule data. Each arrival is matched against feasible same-airline / same-aircraft-type departures within an adaptive turnaround window; the model scores every candidate pair, and the highest-scoring candidate per arrival is the predicted link.

## Problem framing

For each arrival *i*, the candidate set *C(i)* is the set of departures from the same airline and aircraft type whose timestamp falls within an adaptive turnaround window. The task is to assign a score *s(i, j)* to every pair and predict the link as `argmax_{j ∈ C(i)} s(i, j)`.

Two equivalent formulations are explored:

| Formulation | Loss | Models |
|---|---|---|
| Binary classification | logistic / `binary_logloss` | LogReg, RandomForest, XGBClassifier, LGBMClassifier |
| Learning-to-rank | LambdaRank (NDCG) | LGBMRanker |

## Operational metric

- **Top-1 accuracy** — proportion of arrivals for which the model's highest-scoring candidate is the true departure.
- **Top-3 accuracy** — proportion of arrivals where the true departure is in the top three scored candidates.

Pair-level metrics (AUC, Precision, Recall, F1) are reported alongside but are secondary; the operational stakeholder cares about per-arrival winner-takes-all accuracy.

## Methodological highlights

### Adaptive candidate windows
For each (airline, aircraft type), the candidate ceiling is the 99th-percentile observed turnaround time × 1.3 margin, with fallback to airline level then global. Ceilings are fit **on the training period only** and applied to both train and validation candidate generation — the candidate-set composition for the validation period reflects what would be available at deployment.

### Leak-free feature engineering
Features split into two groups:

- **Within-row features** (`features.engineer_within_row_features`) — computable from a single (arrival, departure) pair: temporal, turnaround, match, within-arrival rank, hourly congestion. No risk of temporal leakage.
- **Historical features** (`features.fit_historical_features` / `apply_historical_features`) — per-airline / per-aircraft / per-route turnaround statistics, frequency encodings, route-pair frequencies. Implemented as an explicit fit-on-train / transform-on-val pattern:

  ```python
  T = fit_historical_features(train_mc, train_pairs)   # train slice only
  train_pairs = apply_historical_features(train_pairs, T)
  val_pairs   = apply_historical_features(val_pairs,   T)
  ```

  Validation rows whose categorical key (airline, aircraft type, route pair, …) was unseen in training fall back to train-period global defaults stored alongside the lookup tables.

### PSI monitoring
Population Stability Index is computed twice and saved to `outputs/`:

- `psi_before_fe.csv` — within-row features (`build_training_data.py`)
- `psi_after_fe.csv` — full feature set including historical features (`train.py`)

The "after" report breaks down stability into within-row vs historical groups, providing direct evidence that leak-free engineering does not introduce drift beyond what is already present in the underlying data.

### Grid search with temporal CV and per-fold feature refit
Hyperparameter selection uses `TimeSeriesSplit(n_splits=3)` on chronologically sorted training pairs. Critically, the historical features are refit on each fold's training portion alone — otherwise fold-validation statistics leak back into the candidate hyperparameter scores. Selection is by mean Top-1 across folds (AUC as tiebreaker). All five models are tuned; the strongest single-model lift typically comes from the LightGBM ranker, whose `lambdarank` loss aligns with the Top-1 metric.

After CV, the winning configuration is refit on the full training set with early stopping against the validation set.

## Project layout

```
aircraft-link-prediction-ml/
├── README.md
├── Pipfile
├── .gitignore
├── configs/                            # reserved for future YAML config
├── data/
│   ├── raw/EAL_Linking_Data.xlsx       # input schedule data (gitignored)
│   └── processed/                      # parquet artefacts (gitignored)
├── models/                             # saved .pkl bundles (gitignored)
├── outputs/                            # CSVs, PSI reports (gitignored)
├── notebooks/                          # EDA / results notebooks
├── src/
│   ├── features.py                     # adaptive ceilings, FE fit/transform
│   ├── evaluation.py                   # metrics, PSI
│   ├── models.py                       # grid search with per-fold refit
│   ├── build_training_data.py          # script 1: build pairs + within-row FE
│   ├── train.py                        # script 2: fit FE + train + grid search
│   └── predict.py                      # inference using saved best model
└── tests/
    ├── conftest.py
    ├── test_features.py
    ├── test_evaluation.py
    └── test_smoke.py
```

Artefacts produced at run time (all gitignored):

```
data/processed/candidate_pairs_raw.parquet   # pairs + within-row features
data/processed/merged_clean.parquet          # true (arrival, departure) links
data/processed/safe_feature_cols.json        # within-row feature names
data/processed/split_date.txt                # temporal split date

models/candidate_state.pkl                   # ceilings + min_tt for predict.py
models/best_model.pkl                        # winning model + transforms + scaler
models/model_<name>.pkl                      # one per trained model

outputs/psi_before_fe.csv
outputs/psi_after_fe.csv
outputs/grid_search_{lr,rf,xgb,lgb,lgb_rank}.csv
outputs/model_comparison.csv
outputs/predictions.csv                      # produced by predict.py
```

## Setup

```bash
pip install pipenv
cd aircraft-link-prediction-ml
pipenv install --dev
pipenv shell
```

Python 3.14 is pinned in the Pipfile; 3.10+ works after editing `python_version`.

## Running

Scripts run from `src/` in this order:

```bash
cd src
python build_training_data.py     # ~1–3 min: build pairs + within-row FE + PSI(before)
python train.py                    # ~10 min baselines + ~30–45 min grid search
python predict.py                  # score new candidate pairs using saved best model
```

To skip the grid search during iteration, set `RUN_GRID_SEARCH = False` at the top of `train.py`.

## Tests

```bash
cd aircraft-link-prediction-ml
pipenv run pytest tests/ -v
```

Eighteen unit + integration tests cover:
- Adaptive ceiling correctness (floor, hierarchical fallback)
- Within-row feature schema and value ranges (sin/cos bounded, no NaNs)
- Historical fit/apply correctness, unseen-category fallback, train-only fitting invariance
- Pair-level and link-level metric edge cases (perfect, all-wrong, mixed)
- PSI: identical distributions ≈ 0, shifted distributions ≥ 0.25, infinity handling
- End-to-end smoke test on synthetic data

## Outputs

The final comparison table `outputs/model_comparison.csv` contains one row per model (baseline + tuned variants):

| Column | Meaning |
|---|---|
| AUC | Pair-level ROC AUC |
| Precision / Recall / F1 | Pair-level at threshold 0.5 |
| Top1_Acc | **Operational metric** — fraction of arrivals where argmax is correct |
| Top3_Acc | Fraction where the true link is in the top 3 |

## Methodological caveats

1. **Validation is used for early stopping** in baseline GBMs and the tuned refit. Soft leak — the chosen number of boosting rounds depends on validation performance. For strictly unbiased estimates, add a third held-out test set.
2. **Single temporal split.** With only one train/val cut, reported metrics are subject to period-specific noise (1–2 points). A rolling-origin evaluation would average over multiple cuts.
3. **Train-period ceilings assumed to generalize.** Mostly hold; this is exactly the assumption at deployment. Watch the `lost` counter printed during pair construction.
4. **Class imbalance** handled via `class_weight='balanced'` / `scale_pos_weight`. No SMOTE — the pair-construction logic already constrains negatives to feasibility-bounded candidates.

## Reproducibility

`random_state=42` is set on every model and on `TimeSeriesSplit`. The temporal split date is derived from the data (`TRAIN_FRAC = 0.8`) and written to `split_date.txt` so both scripts agree without manual coordination.
