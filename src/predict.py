"""
Inference script — score new candidate pairs with the saved best model.

Usage
-----
1. Run `build_training_data.py` and `train.py` first to produce
   `models/best_model.pkl` and `models/candidate_state.pkl`.
2. Call `predict_links(pairs_df)` from your code, OR run this script
   with a custom candidate-pairs parquet (see `--pairs` CLI arg).

The function expects a DataFrame with the same SCHEMA as the raw
candidate pairs produced by `build_training_data.py` BEFORE feature
engineering — i.e. these columns:
    arr_flight_id, dep_flight_id, label (optional),
    arr_datetime, dep_datetime,
    arr_pax_cargo, dep_pax_cargo,
    arr_origin_dest, dep_origin_dest,
    arr_airline, dep_airline,
    arr_aircraft_type, dep_aircraft_type,
    turnaround_min

The same within-row + historical features are computed using the
TRAIN-FITTED transforms — no fresh fitting happens here, so the
inference pipeline is fully reproducible and leak-free.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from features import apply_historical_features, engineer_within_row_features


ROOT      = Path(__file__).resolve().parent.parent
MODELS    = ROOT / "models"
DATA_PROC = ROOT / "data" / "processed"


def load_artefacts(model_path: Path = MODELS / "best_model.pkl") -> dict[str, Any]:
    """Load the saved best model bundle."""
    return joblib.load(model_path)


def predict_links(
    pairs_df: pd.DataFrame,
    artefacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Score every candidate pair and pick the best departure per arrival.

    Returns
    -------
    A DataFrame with columns:
      arr_flight_id, predicted_dep_flight_id, score
    The score is the model's raw output (probability or rank score).
    """
    if artefacts is None:
        artefacts = load_artefacts()

    model           = artefacts['model']
    model_name      = artefacts['model_name']
    transforms      = artefacts['transforms']
    scaler          = artefacts['scaler']
    feature_cols    = artefacts['feature_cols']

    # 1. Within-row features
    df = engineer_within_row_features(pairs_df)

    # 2. Historical features via fitted transforms
    df = apply_historical_features(df, transforms)

    # 3. Assemble X
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Score
    if model_name in {'LogReg', 'LogReg_tuned'}:
        X_in = scaler.transform(X)
        scores = model.predict_proba(X_in)[:, 1]
    elif model_name in {'LGBM_Rank', 'LGBM_Rank_tuned'}:
        scores = model.predict(X)
    else:
        scores = model.predict_proba(X)[:, 1]

    # 5. Argmax per arrival
    out = df[['arr_flight_id', 'dep_flight_id']].copy()
    out['score'] = scores
    idx_best = out.groupby('arr_flight_id')['score'].idxmax()
    return (
        out.loc[idx_best]
           .rename(columns={'dep_flight_id': 'predicted_dep_flight_id'})
           .reset_index(drop=True)
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description="Score candidate pairs with the saved best model.")
    p.add_argument("--pairs", type=Path,
                   default=DATA_PROC / "candidate_pairs_raw.parquet",
                   help="Parquet of candidate pairs (raw schema, pre-FE).")
    p.add_argument("--model", type=Path,
                   default=MODELS / "best_model.pkl",
                   help="Saved best-model bundle.")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs" / "predictions.csv",
                   help="Output CSV.")
    args = p.parse_args()

    print(f"Loading model bundle from {args.model}...")
    artefacts = load_artefacts(args.model)
    print(f"  Using model: {artefacts['model_name']}")

    print(f"Loading candidate pairs from {args.pairs}...")
    pairs = pd.read_parquet(args.pairs)
    pairs['arr_datetime'] = pd.to_datetime(pairs['arr_datetime'])
    pairs['dep_datetime'] = pd.to_datetime(pairs['dep_datetime'])
    print(f"  {len(pairs):,} pairs over {pairs['arr_flight_id'].nunique():,} arrivals")

    preds = predict_links(pairs, artefacts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    print(f"\nSaved {len(preds):,} predictions → {args.out}")


if __name__ == "__main__":
    _cli()
