"""
Inference on the FUTURE schedule.

Pipeline:
  1. Load future arrivals + departures from the Excel "Future schedule" tab.
  2. Use the saved adaptive ceilings (fit on the train period) to define
     each arrival's candidate departure window.
  3. Build candidate pairs (same logic as build_training_data.py but
     without a `label` — the true link is unknown for future flights).
  4. Engineer within-row features.
  5. Apply train-fitted historical feature transforms.
  6. Score with the saved best model.
  7. Argmax per arrival → one predicted (arr_flight_id, predicted_dep_flight_id, score).

Usage (from src/)
-----------------
    python predict.py
    python predict.py --excel ../data/raw/EAL_Linking_Data.xlsx \\
                      --out   ../outputs/future_predictions.csv

Prerequisites
-------------
`build_training_data.py` and `train.py` must have run first so that
`models/candidate_state.pkl` and `models/best_model.pkl` exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from features import apply_historical_features, engineer_within_row_features


ROOT     = Path(__file__).resolve().parent.parent
MODELS   = ROOT / "models"
DATA_RAW = ROOT / "data" / "raw" / "EAL_Linking_Data.xlsx"
OUTPUTS  = ROOT / "outputs"


# ════════════════════════════════════════════════════════════════════
# Artefact loading
# ════════════════════════════════════════════════════════════════════

def load_artefacts(model_path: Path = MODELS / "best_model.pkl") -> dict[str, Any]:
    """Load the saved best-model bundle (model + transforms + scaler + feature_cols)."""
    return joblib.load(model_path)


# ════════════════════════════════════════════════════════════════════
# Future-schedule loading & pair construction
# ════════════════════════════════════════════════════════════════════

def load_future_schedule(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the 'Future schedule' tab and split rows into A / D."""
    df = pd.read_excel(excel_path, sheet_name="Future schedule", header=4)
    df.columns = [
        'datetime', 'pax_cargo', 'flight_id',
        'arr_dep', 'aircraft_type', 'airline', 'origin_dest',
    ]
    df['datetime'] = pd.to_datetime(df['datetime'])
    arrivals   = df[df['arr_dep'] == 'A'].copy().reset_index(drop=True)
    departures = df[df['arr_dep'] == 'D'].copy().reset_index(drop=True)
    return arrivals, departures


def build_future_pairs(
    arrivals: pd.DataFrame,
    departures: pd.DataFrame,
    candidate_state: dict[str, Any],
) -> pd.DataFrame:
    """
    Generate candidate (arrival, departure) pairs for future arrivals
    using the train-fitted adaptive ceilings stored in `candidate_state`.

    Identical feasibility logic to `build_pairs` in build_training_data.py;
    the only difference is the absence of a `label` column.
    """
    ceil_dict      = candidate_state['ceil_dict']
    airl_ceil_dict = candidate_state['airl_ceil_dict']
    glob_ceil      = candidate_state['glob_ceil']
    min_tt         = candidate_state['min_tt']

    departures = departures.copy()
    departures['date'] = departures['datetime'].dt.date
    dep_idx = {
        (dt, al, ac): grp.sort_values('datetime')
        for (dt, al, ac), grp in departures.groupby(['date', 'airline', 'aircraft_type'])
    }

    records: list[dict[str, Any]] = []
    n_no_candidates = 0
    for _, arr in arrivals.iterrows():
        arr_dt   = arr['datetime']
        arr_date = arr_dt.date()
        airline  = arr['airline']
        actype   = arr['aircraft_type']

        ceil = ceil_dict.get(
            (airline, actype),
            airl_ceil_dict.get(airline, glob_ceil),
        )

        parts = []
        for d in [arr_date, arr_date + pd.Timedelta(days=1)]:
            key = (d, airline, actype)
            if key in dep_idx:
                parts.append(dep_idx[key])
        if not parts:
            n_no_candidates += 1
            continue

        cand  = pd.concat(parts)
        delta = (cand['datetime'] - arr_dt).dt.total_seconds() / 60
        valid = cand[(delta >= min_tt) & (delta <= ceil)]
        if valid.empty:
            n_no_candidates += 1
            continue

        for _, dep in valid.iterrows():
            records.append({
                'arr_flight_id':     arr['flight_id'],
                'dep_flight_id':     dep['flight_id'],
                'arr_datetime':      arr_dt,
                'arr_pax_cargo':     arr['pax_cargo'],
                'arr_aircraft_type': actype,
                'arr_airline':       airline,
                'arr_origin_dest':   arr['origin_dest'],
                'dep_datetime':      dep['datetime'],
                'dep_pax_cargo':     dep['pax_cargo'],
                'dep_aircraft_type': dep['aircraft_type'],
                'dep_airline':       dep['airline'],
                'dep_origin_dest':   dep['origin_dest'],
                'turnaround_min':    float(delta.loc[dep.name]),
            })

    print(f"  Arrivals with no feasible candidate: "
          f"{n_no_candidates} / {len(arrivals)}")
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════

def predict_links(
    pairs_df: pd.DataFrame,
    artefacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Score every candidate pair and pick the best departure per arrival.

    Parameters
    ----------
    pairs_df  : raw candidate pairs (pre-feature-engineering schema —
                same columns as produced by build_future_pairs).
    artefacts : output of load_artefacts(); loaded lazily if None.

    Returns
    -------
    DataFrame with columns:
      arr_flight_id, predicted_dep_flight_id, score
    """
    if artefacts is None:
        artefacts = load_artefacts()

    model        = artefacts['model']
    model_name   = artefacts['model_name']
    transforms   = artefacts['transforms']
    scaler       = artefacts['scaler']
    feature_cols = artefacts['feature_cols']

    # Feature engineering (within-row first, then historical via fitted transforms)
    df = engineer_within_row_features(pairs_df)
    df = apply_historical_features(df, transforms)

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Model-family-specific scoring
    if model_name in {'LogReg', 'LogReg_tuned'}:
        scores = model.predict_proba(scaler.transform(X))[:, 1]
    elif model_name in {'LGBM_Rank', 'LGBM_Rank_tuned'}:
        scores = model.predict(X)              # raw ranking scores
    else:
        scores = model.predict_proba(X)[:, 1]

    # Argmax per arrival
    out = df[['arr_flight_id', 'dep_flight_id']].copy()
    out['score'] = scores
    idx_best = out.groupby('arr_flight_id')['score'].idxmax()
    return (
        out.loc[idx_best]
           .rename(columns={'dep_flight_id': 'predicted_dep_flight_id'})
           .reset_index(drop=True)
    )


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Predict turnaround links for the Future schedule."
    )
    p.add_argument("--excel", type=Path, default=DATA_RAW,
                   help="Path to the .xlsx containing the 'Future schedule' tab.")
    p.add_argument("--model", type=Path, default=MODELS / "best_model.pkl",
                   help="Saved best-model bundle from train.py.")
    p.add_argument("--candidate-state", type=Path,
                   default=MODELS / "candidate_state.pkl",
                   help="Saved adaptive ceilings from build_training_data.py.")
    p.add_argument("--out", type=Path,
                   default=OUTPUTS / "future_predictions.csv",
                   help="Output CSV with one predicted departure per future arrival.")
    args = p.parse_args()

    print(f"Loading model bundle from {args.model}...")
    artefacts = load_artefacts(args.model)
    print(f"  Using model: {artefacts['model_name']}")

    print(f"Loading candidate state from {args.candidate_state}...")
    candidate_state = joblib.load(args.candidate_state)

    print(f"Loading Future schedule from {args.excel}...")
    arrivals, departures = load_future_schedule(args.excel)
    print(f"  Future arrivals:   {len(arrivals):,}")
    print(f"  Future departures: {len(departures):,}")

    print(f"\nBuilding candidate pairs...")
    pairs = build_future_pairs(arrivals, departures, candidate_state)
    print(f"  Total candidate pairs:        {len(pairs):,}")
    print(f"  Arrivals with ≥1 candidate:   {pairs['arr_flight_id'].nunique():,}")

    print(f"\nScoring with {artefacts['model_name']}...")
    preds = predict_links(pairs, artefacts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    print(f"\nSaved {len(preds):,} predictions → {args.out}")
    print(f"\nFirst 10 predictions:")
    print(preds.head(10).to_string(index=False))


if __name__ == "__main__":
    _cli()
