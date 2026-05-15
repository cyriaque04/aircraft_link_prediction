"""
Build the candidate-pair training dataset.

Steps
-----
1. Load arrival / departure schedule, reconstruct merged_clean.
2. Determine temporal split date (TRAIN_FRAC of date range).
3. Fit adaptive ceilings on the train slice only.
4. Build candidate pairs for ALL arrivals using train-fitted ceilings.
5. Engineer WITHIN-ROW features (no global aggregation, leak-free).
6. Compute PSI (train vs val) on the within-row features.
7. Persist artefacts to ../data/processed/ and ../outputs/.

Run from the `src/` directory:
    python build_training_data.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import (
    compute_adaptive_ceilings,
    engineer_within_row_features,
)
from evaluation import psi_report

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw" / "EAL_Linking_Data.xlsx"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS   = ROOT / "outputs"
MODELS    = ROOT / "models"

TRAIN_FRAC = 0.8                # MUST match train.py
MIN_TT_MIN = 25                 # operational minimum turnaround


# ════════════════════════════════════════════════════════════════════
# 1. LOAD DATA & RECONSTRUCT merged_clean
# ════════════════════════════════════════════════════════════════════

print("Loading data...")
df_hist   = pd.read_excel(DATA_RAW, sheet_name="Historic schedule", header=4)
df_future = pd.read_excel(DATA_RAW, sheet_name="Future schedule",   header=4)

df_hist.columns = [
    'datetime', 'pax_cargo', 'flight_id', 'turnaround_id',
    'arr_dep', 'aircraft_type', 'airline', 'origin_dest',
]
df_future.columns = [
    'datetime', 'pax_cargo', 'flight_id',
    'arr_dep', 'aircraft_type', 'airline', 'origin_dest',
]
df_hist['datetime']   = pd.to_datetime(df_hist['datetime'])
df_future['datetime'] = pd.to_datetime(df_future['datetime'])

arrivals   = df_hist[df_hist['arr_dep'] == 'A'].copy().reset_index(drop=True)
departures = df_hist[df_hist['arr_dep'] == 'D'].copy().reset_index(drop=True)

merged = arrivals.merge(
    departures[['flight_id', 'datetime', 'pax_cargo', 'origin_dest']].rename(columns={
        'flight_id':   'dep_flight_id',
        'datetime':    'dep_datetime',
        'pax_cargo':   'dep_pax_cargo',
        'origin_dest': 'dep_origin_dest',
    }),
    left_on='turnaround_id', right_on='dep_flight_id', how='inner',
)
merged.rename(columns={
    'datetime':      'arr_datetime',
    'flight_id':     'arr_flight_id',
    'pax_cargo':     'arr_pax_cargo',
    'origin_dest':   'arr_origin_dest',
    'aircraft_type': 'arr_aircraft_type',
    'airline':       'arr_airline',
}, inplace=True)

merged['turnaround_time'] = (
    merged['dep_datetime'] - merged['arr_datetime']
).dt.total_seconds() / 60

merged_clean = merged[
    (merged['turnaround_time'] >= MIN_TT_MIN) &
    (merged['turnaround_time'] <= 1440)
].copy().reset_index(drop=True)

print(f"True links (merged_clean): {len(merged_clean):,}")


# ════════════════════════════════════════════════════════════════════
# 2. TEMPORAL SPLIT DATE
# ════════════════════════════════════════════════════════════════════

all_dates  = sorted(merged_clean['arr_datetime'].dt.date.unique())
SPLIT_DATE = all_dates[int(len(all_dates) * TRAIN_FRAC)]
print(f"\n  Split date: {SPLIT_DATE}")

train_mc = merged_clean[merged_clean['arr_datetime'].dt.date < SPLIT_DATE].copy()
print(f"  Train true links: {len(train_mc):,}")


# ════════════════════════════════════════════════════════════════════
# 3. ADAPTIVE CEILINGS (fit on train only)
# ════════════════════════════════════════════════════════════════════

ceil_dict, airl_ceil_dict, glob_ceil = compute_adaptive_ceilings(train_mc)
print(f"\n  Ceilings: {len(ceil_dict)} (airline, aircraft) groups, "
      f"global fallback = {glob_ceil:.0f} min")


# ════════════════════════════════════════════════════════════════════
# 4. BUILD CANDIDATE PAIRS
# ════════════════════════════════════════════════════════════════════

def build_pairs(mc, departures, ceil_dict, airl_ceil_dict, glob_ceil, min_tt=MIN_TT_MIN):
    departures = departures.copy()
    departures['date'] = departures['datetime'].dt.date
    dep_idx = {
        (dt, al, ac): grp.sort_values('datetime')
        for (dt, al, ac), grp in departures.groupby(['date', 'airline', 'aircraft_type'])
    }

    records = []
    lost = 0
    for _, row in mc.iterrows():
        arr_dt      = row['arr_datetime']
        arr_date    = arr_dt.date()
        airline     = row['arr_airline']
        actype      = row['arr_aircraft_type']
        true_dep_id = row['dep_flight_id']

        ceil = ceil_dict.get((airline, actype),
                              airl_ceil_dict.get(airline, glob_ceil))

        parts = []
        for d in [arr_date, arr_date + pd.Timedelta(days=1)]:
            key = (d, airline, actype)
            if key in dep_idx:
                parts.append(dep_idx[key])
        if not parts:
            lost += 1
            continue

        cand = pd.concat(parts)
        delta = (cand['datetime'] - arr_dt).dt.total_seconds() / 60
        valid = cand[(delta >= min_tt) & (delta <= ceil)]
        if valid.empty:
            lost += 1
            continue

        true_in = true_dep_id in valid['flight_id'].values
        for _, dep in valid.iterrows():
            records.append({
                'arr_flight_id':     row['arr_flight_id'],
                'dep_flight_id':     dep['flight_id'],
                'label':             int(dep['flight_id'] == true_dep_id),
                'arr_datetime':      arr_dt,
                'arr_pax_cargo':     row['arr_pax_cargo'],
                'arr_aircraft_type': actype,
                'arr_airline':       airline,
                'arr_origin_dest':   row['arr_origin_dest'],
                'dep_datetime':      dep['datetime'],
                'dep_pax_cargo':     dep['pax_cargo'],
                'dep_aircraft_type': dep['aircraft_type'],
                'dep_airline':       dep['airline'],
                'dep_origin_dest':   dep['origin_dest'],
                'turnaround_min':    delta.loc[dep.name],
            })
        if not true_in:
            lost += 1

    print(f"  Arrivals where true link fell outside window: {lost}")
    return pd.DataFrame(records)


print("\nBuilding candidate pairs...")
t0 = time.time()
pairs = build_pairs(merged_clean, departures, ceil_dict, airl_ceil_dict, glob_ceil)
print(f"  Done in {time.time()-t0:.0f}s   "
      f"Total: {len(pairs):,}   Pos: {pairs['label'].sum():,}   "
      f"Neg: {(pairs['label']==0).sum():,}")


# ════════════════════════════════════════════════════════════════════
# 5. WITHIN-ROW FEATURES
# ════════════════════════════════════════════════════════════════════

print("\nEngineering within-row features...")
pairs_fe = engineer_within_row_features(pairs)

id_cols  = ['arr_flight_id', 'dep_flight_id']
raw_cats = [
    'arr_datetime', 'dep_datetime',
    'arr_pax_cargo', 'dep_pax_cargo',
    'arr_origin_dest', 'dep_origin_dest',
    'arr_airline', 'dep_airline',
    'arr_aircraft_type', 'dep_aircraft_type',
]
keep_for_downstream = ['arr_tod', 'arr_dow', 'arr_month', 'turnaround_min']

safe_feat_cols = [
    c for c in pairs_fe.columns
    if c not in id_cols + raw_cats + ['label'] + keep_for_downstream
    and pd.api.types.is_numeric_dtype(pairs_fe[c])
]
print(f"  Safe features: {len(safe_feat_cols)}")


# ════════════════════════════════════════════════════════════════════
# 6. PSI BEFORE FEATURE ENGINEERING (within-row only)
# ════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print(" PSI before feature engineering (train vs val periods)")
print("="*65)

_train_psi = pairs_fe[pairs_fe['arr_datetime'].dt.date <  SPLIT_DATE]
_val_psi   = pairs_fe[pairs_fe['arr_datetime'].dt.date >= SPLIT_DATE]
psi_before = psi_report(_train_psi, _val_psi, safe_feat_cols)

n_ok  = (psi_before['severity'] == 'OK').sum()
n_mod = (psi_before['severity'] == 'MODERATE').sum()
n_sig = (psi_before['severity'] == 'SIGNIFICANT').sum()
print(f"  Severity: OK={n_ok}  MODERATE={n_mod}  SIGNIFICANT={n_sig}")
print(f"\n  Top 10 features by drift:")
for _, r in psi_before.head(10).iterrows():
    print(f"    {r['feature']:<28s}  PSI={r['PSI']:.4f}  [{r['severity']}]")


# ════════════════════════════════════════════════════════════════════
# 7. SAVE
# ════════════════════════════════════════════════════════════════════

DATA_PROC.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

pairs_fe.to_parquet(DATA_PROC / "candidate_pairs_raw.parquet", index=False)
merged_clean.to_parquet(DATA_PROC / "merged_clean.parquet",    index=False)
psi_before.to_csv(OUTPUTS / "psi_before_fe.csv", index=False)

with open(DATA_PROC / "safe_feature_cols.json", "w") as f:
    json.dump(safe_feat_cols, f, indent=2)
with open(DATA_PROC / "split_date.txt", "w") as f:
    f.write(str(SPLIT_DATE))

# Save the candidate-window state too: needed by predict.py for new arrivals
joblib.dump(
    {'ceil_dict': ceil_dict,
     'airl_ceil_dict': airl_ceil_dict,
     'glob_ceil': glob_ceil,
     'min_tt': MIN_TT_MIN,
     'split_date': str(SPLIT_DATE)},
    MODELS / "candidate_state.pkl",
)

print(f"\n  Saved → {DATA_PROC}")
print(f"          {OUTPUTS / 'psi_before_fe.csv'}")
print(f"          {MODELS  / 'candidate_state.pkl'}")
