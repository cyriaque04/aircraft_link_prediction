"""
Feature engineering — aircraft turnaround link prediction.

Two-tier feature design (leakage-safe by construction):
  • Within-row features   — computable from a single (arrival, departure)
                            pair, no global aggregation. Engineered once
                            during candidate-pair construction.
  • Historical features   — per-airline / per-aircraft / per-route
                            turnaround statistics + frequency encodings.
                            Implemented as fit(train) / transform(any)
                            so train and val never share statistics.

Also exposes the adaptive ceiling fitter used to define the per-arrival
candidate window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════
# Adaptive candidate ceilings
# ════════════════════════════════════════════════════════════════════

def compute_adaptive_ceilings(
    mc: pd.DataFrame,
    margin: float = 1.3,
    floor_min: float = 60.0,
    min_obs: int = 5,
) -> tuple[dict[tuple[str, str], float], dict[str, float], float]:
    """
    Build a hierarchical turnaround-time ceiling per (airline, aircraft):
      (airline × aircraft)  if observations ≥ min_obs
      → (airline)           fallback
      → global              fallback
    Each ceiling is the 99th-percentile turnaround × `margin`, floored
    at `floor_min` minutes. Returned dicts and the global value are
    consumed by the candidate-pair builder.
    """
    cross = mc.groupby(
        ['arr_airline', 'arr_aircraft_type']
    )['turnaround_time'].agg(
        p99=lambda x: x.quantile(0.99), count='count'
    ).reset_index()
    cross['ceiling'] = (cross['p99'] * margin).clip(lower=floor_min)
    cross.loc[cross['count'] < min_obs, 'ceiling'] = np.nan

    airl = mc.groupby('arr_airline')['turnaround_time'].quantile(0.99).reset_index()
    airl.columns = ['arr_airline', 'p99_airl']
    airl['ceil_airl'] = (airl['p99_airl'] * margin).clip(lower=floor_min)

    glob_ceil = float(mc['turnaround_time'].quantile(0.99) * margin)

    cross = cross.merge(airl, on='arr_airline', how='left')
    cross['ceil_final'] = cross['ceiling'].fillna(cross['ceil_airl']).fillna(glob_ceil)

    ceil_dict = {
        (r['arr_airline'], r['arr_aircraft_type']): float(r['ceil_final'])
        for _, r in cross.iterrows()
    }
    airl_dict = dict(zip(airl['arr_airline'], airl['ceil_airl'].astype(float)))
    return ceil_dict, airl_dict, glob_ceil


# ════════════════════════════════════════════════════════════════════
# Within-row features (safe under any temporal split)
# ════════════════════════════════════════════════════════════════════

def engineer_within_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that depend only on a single (arrival, departure)
    pair — temporal, turnaround, match, within-arrival rank, hourly
    congestion. None of these use cross-period aggregation.

    The `dep_hr_congestion` feature groups by a fully-qualified hourly
    timestamp; train hours and val hours never overlap because the date
    is part of the key, so this remains leak-free.
    """
    df = df.copy()

    # Temporal
    df['arr_hour']  = df['arr_datetime'].dt.hour
    df['dep_hour']  = df['dep_datetime'].dt.hour
    df['arr_dow']   = df['arr_datetime'].dt.dayofweek
    df['arr_month'] = df['arr_datetime'].dt.month

    arr_hf = df['arr_hour'] + df['arr_datetime'].dt.minute / 60
    dep_hf = df['dep_hour'] + df['dep_datetime'].dt.minute / 60

    df['arr_hour_sin'] = np.sin(2 * np.pi * arr_hf / 24)
    df['arr_hour_cos'] = np.cos(2 * np.pi * arr_hf / 24)
    df['dep_hour_sin'] = np.sin(2 * np.pi * dep_hf / 24)
    df['dep_hour_cos'] = np.cos(2 * np.pi * dep_hf / 24)
    df['dow_sin']      = np.sin(2 * np.pi * df['arr_dow'] / 7)
    df['dow_cos']      = np.cos(2 * np.pi * df['arr_dow'] / 7)
    df['month_sin']    = np.sin(2 * np.pi * df['arr_month'] / 12)
    df['month_cos']    = np.cos(2 * np.pi * df['arr_month'] / 12)

    df['is_weekend']   = (df['arr_dow'] >= 5).astype(int)
    df['is_overnight'] = (df['dep_datetime'].dt.date > df['arr_datetime'].dt.date).astype(int)

    df['arr_tod'] = pd.cut(
        df['arr_hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]
    ).astype(int)

    # Turnaround
    df['turnaround_log'] = np.log1p(df['turnaround_min'])

    # Match
    df['same_pax_cargo']   = (df['arr_pax_cargo']   == df['dep_pax_cargo']).astype(int)
    df['same_origin_dest'] = (df['arr_origin_dest'] == df['dep_origin_dest']).astype(int)

    # Within-arrival rank / competition
    df['tt_rank']           = df.groupby('arr_flight_id')['turnaround_min'].rank(method='dense')
    df['tt_rank_pct']       = df.groupby('arr_flight_id')['turnaround_min'].rank(pct=True)
    df['is_closest']        = (df['tt_rank'] == 1).astype(int)
    df['n_candidates']      = df.groupby('arr_flight_id')['dep_flight_id'].transform('count')
    df['n_competitors_dep'] = df.groupby('dep_flight_id')['arr_flight_id'].transform('count')

    # Hourly departure congestion (timestamp-floored hour ⇒ no cross-period leak)
    dep_hr = df['dep_datetime'].dt.floor('h')
    df['dep_hr_congestion'] = df.groupby(dep_hr)['dep_flight_id'].transform('nunique')

    return df


# ════════════════════════════════════════════════════════════════════
# Historical features — fit on train, apply anywhere
# ════════════════════════════════════════════════════════════════════

def fit_historical_features(
    train_mc: pd.DataFrame,
    train_pairs: pd.DataFrame,
) -> dict[str, Any]:
    """
    Fit globally-aggregated feature transforms using TRAIN data only.

    Returns a dict containing:
      • lookup tables (DataFrames) to merge by key
      • frequency-encoding maps
      • train-period global defaults for unseen categories
    """
    T: dict[str, Any] = {}

    T['al_stats'] = train_mc.groupby('arr_airline')['turnaround_time'].agg(
        al_tt_mean='mean', al_tt_med='median', al_tt_std='std',
        al_tt_q25=lambda x: x.quantile(0.25),
        al_tt_q75=lambda x: x.quantile(0.75),
    ).reset_index()

    T['ac_stats'] = train_mc.groupby('arr_aircraft_type')['turnaround_time'].agg(
        ac_tt_mean='mean', ac_tt_med='median', ac_tt_std='std',
    ).reset_index()

    T['cx_stats'] = train_mc.groupby(
        ['arr_airline', 'arr_aircraft_type']
    )['turnaround_time'].agg(
        cx_tt_mean='mean', cx_tt_med='median'
    ).reset_index()

    mc_tmp = train_mc.copy()
    mc_tmp['arr_tod'] = pd.cut(
        mc_tmp['arr_datetime'].dt.hour,
        bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]
    ).astype(int)
    tod = mc_tmp.groupby(['arr_airline', 'arr_tod'])['turnaround_time'].median().reset_index()
    tod.columns = ['arr_airline', 'arr_tod', 'al_tod_tt_med']
    T['al_tod'] = tod

    mc_tmp['arr_dow'] = mc_tmp['arr_datetime'].dt.dayofweek
    dow = mc_tmp.groupby(['arr_airline', 'arr_dow'])['turnaround_time'].median().reset_index()
    dow.columns = ['arr_airline', 'arr_dow', 'al_dow_tt_med']
    T['al_dow'] = dow

    mintt = train_mc.groupby('arr_aircraft_type')['turnaround_time'].min().reset_index()
    mintt.columns = ['arr_aircraft_type', 'ac_min_tt']
    T['min_tt'] = mintt

    rp = train_mc.groupby(
        ['arr_origin_dest', 'dep_origin_dest']
    ).size().reset_index(name='route_pair_freq')
    T['route_pair'] = rp

    al_daily = train_mc.groupby(
        ['arr_airline', train_mc['arr_datetime'].dt.date]
    ).size().reset_index(name='n')
    al_daily_avg = al_daily.groupby('arr_airline')['n'].mean().reset_index()
    al_daily_avg.columns = ['arr_airline', 'al_daily_avg']
    T['al_daily'] = al_daily_avg

    # Frequency encodings: fit on TRAIN PAIRS so they reflect the
    # candidate-set composition the model actually sees at fit time.
    T['fenc'] = {
        col: train_pairs[col].value_counts(normalize=True).to_dict()
        for col in ['arr_airline', 'arr_origin_dest', 'dep_origin_dest', 'arr_aircraft_type']
    }

    g_mean = float(train_mc['turnaround_time'].mean())
    g_med  = float(train_mc['turnaround_time'].median())
    g_std  = float(train_mc['turnaround_time'].std())
    g_q25  = float(train_mc['turnaround_time'].quantile(0.25))
    g_q75  = float(train_mc['turnaround_time'].quantile(0.75))
    g_min  = float(train_mc['turnaround_time'].min())
    g_dens = float(al_daily['n'].mean())

    T['defaults'] = {
        'al_tt_mean':    g_mean,
        'al_tt_med':     g_med,
        'al_tt_std':     g_std,
        'al_tt_q25':     g_q25,
        'al_tt_q75':     g_q75,
        'ac_tt_mean':    g_mean,
        'ac_tt_med':     g_med,
        'ac_tt_std':     g_std,
        'cx_tt_mean':    g_mean,
        'cx_tt_med':     g_med,
        'al_tod_tt_med': g_med,
        'al_dow_tt_med': g_med,
        'ac_min_tt':     g_min,
        'route_pair_freq': 0.0,
        'al_daily_avg':  g_dens,
    }
    return T


def apply_historical_features(
    pairs_df: pd.DataFrame,
    T: dict[str, Any],
) -> pd.DataFrame:
    """Apply fitted historical-feature transforms to any pairs DataFrame."""
    df = pairs_df.copy()

    df = df.merge(T['al_stats'],   on='arr_airline',                          how='left')
    df = df.merge(T['ac_stats'],   on='arr_aircraft_type',                    how='left')
    df = df.merge(T['cx_stats'],   on=['arr_airline', 'arr_aircraft_type'],   how='left')
    df = df.merge(T['al_tod'],     on=['arr_airline', 'arr_tod'],             how='left')
    df = df.merge(T['al_dow'],     on=['arr_airline', 'arr_dow'],             how='left')
    df = df.merge(T['min_tt'],     on='arr_aircraft_type',                    how='left')
    df = df.merge(T['route_pair'], on=['arr_origin_dest', 'dep_origin_dest'], how='left')
    df = df.merge(T['al_daily'],   on='arr_airline',                          how='left')

    for col, default in T['defaults'].items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    df['tt_vs_al_med']     = df['turnaround_min'] - df['al_tt_med']
    df['tt_vs_al_mean']    = df['turnaround_min'] - df['al_tt_mean']
    df['tt_z_score']       = (df['turnaround_min'] - df['al_tt_mean']) / df['al_tt_std'].replace(0, 1)
    df['tt_in_iqr'] = (
        (df['turnaround_min'] >= df['al_tt_q25']) &
        (df['turnaround_min'] <= df['al_tt_q75'])
    ).astype(int)
    df['tt_vs_ac_med']     = df['turnaround_min'] - df['ac_tt_med']
    df['tt_vs_cx_med']     = df['turnaround_min'] - df['cx_tt_med']
    df['tt_vs_tod_med']    = df['turnaround_min'] - df['al_tod_tt_med']
    df['tt_vs_dow_med']    = df['turnaround_min'] - df['al_dow_tt_med']
    df['tt_abs_vs_al_med'] = df['tt_vs_al_med'].abs()
    df['tt_abs_vs_cx_med'] = df['tt_vs_cx_med'].abs()

    df['margin_over_min_tt']  = df['turnaround_min'] - df['ac_min_tt']
    df['route_pair_log_freq'] = np.log1p(df['route_pair_freq'])

    for col, fmap in T['fenc'].items():
        df[f'{col}_fenc'] = df[col].map(fmap).fillna(0.0)

    return df
