# -*- coding: utf-8 -*-
"""
feature_experiments.py — data-side experiments, now that the MODEL side is flat.

Four flat modeling attempts (XGB, projection features, emb-MLP, seed-averaging)
say the signal is not in more models. The only real remaining lever is the
FEATURE space. This file gives:

  A. diagnose_temporal_order(...)  -- THE decision fork. Are ID_DAYs ordered
     (=> lagged/rolling features are possible, the big prize) or shuffled
     (=> temporal FE is impossible; you're at the feature ceiling)?

  B. add_rank_features(...)        -- cross-sectional rank transform of the day's
     illiquid returns. A genuinely different representation (scale/outlier-robust),
     valid EVEN IF days are shuffled. Append in FeatureBuilder.transform.

  C. sector_mean_impute(...)       -- fill missing illiquid returns with that day's
     sector mean instead of 0. Cleaner inputs than artificial zeros.

Protocol is unchanged: add ONE, check GroupKFold OOF + re-run the label-permutation
leak check, submit only if OOF and LB both rise.
"""

import numpy as np


# ===========================================================================
# A. Temporal-order diagnostic  —  RUN THIS FIRST
# ===========================================================================
def diagnose_temporal_order(X_day_train, day_ids_train):
    """Decide whether ID_DAY carries a usable time axis.

    Real financial time series show strong VOLATILITY CLUSTERING: the lag-1
    autocorrelation of |daily aggregate return| is materially positive (~0.2-0.4).
    If days are shuffled/anonymised, that autocorrelation collapses to ~0 and no
    lag/rolling feature can help.

    Returns a dict; prints a verdict.
    """
    order = np.argsort(day_ids_train)               # sort rows by ID_DAY
    X = np.nan_to_num(X_day_train[order])
    # Cross-sectional dispersion per day = std across the ~100 assets. With 100
    # assets this is a TIGHT estimate of that day's volatility, far cleaner than
    # |cross-sectional mean| (which is swamped by an iid sign term). Its lag-1
    # autocorrelation directly measures volatility clustering.
    day_vol = X.std(axis=1)
    day_ret = X.mean(axis=1)

    def lag1_autocorr(s):
        s = s - s.mean()
        denom = (s * s).sum() + 1e-12
        return float((s[1:] * s[:-1]).sum() / denom)

    ac_ret = lag1_autocorr(day_ret)    # return autocorr (usually ~0 even when ordered)
    ac_vol = lag1_autocorr(day_vol)    # dispersion autocorr -> vol clustering signal
    ids = np.sort(day_ids_train)
    consecutive = bool(np.all(np.diff(ids) == 1))

    ordered = ac_vol > 0.10
    print("=== Temporal-order diagnostic ===")
    print(f"  ID_DAY consecutive integers : {consecutive}")
    print(f"  lag-1 autocorr of daily returns    : {ac_ret:+.4f}")
    print(f"  lag-1 autocorr of daily dispersion : {ac_vol:+.4f}   (vol clustering)")
    if ordered:
        print("  VERDICT: days appear ORDERED -> lagged/rolling features are worth "
              "trying (the one big untapped lever).")
    else:
        print("  VERDICT: no volatility clustering -> days are effectively SHUFFLED. "
              "Temporal FE is impossible; you are at the feature ceiling. Use the "
              "cross-sectional experiments (B, C) or invest in the write-up.")
    return {"ac_ret": ac_ret, "ac_vol": ac_vol,
            "consecutive": consecutive, "ordered": ordered}


# ===========================================================================
# B. Cross-sectional rank features (valid even when days are shuffled)
# ===========================================================================
def add_rank_features(X_filled, nan_mask):
    """Per-day cross-sectional rank of each illiquid return, scaled to [-1, 1].
    Missing entries -> 0 (neutral). Shape (n_days, n_illiquid).

    Why: rank is invariant to per-day scale and to outliers, so it is a genuinely
    different view of the panel from the raw returns the models already have.
    Append this block inside FeatureBuilder.transform.
    """
    n, p = X_filled.shape
    ranks = np.zeros((n, p), dtype=np.float32)
    obs = nan_mask < 0.5                       # True where observed
    for d in range(n):
        cols = np.where(obs[d])[0]
        if cols.size < 2:
            continue
        order = np.argsort(np.argsort(X_filled[d, cols]))   # 0..k-1
        ranks[d, cols] = (order / (cols.size - 1)) * 2.0 - 1.0
    return ranks


# ===========================================================================
# C. Sector-mean imputation (alternative to zero-fill)
# ===========================================================================
def sector_mean_impute(X_day, illiquid_sectors_lvl):
    """Fill missing illiquid returns with that day's mean over the SAME sector
    (level passed in via illiquid_sectors_lvl, an array of length n_illiquid).
    Falls back to the day's overall mean, then 0. Returns (X_filled, nan_mask).

    Use in place of FeatureBuilder._impute_and_indicator's zero-fill if the OOF
    check likes it.
    """
    nan_mask = np.isnan(X_day).astype(np.float32)
    X = X_day.copy()
    n = X.shape[0]
    sectors = np.unique(illiquid_sectors_lvl)
    for d in range(n):
        row = X[d]
        miss = np.isnan(row)
        if not miss.any():
            continue
        day_mean = np.nanmean(row) if np.isfinite(np.nanmean(row)) else 0.0
        for s in sectors:
            cols = np.where(illiquid_sectors_lvl == s)[0]
            obs = cols[~np.isnan(row[cols])]
            sec_mean = row[obs].mean() if obs.size else day_mean
            fill_cols = cols[np.isnan(row[cols])]
            row[fill_cols] = sec_mean
        row[np.isnan(row)] = day_mean
        X[d] = row
    return np.nan_to_num(X).astype(np.float32), nan_mask


# Wiring for B (add to FeatureBuilder.transform's feats list):
#   from feature_experiments import add_rank_features
#   feats.append(add_rank_features(X_filled, nan_mask))   # +100 columns
#
# Wiring for A (run once, right after build_day_panel):
#   from feature_experiments import diagnose_temporal_order
#   diagnose_temporal_order(X_day_train, day_ids_train)
