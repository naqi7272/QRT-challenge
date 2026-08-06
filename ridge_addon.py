# -*- coding: utf-8 -*-
"""
ridge_addon.py — per-target Ridge as a decorrelated THIRD ensemble member.

Motivation (from the Upgrade-2 result): per-target projection features were flat,
because the raw illiquid returns are ALREADY in F_train, so the trees gained nothing.
Your real gains have always come from ENSEMBLE DIVERSITY, not more features.

A per-target Ridge is the most on-topic diverse member available:
  * it is the exact linear reconstruction Y = eta(X) the challenge is about,
  * it is regularized (alpha) so it generalizes on ~2,200 effective samples,
  * it is structurally maximally decorrelated from both the NN and the GBMs,
  * and 100 targets x 5 folds of closed-form Ridge fits run in seconds.

Run this AFTER section 10 (needs: gkf, Xf_all, Xf_test, y_panel_filled, mask_panel,
day_ids_train, N_LIQUID, N_FOLDS, nn_oof/nn_test/lgbm_oof/lgbm_test, z_norm,
weighted_acc). Xf_all / Xf_test are the winsorised, zero-filled 100-dim illiquid
panels already defined in the Upgrade-2 block.

LEAK-SAFETY: each Ridge is fit ONLY on its fold's training rows and applied to
val/test — identical discipline to the LGBM loop. Re-run your label-permutation
check after adding it; Ridge-only OOF must collapse to ~0.51 under permutation.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# CRITICAL: features are raw returns (~0.02 magnitude). A fixed alpha on
# unscaled returns is mis-calibrated (XtX is tiny, so a large alpha crushes
# every coefficient to ~0 and the model predicts noise). We therefore STANDARDISE
# inside each fold (scaler fit on train rows only -> leak-safe), which makes alpha
# scale-invariant. 50 is then a sensible default for 100 standardised features on
# ~2,200 samples; sweep [10, 30, 50, 100, 300] on OOF to be sure.
# ---------------------------------------------------------------------------
RIDGE_ALPHA = 50.0


def fit_ridge_oof(gkf, Xf_all, Xf_test, y_panel_filled, mask_panel,
                  day_ids_train, N_LIQUID, N_FOLDS, alpha=RIDGE_ALPHA):
    """Return (ridge_oof, ridge_test), same shapes as nn_oof / nn_test."""
    ridge_oof = np.zeros((Xf_all.shape[0], N_LIQUID), dtype=np.float32)
    ridge_test = np.zeros((Xf_test.shape[0], N_LIQUID), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(Xf_all, groups=day_ids_train)):
        Xtr, Xva = Xf_all[tr_idx], Xf_all[va_idx]
        ytr, mtr = y_panel_filled[tr_idx], mask_panel[tr_idx]
        for j in range(N_LIQUID):
            m = mtr[:, j] == 1
            if m.sum() < 50:
                continue
            w = np.abs(ytr[m, j])
            # StandardScaler (fit on train rows only) + Ridge -> alpha is
            # scale-invariant. weight by |y| to match the competition metric.
            r = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            r.fit(Xtr[m], ytr[m, j], ridge__sample_weight=w)
            ridge_oof[va_idx, j] = r.predict(Xva)
            ridge_test[:, j] += r.predict(Xf_test) / N_FOLDS
        print(f"  ridge fold {fold + 1}/{N_FOLDS} done")
    return ridge_oof, ridge_test


# ===========================================================================
# Wiring — paste after the NN+LGBM blend sweep in section 10
# ===========================================================================
#
#   from ridge_addon import fit_ridge_oof
#   ridge_oof, ridge_test = fit_ridge_oof(
#       gkf, Xf_all, Xf_test, y_panel_filled, mask_panel,
#       day_ids_train, N_LIQUID, N_FOLDS)
#   print("Ridge-only OOF:", weighted_acc(ridge_oof, y_panel_filled, mask_panel))
#
#   ridge_oof_z, ridge_test_z = z_norm(ridge_oof), z_norm(ridge_test)
#
#   # 3-way blend: NN + LGBM + Ridge  (replaces the XGB three-way)
#   best, bwn, bwl = -1.0, 0.33, 0.33
#   for w_nn in np.linspace(0, 1, 21):
#       for w_lgbm in np.linspace(0, 1 - w_nn, 21):
#           w_r = 1 - w_nn - w_lgbm
#           if w_r < 0:
#               continue
#           s = weighted_acc(w_nn * nn_oof_z + w_lgbm * lgbm_oof_z + w_r * ridge_oof_z,
#                            y_panel_filled, mask_panel)
#           if s > best:
#               best, bwn, bwl = s, w_nn, w_lgbm
#   print(f"Best 3-way (NN+LGBM+Ridge): w_nn={bwn:.2f} w_lgbm={bwl:.2f} "
#         f"w_ridge={1 - bwn - bwl:.2f} OOF={best:.5f}")
#
# INTERPRET:
#   * If the 3-way OOF beats 0.7445 by more than ~1 fold-std, Ridge is adding
#     decorrelated signal -> submit it.
#   * If w_ridge -> 0 in the sweep, Ridge is redundant with LGBM; drop it and
#     the linear signal is genuinely exhausted (strong evidence of the ceiling).
#   * Ridge-only OOF itself tells you how much pure-linear signal exists; compare
#     to the 0.511 QRT paired-beta benchmark.
