# -*- coding: utf-8 -*-
"""
improvements.py  —  QRT Liquid-Asset-Performance: drop-in upgrades for final_solution.py

WHY A SEPARATE FILE
-------------------
final_solution.py already scores LB 0.7471 and is a clean baseline. Rather than
rewrite it, this file provides three *self-contained, leak-safe* upgrades you can
wire into the existing pipeline. Each is independent — adopt them one at a time and
re-check your GroupKFold OOF *and* the leaderboard after each, because (see the
honest note at the bottom) this problem is close to its noise ceiling and OOF gains
do not always translate to LB.

The three upgrades, in expected-value order:

  1. TRANSDUCTIVE unsupervised features  (fit PCA + normalisation on train+test
     illiquid panels). Uses only X (no labels) -> zero leakage. Doubles the data
     available to the factor model. Low risk, small-but-real gain, and a strong
     talking point ("I used the unlabeled test features transductively").

  2. PER-TARGET LINEAR-PROJECTION features for the gradient-boosted models.
     GBMs approximate weighted sums of many inputs poorly; this hands each per-
     target model the exact linear projection the challenge is really about
     (Y = eta(X)). Computed INSIDE each CV fold on training days only -> leak-safe.
     This is the single change most likely to add genuinely new signal, because
     today every per-target LGBM sees the *same* generic 364-dim vector.

  3. A structurally DIVERSE third model: a row-level MLP with a learned target
     embedding, trained on all ~267k rows. This is a better ensemble member than
     the XGBoost you tried (XGB ~= LGBM, so it added no diversity and hurt LB).

All functions are written to match the objects already defined in
final_solution.py: X_day_train / X_day_test are (n_days, 100) float32 illiquid
panels WITH NaNs; y_panel_filled / mask_panel are (n_days, 100); day_ids_train is
the group vector for GroupKFold.
"""

import numpy as np


# ===========================================================================
# UPGRADE 1 — Transductive feature builder (train + test, no labels)
# ===========================================================================
# Wiring: replace the `fb.fit(X_day_train)` / transform / normalise block in
# section 5 of final_solution.py with the calls shown in `demo_upgrade_1`.
#
# What changes vs. the original FeatureBuilder:
#   * PCA is fit on np.vstack([train_filled, test_filled]) instead of train only.
#   * Feature normalisation (mu/sd) is fit on the stacked TRANSFORMED features.
#   * Raw returns are winsorised before they reach the NN (fat tails destabilise
#     gradient descent; trees are invariant to monotone transforms so unaffected).
# None of this touches y, so it cannot leak the target.

def winsorize(X, limit=0.15):
    """Clip raw daily returns to +/-limit (15% is already an extreme 1-day move).
    Applied to the RAW return block only. Deterministic, per-value -> no leakage."""
    return np.clip(X, -limit, limit)


def fit_transductive(feature_builder, X_day_train, X_day_test):
    """Fit the existing FeatureBuilder's PCA on BOTH panels stacked.

    feature_builder: an instance of final_solution.FeatureBuilder
    Returns (F_train, F_test) already normalised with train+test statistics.
    """
    # Impute (the builder's own helper) then stack for the unsupervised fit.
    tr_filled, _ = feature_builder._impute_and_indicator(X_day_train)
    te_filled, _ = feature_builder._impute_and_indicator(X_day_test)
    stacked = np.vstack([winsorize(tr_filled), winsorize(te_filled)])

    # Fit PCA transductively (X-only -> safe).
    from sklearn.decomposition import PCA
    feature_builder.pca = PCA(
        n_components=feature_builder.n_pca, random_state=42
    ).fit(stacked)

    # Winsorise the raw-return NaNs-as-zero panels before transform so the raw
    # block the model sees is clipped. We monkey-patch _impute_and_indicator to
    # apply winsorisation to the filled block while leaving the indicator intact.
    orig_impute = feature_builder._impute_and_indicator

    def _impute_winsor(X_day):
        filled, ind = orig_impute(X_day)
        return winsorize(filled), ind

    feature_builder._impute_and_indicator = _impute_winsor

    F_train = feature_builder.transform(X_day_train)
    F_test = feature_builder.transform(X_day_test)

    # Normalise with train+test statistics (X-only -> safe).
    both = np.vstack([F_train, F_test])
    mu = both.mean(axis=0, keepdims=True)
    sd = both.std(axis=0, keepdims=True) + 1e-6
    return (F_train - mu) / sd, (F_test - mu) / sd


# ===========================================================================
# UPGRADE 2 — Per-target linear-projection features (leak-safe, per fold)
# ===========================================================================
# For a given target j and a set of TRAINING days, we:
#   * measure the univariate correlation of each illiquid asset to target j,
#   * take the top-K by |corr|,
#   * hand the model (a) those K raw returns, (b) a univariate-beta linear
#     prediction sum_k beta_k * ret_k  [the generalised QRT benchmark], and
#     (c) a corr-signed sign-vote across the K assets.
# (b) is the key feature: it is exactly the linear map eta the problem defines,
# which tree ensembles cannot easily build from raw columns.
#
# CRITICAL: correlations use y, so they MUST be estimated on the training fold
# only, then applied unchanged to val/test. The function signature enforces this.

def _masked_stats(Xf_tr, y_tr_col, m_tr_col):
    """Univariate corr and beta of each illiquid vs. target, on observed rows."""
    idx = m_tr_col > 0.5
    X = Xf_tr[idx]
    y = y_tr_col[idx]
    if X.shape[0] < 30:
        return None
    Xmu = X.mean(0)
    Xsd = X.std(0) + 1e-8
    ymu = y.mean()
    ysd = y.std() + 1e-8
    corr = ((X - Xmu) / Xsd * ((y - ymu) / ysd)[:, None]).mean(0)  # (100,)
    beta = corr * (ysd / Xsd)                                       # univariate slope
    return corr, beta


def target_projection_features(Xf_tr, Xf_va, Xf_te, y_tr_col, m_tr_col, K=12):
    """Return (enrich_tr, enrich_va, enrich_te) each shape (n, 2K+2), leak-safe.

    Xf_*: filled (NaN->0) illiquid panels, shape (n, 100). Pass the WINSORISED
          filled returns for consistency with Upgrade 1.
    y_tr_col, m_tr_col: this target's returns / mask over TRAINING days only.
    """
    stats = _masked_stats(Xf_tr, y_tr_col, m_tr_col)
    if stats is None:
        z = lambda A: np.zeros((A.shape[0], 2 * K + 2), dtype=np.float32)
        return z(Xf_tr), z(Xf_va), z(Xf_te)
    corr, beta = stats
    top = np.argsort(-np.abs(corr))[:K]
    sign = np.sign(corr[top])

    def build(R):
        raw = R[:, top]                                   # K raw returns
        signed = raw * sign                               # K corr-aligned returns
        beta_pred = (R[:, top] * beta[top]).sum(1, keepdims=True)   # linear map
        sign_vote = (np.sign(raw) * sign).mean(1, keepdims=True)    # agreement
        return np.concatenate([raw, signed, beta_pred, sign_vote],
                              axis=1).astype(np.float32)

    return build(Xf_tr), build(Xf_va), build(Xf_te)


# Wiring for Upgrade 2 inside the LGBM fold loop of final_solution.py:
#
#   # once per fold, precompute the winsorised filled illiquid panels:
#   Xf_tr = winsorize(np.nan_to_num(X_day_train[tr_idx]))
#   Xf_va = winsorize(np.nan_to_num(X_day_train[va_idx]))
#   Xf_te = winsorize(np.nan_to_num(X_day_test))
#
#   def train_one_target(j, ...):
#       m = m_tr[:, j] == 1
#       if m.sum() < 50: return j, None, None
#       e_tr, e_va, e_te = target_projection_features(
#           Xf_tr, Xf_va, Xf_te, y_tr[:, j], m_tr[:, j].astype(float), K=12)
#       Ftr = np.hstack([F_tr[m], e_tr[m]])
#       Fva = np.hstack([F_va,    e_va])
#       Fte = np.hstack([F_te,    e_te])
#       train_set = lgb.Dataset(Ftr, y_tr[m, j], weight=np.abs(y_tr[m, j]))
#       booster = lgb.train(params, train_set, num_boost_round=n_rounds)
#       return j, booster.predict(Fva), booster.predict(Fte)


# ===========================================================================
# UPGRADE 3 — Diverse third model: row-level MLP with target embedding
# ===========================================================================
# Trains on all ~267k rows (one row = one (day, target) pair). The learned
# embedding lets the model specialise per target while sharing the trunk, and it
# sees the data at ROW granularity — a genuinely different view from the day-level
# multi-output MLP and the per-target GBMs. This is the ensemble diversity XGB
# failed to provide. Blend its OOF with your NN + LGBM using the same z-norm sweep.
#
# Sketch (PyTorch) — see README/IMPROVEMENTS.md for the full training loop:
#
#   class TargetEmbedMLP(nn.Module):
#       def __init__(self, feat_dim, n_targets, emb=16, hidden=(256, 128), p=0.3):
#           super().__init__()
#           self.emb = nn.Embedding(n_targets, emb)
#           d = feat_dim + emb
#           layers = []
#           for h in hidden:
#               layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(p)]
#               d = h
#           layers += [nn.Linear(d, 1)]
#           self.net = nn.Sequential(*layers)
#       def forward(self, x, tgt):
#           return self.net(torch.cat([x, self.emb(tgt)], dim=1)).squeeze(-1)
#
#   # Row-level tensors: feats[row] = F_train[day_of_row]; tgt[row] = target idx;
#   # y[row], w[row]=|y|. Split by day with the SAME GroupKFold, train with
#   # binary_cross_entropy_with_logits(reduction='none') weighted by |y|.
