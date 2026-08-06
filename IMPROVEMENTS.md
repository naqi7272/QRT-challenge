# Can this score higher than 0.7471? — an honest assessment + concrete upgrades

## Short answer

Yes, but modestly, and only from **new signal** — not from more models or more tuning
on the same features. Your own experiments already prove you are near this feature
set's noise ceiling:

- **Seed-averaged NN:** OOF **+9 bps**, LB **−5 bps**.
- **Adding XGBoost (3-way blend):** OOF −3 bps, LB −2 bps.

When OOF improves but LB drops, you are fitting noise. XGBoost hurt because it is
almost perfectly correlated with LightGBM (same features, same tree inductive bias),
so it added variance without diversity. The lesson: chase *decorrelated signal*, not
more estimators.

Realistic headroom on the public LB is single-digit basis points per idea. The three
upgrades below are ordered by expected value and, importantly, by how defensible they
are in a QRT interview.

---

## Upgrade 1 — Transductive unsupervised features (safest)

**Idea.** Fit PCA and feature normalisation on the **train + test** illiquid panels
stacked together. The test set is 1,177 unlabeled days of the *same* 100 illiquid
assets — free data for the factor model. It uses only `X`, never `y`, so there is
zero label leakage.

**Why it helps.** Your PCA factors and z-scores are currently estimated from 2,748
days; stacking gives ~3,925. A better factor basis and more stable normalisation,
especially for the fat-tailed and frequently-missing assets.

**Bonus:** winsorise the raw returns at ±15% before the NN. Fat tails destabilise
gradient descent; trees are invariant to monotone transforms so LGBM is unaffected.

**Expected:** +2–5 bps. **Risk:** very low. **Interview line:** *"I used the unlabeled
test features transductively for the unsupervised parts of the pipeline — legitimate
because it never touches the target."*

Code: `improvements.py :: fit_transductive`, `winsorize`.

---

## Upgrade 2 — Per-target linear-projection features for the GBMs (biggest real gain)

**The gap.** Today every one of your 100 per-target LightGBMs sees the **same** generic
364-dim vector. The only target-specific thing is which `y` column it fits. So each
tree model must rediscover, from 364 columns and only ~2,200 samples, which handful of
illiquid assets actually drive *this* target — and then approximate a *weighted sum* of
them, which is exactly what tree ensembles are worst at.

**The fix.** Inside each CV fold, on training days only:
1. correlate each illiquid asset with target `j`,
2. take the top-K (K≈12) by |corr|,
3. feed the model (a) those K raw returns, (b) **`beta_pred = Σ βₖ·retₖ`** — the
   generalised QRT benchmark, i.e. the linear map `η` the challenge is literally about,
   and (c) a correlation-signed sign-vote.

Feature (b) is the important one: you are handing the GBM the linear projection it
cannot easily build itself. This is the change most likely to add genuinely new signal.

**Leakage.** Correlations use `y`, so they are computed **per fold on training days only**
and applied unchanged to val/test. The provided function enforces this by signature.

**Expected:** +5–15 bps *if* generic features currently underserve per-target
specificity (likely, since LGBM already beats the NN 0.741 vs 0.731 with generic
features — it's starved of target-specific structure). **Risk:** medium — must re-run
the label-permutation leak check after adding it.

Code: `improvements.py :: target_projection_features` (+ wiring comment).

---

## Upgrade 3 — A genuinely diverse third model (replace XGBoost)

**Idea.** A **row-level MLP with a learned target embedding**, trained on all ~267k
rows. `forward(day_features, target_id) → 1 logit`. It sees the data at row
granularity with an explicit per-asset embedding — a different view from both the
day-level multi-output NN and the per-target GBMs.

**Why it beats XGB in the blend.** XGB duplicated LGBM. This model's errors are
decorrelated from both existing members (different granularity, different
target-handling, gradient-based), which is the only thing that makes an ensemble
member worth adding. Blend its OOF with your NN + LGBM using the same z-norm sweep.

**Expected:** +3–8 bps on the blend. **Risk:** medium (it's a new training loop).

Code sketch: `improvements.py :: UPGRADE 3` block.

---

## Things NOT worth doing (and why)

- **More boosting rounds / bigger NN / more seeds.** You're at the noise floor; this
  buys OOF, not LB. Proven by your own seed-averaging result.
- **Per-target decision thresholds tuned on OOF.** Near-balanced classes + tiny
  effective N → this overfits the OOF and reliably disappoints on LB.
- **A stacking meta-learner.** With ~2,748 effective samples the meta-model overfits
  the OOF matrix. The flat z-norm blend curve (0.20–0.45) you already found is the
  robust choice; keep it.

---

## How to validate (do this for every change, in order)

1. Keep the **exact same GroupKFold-by-`ID_DAY`** splits.
2. Measure OOF weighted accuracy with your existing `weighted_accuracy`.
3. Re-run the **label-permutation leak check** — must stay ≈0.51 (especially after
   Upgrade 2, which touches `y` during feature construction).
4. Only then submit to the LB. Adopt the change **only if OOF and LB both move up.**
   If they disagree, trust the LB and revert.

---

## Status / blocker

These are written against your `final_solution.py` objects and are drop-in, but they
are **unvalidated here** — this environment has no ML stack and, more importantly, the
shared Drive folder is missing `X_train_itDkypA.csv` and `X_test_Beg4ey3.csv` (the
illiquid-return features). Add those two files and I can benchmark each upgrade's OOF
delta before you spend an LB submission.
