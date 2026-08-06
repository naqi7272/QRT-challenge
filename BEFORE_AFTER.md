# Before → After: the three upgrades against `final_solution.py`

Each block shows your current code (**Before**) and the drop-in change (**After**).
Adopt them one at a time; after each, check GroupKFold OOF **and** the LB, and re-run
the label-permutation leak check (especially after Upgrade 2).

Status: Upgrades 1 & 2 are verified for correctness + leak-safety in
`test_improvements.py` (synthetic data, passes here). Upgrade 3 is a working sketch
(needs torch + your data to tune).

---

## Upgrade 1 — Transductive unsupervised features (§5, end of Feature engineering)

**Why:** fit PCA + normalisation on **train + test** illiquid panels (test = 1,177
extra unlabeled days of the *same* assets). Uses only `X`, never `y` → zero leakage.
Plus winsorise raw returns so the NN isn't destabilised by fat tails (trees are
invariant, so LGBM is unaffected). *Expected +2–5 bps, lowest risk.*

### Before
```python
fb = FeatureBuilder(supp, illiquid_ids, liquid_ids, n_pca=CFG["n_pca"])
fb.fit(X_day_train)
F_train = fb.transform(X_day_train)
F_test = fb.transform(X_day_test)
print(f"Engineered feature dim: {F_train.shape[1]}")

mu = F_train.mean(axis=0, keepdims=True)
sd = F_train.std(axis=0, keepdims=True) + 1e-6
F_train = (F_train - mu) / sd
F_test = (F_test - mu) / sd
```

### After
```python
fb = FeatureBuilder(supp, illiquid_ids, liquid_ids, n_pca=CFG["n_pca"])

def _winsorize(X, lim=0.15):
    return np.clip(X, -lim, lim)

# winsorise the filled-return block the models see (patch the imputer once)
_orig_impute = fb._impute_and_indicator
def _impute_winsor(X_day):
    filled, ind = _orig_impute(X_day)
    return _winsorize(filled), ind
fb._impute_and_indicator = _impute_winsor

# fit PCA transductively on train+test stacked (label-free -> no leakage)
_tr_filled, _ = fb._impute_and_indicator(X_day_train)
_te_filled, _ = fb._impute_and_indicator(X_day_test)
fb.pca = PCA(n_components=CFG["n_pca"], random_state=SEED).fit(
    np.vstack([_tr_filled, _te_filled]))

F_train = fb.transform(X_day_train)
F_test  = fb.transform(X_day_test)
print(f"Engineered feature dim: {F_train.shape[1]}")

# normalise with train+test statistics (still label-free)
_both = np.vstack([F_train, F_test])
mu = _both.mean(axis=0, keepdims=True)
sd = _both.std(axis=0, keepdims=True) + 1e-6
F_train = (F_train - mu) / sd
F_test  = (F_test  - mu) / sd
```

---

## Upgrade 2 — Per-target linear-projection features for the GBMs (§10)

**Why:** today all 100 per-target LightGBMs see the *same* generic 364-dim vector, so
each must rediscover which few illiquid assets drive *its* target from 364 columns and
~2,200 rows — and then approximate a weighted sum, which trees do poorly. This hands
each model its top-K correlated illiquids, the sign-aligned versions, **and
`beta_pred = Σ βₖ·retₖ`** (the linear map η the challenge is about). Correlations use
`y`, so they are computed **per fold on training rows only**. *Biggest expected real
gain (+5–15 bps); re-run the leak check after adding it.*

### Before
```python
def train_one_target(j, F_tr, y_tr, m_tr, F_va, F_te, params, n_rounds):
    """Train LGBM for one target, return (j, val_pred, test_pred)."""
    m = m_tr[:, j] == 1
    if m.sum() < 50:
        return j, None, None
    weights = np.abs(y_tr[m, j])
    train_set = lgb.Dataset(F_tr[m], y_tr[m, j], weight=weights)
    booster = lgb.train(params, train_set, num_boost_round=n_rounds)
    return j, booster.predict(F_va), booster.predict(F_te)

# ... inside the fold loop:
    F_tr_fold = F_train[tr_idx]
    F_va_fold = F_train[va_idx]
    y_tr_fold = y_panel_filled[tr_idx]
    m_tr_fold = mask_panel[tr_idx]

    results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
        delayed(train_one_target)(
            j, F_tr_fold, y_tr_fold, m_tr_fold,
            F_va_fold, F_test, LGBM_PARAMS_PARALLEL, N_ROUNDS
        )
        for j in range(N_LIQUID)
    )
```

### After
```python
# --- one-time: winsorised, zero-filled RAW illiquid panels, day-aligned to F_* ---
Xf_all  = _winsorize(np.nan_to_num(X_day_train))   # (n_days_train, 100)
Xf_test = _winsorize(np.nan_to_num(X_day_test))    # (n_days_test,  100)

def _proj_features(Xf_fit, Xf_apply, y_fit_col, m_fit_col, K=12):
    """Top-K |corr| raw returns + sign-aligned copies + beta_pred (linear map)
       + sign-vote. Stats estimated ONLY on fit rows -> leak-safe."""
    idx = m_fit_col > 0.5
    Xo, yo = Xf_fit[idx], y_fit_col[idx]
    if Xo.shape[0] < 30:
        return np.zeros((Xf_apply.shape[0], 2 * K + 2), np.float32)
    Xmu, Xsd = Xo.mean(0), Xo.std(0) + 1e-8
    ymu, ysd = yo.mean(), yo.std() + 1e-8
    corr = (((Xo - Xmu) / Xsd) * ((yo - ymu) / ysd)[:, None]).mean(0)
    beta = corr * (ysd / Xsd)
    top  = np.argsort(-np.abs(corr))[:K]
    sign = np.sign(corr[top])
    raw       = Xf_apply[:, top]
    beta_pred = (Xf_apply[:, top] * beta[top]).sum(1, keepdims=True)
    sign_vote = (np.sign(raw) * sign).mean(1, keepdims=True)
    return np.concatenate([raw, raw * sign, beta_pred, sign_vote],
                          axis=1).astype(np.float32)

def train_one_target(j, F_tr, y_tr, m_tr, F_va, F_te,
                     Xf_tr, Xf_va, Xf_te, params, n_rounds):
    m = m_tr[:, j] == 1
    if m.sum() < 50:
        return j, None, None
    # per-target features fit on THIS fold's train rows only
    e_tr = _proj_features(Xf_tr, Xf_tr, y_tr[:, j], m_tr[:, j], K=12)
    e_va = _proj_features(Xf_tr, Xf_va, y_tr[:, j], m_tr[:, j], K=12)
    e_te = _proj_features(Xf_tr, Xf_te, y_tr[:, j], m_tr[:, j], K=12)
    Ftr = np.hstack([F_tr, e_tr])[m]
    Fva = np.hstack([F_va, e_va])
    Fte = np.hstack([F_te, e_te])
    train_set = lgb.Dataset(Ftr, y_tr[m, j], weight=np.abs(y_tr[m, j]))
    booster = lgb.train(params, train_set, num_boost_round=n_rounds)
    return j, booster.predict(Fva), booster.predict(Fte)

# ... inside the fold loop, add the raw-panel slices and pass them through:
    F_tr_fold = F_train[tr_idx]
    F_va_fold = F_train[va_idx]
    y_tr_fold = y_panel_filled[tr_idx]
    m_tr_fold = mask_panel[tr_idx]
    Xf_tr_fold = Xf_all[tr_idx]        # NEW
    Xf_va_fold = Xf_all[va_idx]        # NEW

    results = Parallel(n_jobs=-1, backend="loky", verbose=1)(
        delayed(train_one_target)(
            j, F_tr_fold, y_tr_fold, m_tr_fold, F_va_fold, F_test,
            Xf_tr_fold, Xf_va_fold, Xf_test,          # NEW
            LGBM_PARAMS_PARALLEL, N_ROUNDS
        )
        for j in range(N_LIQUID)
    )
```

---

## Upgrade 3 — Replace XGBoost with a target-embedding row-level MLP (§"XGB as a Third model")

**Why:** your XGB was ~a clone of LightGBM (same features, same tree bias), so it added
no diversity and cost 2 bps. A row-level MLP with a learned target embedding sees the
data at a genuinely different granularity (one row = one `(day, target)` pair) and
learns per-target specialisation via the embedding — the decorrelated third member the
blend actually wants. *Expected +3–8 bps on the blend; least-validated — tune on your
data.*

### Before
```python
import xgboost as xgb
XGB_PARAMS = dict(objective="reg:absoluteerror", learning_rate=0.03, max_depth=5, ...)
XGB_ROUNDS = 200
xgb_oof = np.zeros_like(nn_oof); xgb_test = np.zeros_like(nn_test)

def train_one_xgb(j, F_tr, y_tr, m_tr, F_va, F_te):
    ...
# ... per-fold Parallel loop, then:
xgb_oof_z = z_norm(xgb_oof); xgb_test_z = z_norm(xgb_test)
# three-way blend NN + LGBM + XGB
```

### After
```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TargetEmbedMLP(nn.Module):
    def __init__(self, feat_dim, n_targets, emb=16, hidden=(256, 128), p=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_targets, emb)
        d = feat_dim + emb; layers = []
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(p)]; d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, t):
        return self.net(torch.cat([x, self.emb(t)], dim=1)).squeeze(-1)

def _make_rows(day_idx, F, yp, mp):
    xs, ts, ys = [], [], []
    for di in day_idx:
        for j in np.where(mp[di] == 1)[0]:
            xs.append(F[di]); ts.append(j); ys.append(yp[di, j])
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(ts, dtype=torch.long),
            torch.tensor(ys, dtype=torch.float32))

emb_oof  = np.zeros_like(nn_oof)
emb_test = np.zeros_like(nn_test)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(F_train, groups=day_ids_train)):
    Xr, Tr, Yr = _make_rows(tr_idx, F_train, y_panel_filled, mask_panel)
    loader = DataLoader(TensorDataset(Xr, Tr, Yr), batch_size=4096, shuffle=True)
    model = TargetEmbedMLP(F_train.shape[1], N_LIQUID).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(25):
        model.train()
        for xb, tb, yb in loader:
            xb, tb, yb = xb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logit = model(xb, tb)
            w = yb.abs()
            loss = (nn.functional.binary_cross_entropy_with_logits(
                        logit, (yb > 0).float(), reduction="none") * w).sum() / (w.sum() + 1e-8)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        allt = torch.arange(N_LIQUID, device=DEVICE)
        for di in va_idx:
            x = torch.tensor(F_train[di], device=DEVICE).repeat(N_LIQUID, 1)
            emb_oof[di] = model(x, allt).cpu().numpy()
        for di in range(len(F_test)):
            x = torch.tensor(F_test[di], device=DEVICE).repeat(N_LIQUID, 1)
            emb_test[di] += model(x, allt).cpu().numpy() / CFG["n_folds"]
    print(f"emb-MLP fold {fold+1} done")

print(f"emb-MLP OOF: {weighted_acc(emb_oof, y_panel_filled, mask_panel):.5f}")

emb_oof_z, emb_test_z = z_norm(emb_oof), z_norm(emb_test)

# 3-way blend: NN + LGBM + emb-MLP (replaces XGB)
best, bwn, bwl = -1.0, 0.33, 0.33
for w_nn in np.linspace(0, 1, 11):
    for w_lgbm in np.linspace(0, 1 - w_nn, 11):
        w_e = 1 - w_nn - w_lgbm
        if w_e < 0: continue
        s = weighted_acc(w_nn * nn_oof_z + w_lgbm * lgbm_oof_z + w_e * emb_oof_z,
                         y_panel_filled, mask_panel)
        if s > best: best, bwn, bwl = s, w_nn, w_lgbm
print(f"Best 3-way (NN+LGBM+embMLP): w_nn={bwn:.2f} w_lgbm={bwl:.2f} "
      f"w_emb={1 - bwn - bwl:.2f} OOF={best:.5f}")
```

---

## Validation protocol (unchanged, per upgrade)

1. Same `GroupKFold(groups=day_ids_train)` splits → report OOF weighted accuracy.
2. Re-run the label-permutation leak check → must stay ≈0.51 (critical after Upgrade 2).
3. Submit; keep the change only if **OOF and LB both rise**. If they disagree, trust LB.
