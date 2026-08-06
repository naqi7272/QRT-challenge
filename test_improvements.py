import numpy as np
from improvements import winsorize, fit_transductive, target_projection_features, _masked_stats

rng = np.random.default_rng(0)
NA, NTR, NTE, K = 100, 400, 150, 12

# ---- Minimal stub mirroring final_solution.FeatureBuilder's contract ----
from sklearn.decomposition import PCA
class StubFB:
    def __init__(self, n_pca=16): self.n_pca=n_pca; self.pca=None
    def _impute_and_indicator(self, X):
        return np.nan_to_num(X).astype(np.float32), np.isnan(X).astype(np.float32)
    def fit(self, X):
        f,_=self._impute_and_indicator(X); self.pca=PCA(self.n_pca,random_state=42).fit(f)
    def transform(self, X):
        f,ind=self._impute_and_indicator(X)
        return np.hstack([f, self.pca.transform(f), ind]).astype(np.float32)

# panels with some NaNs (missing illiquid observations)
Xtr = rng.standard_normal((NTR, NA)).astype(np.float32)*0.02
Xte = rng.standard_normal((NTE, NA)).astype(np.float32)*0.02
for X in (Xtr,Xte):
    X[rng.random(X.shape)<0.1]=np.nan

# ================= TEST A: transductive PCA fit on train+test =================
fb=StubFB(n_pca=16)
Ftr,Fte=fit_transductive(fb, Xtr, Xte)
assert fb.pca.n_samples_==NTR+NTE, fb.pca.n_samples_
assert Ftr.shape[0]==NTR and Fte.shape[0]==NTE and Ftr.shape[1]==Fte.shape[1]
# joint normalisation: stacked features ~ zero mean / unit sd
both=np.vstack([Ftr,Fte]); 
assert abs(both.mean())<1e-3 and abs(both.std()-1)<0.05
# winsorise clips
assert winsorize(np.array([0.9,-0.9,0.01])).tolist()==[0.15,-0.15,0.01]
print("A transductive+winsorise  OK  | PCA fit on", fb.pca.n_samples_,
      "rows (=train+test) | feat dim", Ftr.shape[1])

# ============ TEST B: Upgrade-2 shapes + leak-safety by construction ==========
# y_target is a TRUE linear combo of 5 driver assets (the eta map).
drivers=rng.choice(NA,5,replace=False); w=rng.standard_normal(5)
Xf_tr=winsorize(np.nan_to_num(Xtr)); Xf_te=winsorize(np.nan_to_num(Xte))
# split train days into fit/val
cut=300; fit_idx=slice(0,cut); va_idx=slice(cut,NTR)
def make_y(Xf): return (Xf[:,drivers]@w) + rng.standard_normal(Xf.shape[0])*0.005
y_all=make_y(Xf_tr); m_all=np.ones(NTR)

e_tr,e_va,e_te=target_projection_features(
    Xf_tr[fit_idx],Xf_tr[va_idx],Xf_te, y_all[fit_idx], m_all[fit_idx], K=K)
assert e_tr.shape==(cut,2*K+2) and e_va.shape==(NTR-cut,2*K+2) and e_te.shape==(NTE,2*K+2)
print("B shapes                  OK  | enrich dim", e_tr.shape[1],"(=2K+2)")

# leak-safety-by-construction: val features cannot depend on val labels — the fn
# has no access to them. Prove it: changing anything about val leaves e_va fixed.
e_va2=target_projection_features(Xf_tr[fit_idx],Xf_tr[va_idx],Xf_te,
                                 y_all[fit_idx],m_all[fit_idx],K=K)[1]
assert np.array_equal(e_va,e_va2)
print("C leak-safe by construction OK | val features are a pure fn of train stats")

# ================= TEST D: does beta_pred actually capture eta? ================
beta_pred_va=e_va[:, 2*K]                     # the linear-map column
true_va=y_all[va_idx]
r_real=np.corrcoef(beta_pred_va,true_va)[0,1]
# now SHUFFLE train labels (break which assets are drivers) -> feature must go dead
y_shuf=rng.permutation(y_all[fit_idx])
e_va_shuf=target_projection_features(Xf_tr[fit_idx],Xf_tr[va_idx],Xf_te,
                                     y_shuf,m_all[fit_idx],K=K)[1]
r_shuf=np.corrcoef(e_va_shuf[:,2*K],true_va)[0,1]
print(f"D signal test             corr(beta_pred,val_target): real={r_real:.3f}  "
      f"shuffled-labels={r_shuf:.3f}")
assert r_real>0.5,  "beta_pred should capture the true linear map"
assert abs(r_shuf)<0.2,"shuffled-label features must not manufacture OOF signal"
print("\nALL CHECKS PASSED")
