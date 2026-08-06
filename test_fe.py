import numpy as np
from feature_experiments import diagnose_temporal_order, add_rank_features, sector_mean_impute
rng=np.random.default_rng(0); ND,NA=800,100

# --- A: diagnostic must distinguish ordered (vol-clustering) vs shuffled ---
# ordered: GARCH-ish vol clustering
logv=np.zeros(ND)
for t in range(1,ND): logv[t]=0.9*logv[t-1]+rng.standard_normal()*0.3   # persistent log-vol
vol=0.02*np.exp(logv)
Xord=(rng.standard_normal((ND,NA))*vol[:,None]).astype(np.float32)
ids=np.arange(ND)
r_ord=diagnose_temporal_order(Xord, ids)
# shuffled: same rows, permuted day order -> vol clustering destroyed
perm=rng.permutation(ND)
r_shuf=diagnose_temporal_order(Xord[perm], ids)   # ids consecutive but content shuffled
print()
assert r_ord["ordered"] is True,  "GARCH panel should read ORDERED"
assert r_shuf["ordered"] is False, "shuffled panel should read SHUFFLED"
print("A diagnostic OK  (ordered vol-ac=%.3f  shuffled vol-ac=%.3f)"%(r_ord["ac_vol"],r_shuf["ac_vol"]))

# --- B: rank features in [-1,1], missing->0, monotone within a day ---
X=rng.standard_normal((5,NA)).astype(np.float32); X[X>1.5]=np.nan
nan=np.isnan(X).astype(np.float32); Xf=np.nan_to_num(X)
R=add_rank_features(Xf,nan)
assert R.shape==(5,NA) and R.min()>=-1.0001 and R.max()<=1.0001
# largest observed return in a row must get the top rank among observed
d=0; obs=np.where(nan[d]<0.5)[0]
assert obs[np.argmax(Xf[d,obs])]==obs[np.argmax(R[d,obs])]
print("B rank features OK  range[%.2f,%.2f]"%(R.min(),R.max()))

# --- C: sector-mean impute fills NaNs, no NaNs remain, observed untouched ---
sec=np.repeat(np.arange(10),10)  # 10 sectors x 10 assets
Xc=rng.standard_normal((6,NA)).astype(np.float32); Xc[rng.random(Xc.shape)<0.2]=np.nan
Xi,mask=sector_mean_impute(Xc, sec)
assert not np.isnan(Xi).any()
obsmask=~np.isnan(Xc)
assert np.allclose(Xi[obsmask], np.nan_to_num(Xc)[obsmask])
print("C sector-mean impute OK  (filled %d NaNs)"%int(np.isnan(Xc).sum()))
print("\nALL FE CHECKS PASSED")
