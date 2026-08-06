import numpy as np
from sklearn.model_selection import GroupKFold
from ridge_addon import fit_ridge_oof

rng=np.random.default_rng(0)
ND,NA,NL,NF=500,100,8,5
Xf_all=np.clip(rng.standard_normal((ND,NA))*0.02,-0.15,0.15).astype(np.float32)
Xf_test=np.clip(rng.standard_normal((150,NA))*0.02,-0.15,0.15).astype(np.float32)
# each target = linear combo of a few illiquids
W=rng.standard_normal((NA,NL))*(rng.random((NA,NL))<0.05)
Y=(Xf_all@W + rng.standard_normal((ND,NL))*0.003).astype(np.float32)
mask=np.ones((ND,NL),np.float32)
days=np.arange(ND)
gkf=GroupKFold(n_splits=NF)
def wacc(s,y,m):
    ps=np.where(s>0,1.,-1.); ts=np.sign(y); ts[ts==0]=1
    c=(ps==ts).astype(np.float32)*m
    return (c*np.abs(y)).sum()/((m*np.abs(y)).sum()+1e-8)

r_oof,r_test=fit_ridge_oof(gkf,Xf_all,Xf_test,Y,mask,days,NL,NF)
print("shapes",r_oof.shape,r_test.shape)
print("Ridge-only OOF (real linear targets):",round(wacc(r_oof,Y,mask),4))
# leak check: permute day labels -> OOF must collapse to ~0.5
perm=rng.permutation(ND); Yp=Y[perm]
rp_oof,_=fit_ridge_oof(gkf,Xf_all,Xf_test,Yp,mask,days,NL,NF)
print("Ridge-only OOF (permuted labels):    ",round(wacc(rp_oof,Yp,mask),4))
assert r_oof.shape==(ND,NL) and r_test.shape==(150,NL)
assert wacc(r_oof,Y,mask)>0.7, "should recover strong linear signal"
assert abs(wacc(rp_oof,Yp,mask)-0.5)<0.06, "permuted must be ~0.5 (leak-safe)"
print("\nRIDGE ADD-ON OK")
