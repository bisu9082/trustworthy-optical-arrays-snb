"""R2-3: decompose the gas-sensor pooled-vs-temporal gap into (size) + (temporal/composition).
A: pooled random 5-fold CV (full training).
C: train on batch1 only -> test on each later batch.
B: size-matched control -> train on |batch1| samples drawn iid from the full pool, test on held-out pool
   (isolates the effect of the smaller training set at batch-1 size, with no temporal shift).
Gap A->B = pure training-size effect; gap B->C = temporal + composition effect at matched size.
"""
import glob, os, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import os
RAW=os.environ.get("GAS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","gas_uci_download_here")); NTREE=300
def load(b):
    X,y=[],[]
    for ln in open(f"{RAW}/batch{b}.dat"):
        t=ln.split(); y.append(int(t[0])); X.append([float(p.split(":")[1]) for p in t[1:]])
    return np.array(X),np.array(y)
B={b:load(b) for b in range(1,11)}
sizes={b:len(B[b][1]) for b in range(1,11)}
print("batch sizes:",sizes,"total",sum(sizes.values()))
Xall=np.vstack([B[b][0] for b in range(1,11)]); yall=np.concatenate([B[b][1] for b in range(1,11)])
n1=sizes[1]; K=len(set(yall)); print(f"classes {K} chance {1/K:.3f}  batch1 size {n1}")

def rf(seed): return RandomForestClassifier(n_estimators=NTREE,random_state=seed,n_jobs=1)

# A: pooled random 5-fold CV
accA=[]
for s in range(5):
    skf=StratifiedKFold(5,shuffle=True,random_state=s); yp=np.empty_like(yall)
    for tr,te in skf.split(Xall,yall):
        sc=StandardScaler().fit(Xall[tr]); m=rf(s).fit(sc.transform(Xall[tr]),yall[tr]); yp[te]=m.predict(sc.transform(Xall[te]))
    accA.append(accuracy_score(yall,yp))
A=np.mean(accA); print(f"\nA pooled random 5-fold CV: {A:.3f} (+/-{np.std(accA):.3f})")

# C: train batch1 -> test later batches
X1,y1=B[1]; sc1=StandardScaler().fit(X1); m1=rf(0).fit(sc1.transform(X1),y1)
perC={}
for b in range(2,11):
    Xb,yb=B[b]; perC[b]=accuracy_score(yb,m1.predict(sc1.transform(Xb)))
C=np.mean(list(perC.values()))
print(f"C batch1->later: per-batch {[round(perC[b],2) for b in range(2,11)]}")
print(f"  mean {C:.3f}  min {min(perC.values()):.3f}  (next-batch b2 {perC[2]:.3f})")

# B: size-matched iid control -> train |batch1| drawn from pool, test on held-out pool
rng=np.random.default_rng(0); accB=[]
idx_all=np.arange(len(yall))
for s in range(10):
    rng=np.random.default_rng(s)
    tr=rng.choice(idx_all,size=n1,replace=False); te=np.setdiff1d(idx_all,tr)
    sc=StandardScaler().fit(Xall[tr]); m=rf(s).fit(sc.transform(Xall[tr]),yall[tr])
    accB.append(accuracy_score(yall[te],m.predict(sc.transform(Xall[te]))))
Bacc=np.mean(accB); print(f"B size-matched iid (train n={n1} from pool): {Bacc:.3f} (+/-{np.std(accB):.3f})")

# B2: size-matched AND class-proportion-matched to batch1
from collections import Counter
cnt1=Counter(y1); accB2=[]
for s in range(10):
    rng=np.random.default_rng(100+s); tr=[]
    for c,k in cnt1.items():
        pool_c=idx_all[yall==c]; tr.extend(rng.choice(pool_c,size=min(k,len(pool_c)),replace=False))
    tr=np.array(tr); te=np.setdiff1d(idx_all,tr)
    sc=StandardScaler().fit(Xall[tr]); m=rf(s).fit(sc.transform(Xall[tr]),yall[tr])
    accB2.append(accuracy_score(yall[te],m.predict(sc.transform(Xall[te]))))
B2=np.mean(accB2); print(f"B2 size+class-composition-matched to batch1: {B2:.3f} (+/-{np.std(accB2):.3f})")

print(f"\n=== DECOMPOSITION ===")
print(f"A full pooled CV            {A:.3f}")
print(f"B size-matched iid          {Bacc:.3f}   (A->B size effect: {A-Bacc:+.3f})")
print(f"B2 size+composition matched {B2:.3f}   (A->B2: {A-B2:+.3f})")
print(f"C batch1->later transfer    {C:.3f}   (B->C temporal/composition effect: {Bacc-C:+.3f})")
print(f"Interpretation: of the total A-C gap {A-C:.3f}, size alone explains ~{A-Bacc:.3f}, "
      f"temporal+composition explains ~{Bacc-C:.3f}")
# class composition per batch
print("\nclass composition per batch (fraction):")
for b in range(1,11):
    yb=B[b][1]; frac=[round((yb==c).mean(),2) for c in range(1,7)]
    print(f"  batch{b} n={sizes[b]:4d}: {frac}")
import json
json.dump({"A_pooledCV":A,"B_sizematched":Bacc,"B2_size_comp_matched":B2,"C_batch1_later":C,
 "perC":perC,"min_later":min(perC.values()),"size_effect":A-Bacc,"temporal_comp_effect":Bacc-C,
 "batch1_size":n1,"K":K,"chance":1/K,"sizes":sizes},open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","gas_drift_results.json"),"w"),indent=1)
