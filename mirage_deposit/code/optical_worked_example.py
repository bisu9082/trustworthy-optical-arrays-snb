"""Canonical optical worked-example analysis for the SnAB review (R2-4 resolution).
Fully specified: experimental/resampling unit = one 29-dye Delta-E vector per
(panel, agent, concentration, illumination); concentration-0 blanks EXCLUDED from
the closed-set agent-classification (an all-zero blank cannot be assigned to an
agent and belongs to the reject/open-set category). RF(300 trees), StandardScaler
fit on train only, StratifiedKFold(4) within-illumination, 10 seeds. No Cohen's d
(ill-defined without physical replicates); inference via label-permutation test;
uncertainty via bootstrap over the condition-vector resampling unit.
"""
import csv, glob, os, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
DATA=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data","optical_deltaE"); SEEDS=list(range(10)); NTREE=300
def load(illum):
    X,y,c,p=[],[],[],[]
    for f in sorted(glob.glob(f"{DATA}/d*.{illum}.csv")):
        pan=os.path.basename(f).split(".")[0]
        for row in list(csv.reader(open(f)))[1:]:
            cc=int(row[1])
            if cc==0: continue                      # exclude blanks
            X.append([float(v) for v in row[2:]]); y.append(f"{pan}:{row[0]}"); c.append(cc); p.append(pan)
    return np.array(X),np.array(y),np.array(c),np.array(p)
Xn,yn,cn,pn=load("normal"); Xu,yu,cu,pu=load("uv")
assert list(yn)==list(yu) and list(cn)==list(cu)
K=len(set(yn))
def within_oof(X,y,seed):
    skf=StratifiedKFold(4,shuffle=True,random_state=seed); yp=np.empty_like(y)
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr]); rf=RandomForestClassifier(n_estimators=NTREE,random_state=seed,n_jobs=1).fit(sc.transform(X[tr]),y[tr]); yp[te]=rf.predict(sc.transform(X[te]))
    return yp
def cross_pred(Xtr,ytr,Xte,seed):
    sc=StandardScaler().fit(Xtr); rf=RandomForestClassifier(n_estimators=NTREE,random_state=seed,n_jobs=1).fit(sc.transform(Xtr),ytr); return rf.predict(sc.transform(Xte))
def acc_seeds(fn): return np.array([fn(s) for s in SEEDS])

R={"meta":{"K":int(K),"chance":1/K,"n_per_illum":int(len(yn)),"seeds":len(SEEDS),"ntree":NTREE,
   "unit":"one 29-dye deltaE vector per (panel,agent,concentration,illumination); conc-0 blanks excluded",
   "panels":{p:int((pn==p).sum()) for p in ["d1","d2","d3"]}}}
# within per illumination (pooled)
R["within_pooled_normal"]=acc_seeds(lambda s: accuracy_score(yn,within_oof(Xn,yn,s)))
R["within_pooled_uv"]=acc_seeds(lambda s: accuracy_score(yu,within_oof(Xu,yu,s)))
# within per panel (both illum)
R["within_panel"]={}
for pan in ["d1","d2","d3"]:
    mn=pn==pan; mu=pu==pan
    R["within_panel"][pan]={
      "normal":acc_seeds(lambda s,mn=mn: accuracy_score(yn[mn],within_oof(Xn[mn],yn[mn],s))),
      "uv":acc_seeds(lambda s,mu=mu: accuracy_score(yu[mu],within_oof(Xu[mu],yu[mu],s))),
      "chance":1/len(set(yu[mu]))}
# cross both directions (pooled + per panel)
R["cross_amb2uv"]=acc_seeds(lambda s: accuracy_score(yu,cross_pred(Xn,yn,Xu,s)))
R["cross_uv2amb"]=acc_seeds(lambda s: accuracy_score(yn,cross_pred(Xu,yu,Xn,s)))
R["cross_panel_amb2uv"]={}
for pan in ["d1","d2","d3"]:
    mn=pn==pan; mu=pu==pan
    R["cross_panel_amb2uv"][pan]=acc_seeds(lambda s,mn=mn,mu=mu: accuracy_score(yu[mu],cross_pred(Xn[mn],yn[mn],Xu[mu],s)))
# LOCO pooled
def loco(X,y,c,seed):
    yp=np.empty_like(y)
    for cc in sorted(set(c)):
        te=c==cc; tr=~te; sc=StandardScaler().fit(X[tr]); rf=RandomForestClassifier(n_estimators=NTREE,random_state=seed,n_jobs=1).fit(sc.transform(X[tr]),y[tr]); yp[te]=rf.predict(sc.transform(X[te]))
    return accuracy_score(y,yp)
R["loco_uv"]=acc_seeds(lambda s: loco(Xu,yu,cu,s)); R["loco_normal"]=acc_seeds(lambda s: loco(Xn,yn,cn,s))
# paired bootstrap CI on the drop (within-uv vs amb->uv), resampling unit = uv condition vector
s0=0; w=(within_oof(Xu,yu,s0)==yu).astype(int); cc=(cross_pred(Xn,yn,Xu,s0)==yu).astype(int)
rng=np.random.default_rng(0); n=len(yu); boots=[(w[i:=rng.integers(0,n,n)].mean()-cc[i].mean()) for _ in range(2000)]
R["drop_point"]=float(w.mean()-cc.mean()); R["drop_CI95"]=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))]
# permutation tests (single split, B=300): within-uv and cross amb->uv
def perm(get_obs_and_null,B=300,seed=0):
    rng=np.random.default_rng(seed); obs,null=get_obs_and_null(rng,B); null=np.array(null)
    return float(obs),float((np.sum(null>=obs)+1)/(B+1)),float(null.mean()),float(np.percentile(null,95))
def within_perm(rng,B):
    skf=StratifiedKFold(4,shuffle=True,random_state=0); tr,te=next(iter(skf.split(Xu,yu)))
    def fit(yt): sc=StandardScaler().fit(Xu[tr]); rf=RandomForestClassifier(n_estimators=NTREE,random_state=0,n_jobs=1).fit(sc.transform(Xu[tr]),yt); return accuracy_score(yu[te],rf.predict(sc.transform(Xu[te])))
    return fit(yu[tr]),[fit(rng.permutation(yu[tr])) for _ in range(B)]
def cross_perm(rng,B):
    def fit(yt): sc=StandardScaler().fit(Xn); rf=RandomForestClassifier(n_estimators=NTREE,random_state=0,n_jobs=1).fit(sc.transform(Xn),yt); return accuracy_score(yu,rf.predict(sc.transform(Xu)))
    return fit(yn),[fit(rng.permutation(yn)) for _ in range(B)]
R["perm_within_uv"]=dict(zip(["obs","p","null_mean","null_p95"],perm(within_perm)))
R["perm_cross_amb2uv"]=dict(zip(["obs","p","null_mean","null_p95"],perm(cross_perm)))

# ---- print summary ----
def ms(a): return f"{np.mean(a):.3f}±{np.std(a):.3f}"
print("K",K,"chance %.4f"%(1/K),"n/illum",len(yn),"panels",R["meta"]["panels"])
print("within pooled: normal",ms(R["within_pooled_normal"]),"uv",ms(R["within_pooled_uv"]),"mean %.3f"%np.mean(np.r_[R['within_pooled_normal'],R['within_pooled_uv']]))
for p in ["d1","d2","d3"]: print(f"  within {p}: normal {np.mean(R['within_panel'][p]['normal']):.3f} uv {np.mean(R['within_panel'][p]['uv']):.3f} (chance {R['within_panel'][p]['chance']:.3f})  cross amb->uv {np.mean(R['cross_panel_amb2uv'][p]):.3f}")
print("cross: amb->uv",ms(R["cross_amb2uv"]),"uv->amb",ms(R["cross_uv2amb"]))
print("drop point %.3f CI"%R["drop_point"],["%.3f"%x for x in R["drop_CI95"]])
print("LOCO uv",ms(R["loco_uv"]),"normal",ms(R["loco_normal"]))
print("perm within_uv",{k:round(v,4) for k,v in R["perm_within_uv"].items()})
print("perm cross_amb2uv",{k:round(v,4) for k,v in R["perm_cross_amb2uv"].items()})

# ---- save results.json (arrays -> lists) ----
def conv(o):
    if isinstance(o,np.ndarray): return [float(x) for x in o]
    if isinstance(o,dict): return {k:conv(v) for k,v in o.items()}
    if isinstance(o,(np.floating,)): return float(o)
    return o
json.dump(conv(R),open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","optical_results.json"),"w"),indent=1)

# ---- regenerate Fig6: pooled + d1 + d2 + d3, within(uv) vs cross(amb->uv) ----
groups=["Pooled (17)","d1 (6)","d2 (7)","d3 (4)"]
wvals=[np.mean(R["within_pooled_uv"])]+[np.mean(R["within_panel"][p]["uv"]) for p in ["d1","d2","d3"]]
werr=[np.std(R["within_pooled_uv"])]+[np.std(R["within_panel"][p]["uv"]) for p in ["d1","d2","d3"]]
cvals=[np.mean(R["cross_amb2uv"])]+[np.mean(R["cross_panel_amb2uv"][p]) for p in ["d1","d2","d3"]]
cerr=[np.std(R["cross_amb2uv"])]+[np.std(R["cross_panel_amb2uv"][p]) for p in ["d1","d2","d3"]]
chances=[1/K]+[R["within_panel"][p]["chance"] for p in ["d1","d2","d3"]]
x=np.arange(len(groups)); w=0.36
fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=300)
ax.bar(x-w/2,wvals,w,yerr=werr,capsize=3,color="#2e7d32",label="Within illumination (random CV)")
ax.bar(x+w/2,cvals,w,yerr=cerr,capsize=3,color="#c62828",label="Cross illumination (ambient$\\rightarrow$ultraviolet)")
for i,ch in enumerate(chances): ax.hlines(ch,x[i]-0.46,x[i]+0.46,colors="k",linestyles="--",lw=1)
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Classification accuracy"); ax.set_ylim(0,1.0)
ax.legend(frameon=False,fontsize=9,loc="upper right"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","Fig6_acquisition_transfer.png"),dpi=300); print("Fig6 saved")
