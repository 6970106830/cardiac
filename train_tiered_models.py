import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc
C=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
def load(f):
    d=pd.read_csv(f"heartdisease/processed.{f}.data",header=None,names=C,na_values="?"); d["y"]=(d.num>0).astype(int); d.loc[d.chol==0,"chol"]=np.nan
    d["restecg"]=(d.restecg>0).astype(int)  # collapse ST-T(1)+LVH(2) -> abnormal: only 1 dev patient has ST-T, its dummy was memorized noise
    return d
cl=load("cleveland"); hu=load("hungarian")
dev,test=train_test_split(cl,test_size=0.25,stratify=cl.y,random_state=42); yd=dev.y.values
TI={"A":["age","sex","cp","trestbps"],"B":["age","sex","cp","trestbps","restecg"],"C":["age","sex","cp","trestbps","restecg","chol","fbs"]}
CAT=["cp","restecg"]
def mk(cols):
    return make_pipeline(ColumnTransformer([("cat",make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown="ignore")),[c for c in cols if c in CAT]),("num",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),[c for c in cols if c not in CAT])]),LogisticRegression(max_iter=2000))
def thr_for(y,p,t=.95):
    b=p.min()
    for th in np.unique(p):
        if y[p>=th].sum()/y.sum()>=t: b=th
    return b
def m(y,ref):
    fn=(~ref&(y==1)).sum(); tn=(~ref&(y==0)).sum(); return dict(sens=round(1-fn/y.sum(),2),out=round((~ref).mean(),2),npv=round(tn/(tn+fn),2) if tn+fn else None)
def cal_slope(y,p,n_bins=5):
    # calibration slope: logistic fit of y on logit(p); <1 = predictions too extreme (overfit)
    p=np.clip(p,1e-6,1-1e-6); z=np.log(p/(1-p))
    return LogisticRegression(C=np.inf,max_iter=1000).fit(z.reshape(-1,1),y).coef_[0,0]
out={}
for k,cols in TI.items():
    p=np.mean([cross_val_predict(mk(cols),dev[cols],yd,cv=StratifiedKFold(5,shuffle=True,random_state=s),method="predict_proba")[:,1] for s in range(10)],axis=0)
    thr=thr_for(yd,p); f=mk(cols).fit(dev[cols],yd)
    ct=f.named_steps["columntransformer"]; lr=f.named_steps["logisticregression"]
    Z=ct.transform(dev[cols]); Z=Z.toarray() if hasattr(Z,"toarray") else Z
    names=[n.split("__")[1] for n in ct.get_feature_names_out()]
    sc=ct.named_transformers_["num"].named_steps["standardscaler"]
    numc=[c for c in cols if c not in CAT]
    pt=f.predict_proba(test[cols])[:,1]; ph=f.predict_proba(hu[cols])[:,1]
    def pra(y,pp):
        pr,re=_=precision_recall_curve(y,pp)[:2]; return round(sk_auc(re,pr),3)
    out[k]=dict(cols=cols,names=names,coef=lr.coef_[0].round(6).tolist(),b=round(float(lr.intercept_[0]),6),zmean=Z.mean(0).round(6).tolist(),
        num=numc,mean=sc.mean_.round(6).tolist(),scale=sc.scale_.round(6).tolist(),
        median={c:float(dev[c].median()) for c in numc},thr=round(float(thr),4),
        auc_cv=round(roc_auc_score(yd,p),3),auc_test=round(roc_auc_score(test.y,pt),3),auc_hu=round(roc_auc_score(hu.y,ph),3),
        pr_auc_test=pra(test.y.values,pt),pr_auc_hu=pra(hu.y.values,ph),
        cal_slope_cv=round(float(cal_slope(yd,p)),3),cal_slope_test=round(float(cal_slope(test.y.values,pt)),3))
    print(k,"thr",out[k]["thr"],"AUC cv/test/hu",out[k]["auc_cv"],out[k]["auc_test"],out[k]["auc_hu"],
          "PR-AUC test/hu",out[k]["pr_auc_test"],out[k]["pr_auc_hu"],"cal slope cv/test",out[k]["cal_slope_cv"],out[k]["cal_slope_test"])
    for cut in [thr,0.35]:
        print(f"   cut {cut:.2f} dev",m(yd,p>=cut),"test",m(test.y.values,pt>=cut),"hu",m(hu.y.values,ph>=cut))
    print("   >=0.70 share dev",(p>=.7).mean().round(2),"PPV",yd[p>=.7].mean().round(2))
json.dump(out,open("models.json","w"))
