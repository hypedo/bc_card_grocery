# -*- coding: utf-8 -*-
"""16_predict.py — 4차 시도: 다운사이징 속도 예측 모델
앞선 실패를 막는 설계:
 (1) 누출 차단 — 피처는 1~3월만, 종속변수는 4~6월만
 (2) 5월 교란 회피 — 종속변수를 '수준'이 아닌 '3채널 내 비중'으로 (전채널 동시 급등은 상쇄)
 (3) 규모 지배 방지 — 규모를 넣되 기여도를 분리 보고
 (4) 정직한 기준 — 단순 지속예측(1~3월 추세가 이어진다) 베이스라인을 이겨야 의미 있음"""
import pandas as pd, numpy as np, pathlib
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
CH=["대형할인점","슈퍼마켓","편의점"]

# ── 패널 ───────────────────────────────────────────────────────────────────
p=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index=["REGION","T"],columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
c=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index=["REGION","T"],columns="TP_BUZ_NM",values="CNT",aggfunc="sum")
ok=p.groupby("REGION").size()==6
regs=[r for r in ok[ok].index if (p.loc[r]>0).all().all()]
p,c=p.loc[regs],c.loc[regs]
sh=p.div(p.sum(axis=1),axis=0)                       # 3채널 내 금액 비중 (5월 급등에 강건)

def seg(s,ts): return s.loc[(slice(None),ts),:].groupby("REGION").mean()
def trend(s,ts):
    x=np.arange(len(ts))
    return s.loc[(slice(None),ts),:].groupby("REGION").apply(
        lambda g: pd.Series(np.polyfit(x,g.values,1)[0],index=s.columns), include_groups=False)

PRE,POST=[0,1,2],[3,4,5]
y=(seg(sh,POST)["대형할인점"]-seg(sh,PRE)["대형할인점"])*100      # %p, 음수=마트 이탈 빠름
y_noMay=(seg(sh,[3,5])["대형할인점"]-seg(sh,PRE)["대형할인점"])*100
print(f"[표본] {len(y)}개 시군구 | 종속변수 평균 {y.mean():+.2f}%p (표준편차 {y.std():.2f})")
print(f"       5월 제외판과의 상관 {y.corr(y_noMay):+.3f} → 비중 기반이라 5월 영향 작음")

# ── 피처: 1~3월 정보만 ─────────────────────────────────────────────────────
lv=seg(sh,PRE)*100; tr=trend(sh,PRE)*100
unit=(p/c); ulv=seg(unit,PRE); utr=trend(np.log(unit),PRE)*100
cnt_tr=trend(np.log(c),PRE)*100
H=pd.read_csv("ext/households_raw.csv",thousands=",",encoding="cp949")
H["JK"]=H["행정구역"].str.replace(r"\s*\(\d+\)$","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
H=H.drop_duplicates(subset="JK").set_index("JK")
one=H["2026년01월_1인세대"]/H["2026년01월_전체세대"]*100
def jkey(r):
    q=r.split()
    if q[0]=="세종특별자치시": return "세종특별자치시"
    if len(q)>=3 and q[1].endswith("시"): return f"{q[0]} {q[1]}"
    return r
pre3=df[df["T"].isin(PRE)]
kn=pre3[pre3.GENDER_CD!="x"]
ag=kn.pivot_table(index="REGION",columns="AGE",values="AMT",aggfunc="sum"); ag=ag.div(ag.sum(axis=1),axis=0)*100
bz=pre3.pivot_table(index="REGION",columns="TP_BUZ_NM",values="AMT",aggfunc="sum").fillna(0)
bzs=bz.div(bz.sum(axis=1),axis=0)
X=pd.DataFrame({
 "마트비중":lv["대형할인점"], "편의점비중":lv["편의점"],
 "마트비중추세":tr["대형할인점"], "편의점비중추세":tr["편의점"],
 "마트객단가":ulv["대형할인점"], "마트객단가추세":utr["대형할인점"],
 "마트건수추세":cnt_tr["대형할인점"], "편의점건수추세":cnt_tr["편의점"],
 "1인세대%":pd.Series({r:one.get(jkey(r),np.nan) for r in lv.index}),
 "청년2030":(ag["20대"]+ag["30대"]).reindex(lv.index),
 "고령60":ag["60대이상"].reindex(lv.index),
 "업종다양성":(1-(bzs**2).sum(axis=1)).reindex(lv.index),
 "규모log":np.log(pre3.groupby("REGION").AMT.sum()).reindex(lv.index),
}).loc[y.index]
X=X.replace([np.inf,-np.inf],np.nan); keep=X.notna().all(axis=1)
X,y=X[keep],y[keep]
print(f"[피처] {X.shape[1]}개, 표본 {len(X)}개 (1~3월 정보만 사용)")

# ── 베이스라인 vs 모델 ─────────────────────────────────────────────────────
line("1. 베이스라인 — 모델이 이겨야 할 기준")
cv=KFold(5,shuffle=True,random_state=0)
def score(pred,lab):
    r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    rho=spearmanr(y,pred).statistic
    print(f"  {lab:34s} R²={r2:+.3f}  Spearman={rho:+.3f}")
    return r2
score(pd.Series(y.mean(),index=y.index),"① 전국 평균 (상수)")
b2=X.마트비중추세*3   # 1~3월 추세가 그대로 3개월 더 이어진다고 가정
score(b2,"② 단순 지속예측 (추세 연장)")

line("2. 모델")
res={}
for nm,mdl in [("Ridge", make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,30)))),
               ("GBM", GradientBoostingRegressor(n_estimators=300,max_depth=2,learning_rate=.05,random_state=0))]:
    pr=pd.Series(cross_val_predict(mdl,X,y,cv=cv),index=y.index)
    res[nm]=pr; score(pr,f"③ {nm} (5-fold CV)")

best=max(res,key=lambda k:1-((y-res[k])**2).sum()/((y-y.mean())**2).sum())
pr=res[best]
rng=np.random.default_rng(0)
r2=1-((y-pr)**2).sum()/((y-y.mean())**2).sum()
null=[1-((y.values[rng.permutation(len(y))]-pr.values)**2).sum()/((y-y.mean())**2).sum() for _ in range(5000)]
print(f"\n  최고 모델 {best}: 순열검정 p = {np.mean(np.array(null)>=r2):.4f}")

line("3. 무엇이 예측에 기여하나 (규모 지배 여부 확인)")
m=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,30))).fit(X,y)
co=pd.Series(m[-1].coef_,index=X.columns).sort_values(key=abs,ascending=False)
print(co.round(3).to_string())
print(f"\n  규모log 계수 절대값 순위: {list(co.index).index('규모log')+1}/{len(co)}")

line("4. 규모·추세를 뺀 순수 구조 피처만으로도 되는가")
X2=X.drop(columns=["규모log","마트비중추세","편의점비중추세"])
pr2=pd.Series(cross_val_predict(make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,30))),X2,y,cv=cv),index=y.index)
score(pr2,"④ 구조 피처만 (규모·추세 제외)")
pd.DataFrame({"실제":y,"예측":pr}).to_csv(OUT/"prediction.csv",encoding="utf-8-sig")
print(f"\n[save] {OUT/'prediction.csv'}")
