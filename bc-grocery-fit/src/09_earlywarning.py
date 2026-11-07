# -*- coding: utf-8 -*-
"""09_earlywarning.py — [핵심검정] 폐점은 사전에 예측 가능한가
홈플러스 영업중단은 2026.5.10. 만약 폐점 시군구가 그 이전(1~4월)부터 이미 유의하게
더 빠지고 있었다면, 카드 소비데이터로 폐점을 수개월 전 감지할 수 있다는 뜻이다."""
import pandas as pd, numpy as np, pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
HP=["서울특별시 중랑구","서울특별시 송파구","서울특별시 노원구","부산광역시 해운대구","부산광역시 사상구",
 "부산광역시 영도구","대구광역시 달서구","인천광역시 서구","인천광역시 남동구","인천광역시 연수구",
 "인천광역시 미추홀구","경기도 하남시","경기도 고양시 덕양구","경기도 수원시 팔달구","경기도 부천시 소사구",
 "경기도 성남시 분당구","경기도 남양주시","경기도 고양시 일산서구","경기도 포천시","충청남도 계룡시",
 "전북특별자치도 김제시","전북특별자치도 익산시","전라남도 목포시","전라남도 순천시","경상북도 경산시",
 "경상북도 포항시 북구","경상북도 포항시 남구","경상남도 김해시","경상남도 창원시 마산회원구",
 "경상남도 밀양시","경상남도 사천시","경상남도 진주시","경상남도 창원시 진해구"]

m=df[df.TP_BUZ_NM=="대형할인점"].groupby(["REGION","T"]).agg(AMT=("AMT","sum"),CNT=("CNT","sum")).unstack()
a,c=m["AMT"],m["CNT"]
ok=a.notna().all(axis=1)&(a.min(axis=1)>0)&c.notna().all(axis=1)&(c.min(axis=1)>0)
a,c=a[ok],c[ok]
nat=a.sum(); a=a.div(nat/nat.mean(),axis=1)                     # 전국 월 패턴 보정
natc=c.sum(); c=c.div(natc/natc.mean(),axis=1)
y=pd.Series(a.index.isin(HP).astype(int),index=a.index)
print(f"[표본] 대형할인점 보유 {len(a)}개 시군구 | 홈플러스 폐점 {y.sum()}곳")

# ── 1. 폐점 이전(1~4월)만으로 이미 차이가 나는가 ───────────────────────────
line("1. 사전 신호 검정 — 영업중단(5.10) 이전 1~4월 추세만 사용")
def sl(d,cols): return d[cols].apply(lambda r: np.polyfit(range(len(cols)),np.log(r.values),1)[0]*100,axis=1)
pre_amt, pre_cnt = sl(a,[0,1,2,3]), sl(c,[0,1,2,3])
pre_unit = pre_amt-pre_cnt
for nm,v in [("금액 추세",pre_amt),("건수 추세",pre_cnt),("객단가 추세",pre_unit)]:
    o=v[y==1].mean()-v[y==0].mean()
    rng=np.random.default_rng(5); vv=v.values; n=int(y.sum())
    null=[(lambda p:p[:n].mean()-p[n:].mean())(vv[rng.permutation(len(vv))]) for _ in range(20000)]
    p=np.mean(np.abs(null)>=abs(o))
    print(f"  {nm:8s} 폐점지역 {v[y==1].mean():+6.2f}%/월 | 그외 {v[y==0].mean():+6.2f}%/월 | "
          f"차이 {o:+5.2f}%p | p={p:.4f} {'★유의' if p<0.05 else ''}")

# ── 2. 1~4월 정보만으로 폐점 예측 (누출 없음) ──────────────────────────────
line("2. 조기경보 모델 — 1~4월 피처만으로 5월 폐점 예측")
X=pd.DataFrame({"금액추세":pre_amt,"건수추세":pre_cnt,"객단가추세":pre_unit,
                "규모log":np.log(a.sum(axis=1)),
                "지역내점유율":a.sum(axis=1)/df.groupby("REGION").AMT.sum().reindex(a.index)})
X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median())
mdl=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pr=cross_val_predict(mdl,X,y,cv=cv,method="predict_proba")[:,1]
auc=roc_auc_score(y,pr)
print(f"  5-fold 교차검증 AUC = {auc:.3f}   (0.5=무작위)")
rng=np.random.default_rng(9)
nl=[roc_auc_score(y.values[rng.permutation(len(y))],pr) for _ in range(5000)]
print(f"  순열검정 p = {np.mean(np.array(nl)>=auc):.4f}")
mdl.fit(X,y)
co=pd.Series(mdl[-1].coef_[0],index=X.columns).sort_values()
print("\n  [표준화 회귀계수] 음수 = 폐점 위험 ↑"); print(co.round(3).to_string())
k=int(y.sum())
top=pd.Series(pr,index=X.index).nlargest(k)
print(f"\n  위험 상위 {k}곳 중 실제 폐점 {int(y[top.index].sum())}곳 (무작위 기대 {k*k/len(y):.1f}곳) "
      f"→ 향상도 {y[top.index].sum()/(k*k/len(y)):.1f}배")
print(f"\n  [위험 상위 12곳 — ●실제폐점]")
for r,v in pr_top.items() if False else list(zip(top.index,top.values))[:12]:
    print(f"   {'●' if r in HP else '○'} {r:22s} {v:.3f}")
