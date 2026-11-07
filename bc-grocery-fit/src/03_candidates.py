# -*- coding: utf-8 -*-
"""03_candidates.py — 후보 1·5·6·7 go/no-go 검정"""
import pandas as pd, numpy as np, pathlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
OUT = pathlib.Path("out"); np.random.seed(0)
df = pd.read_parquet(OUT/"clean.parquet"); R = pd.read_csv(OUT/"region_metrics.csv", index_col=0)
line = lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

# ══ 후보1: 외국인 소비 — 관광형 vs 정주형 분리 가능한가 ═══════════════════
line("[후보1] 외국인 소비 상권 유형 분리")
fo = df[df.GENDER_CD == "3"]
fp = fo.groupby(["REGION","T"], as_index=False).AMT.sum()
fcv = fp.groupby("REGION").AMT.agg(lambda s: s.std()/s.mean()*100).rename("외국인CV")
fb  = fo.pivot_table(index="REGION", columns="TP_BUZ_NM", values="AMT", aggfunc="sum").fillna(0)
fb  = fb.div(fb.sum(axis=1), axis=0)*100
fa  = fo.pivot_table(index="REGION", columns="AGE", values="AMT", aggfunc="sum").fillna(0)
fa  = fa.div(fa.sum(axis=1), axis=0)*100

C = pd.DataFrame({"외국인비중%": R["외국인%"], "외국인CV": fcv,
                  "관광업종%": fb.get("일식회집",0)+fb.get("서양음식",0),
                  "생필품업종%": fb.get("슈퍼마켓",0)+fb.get("대형할인점",0)+fb.get("편의점",0),
                  "청년2030%": fa["20대"]+fa["30대"]}).dropna()
C = C[C["외국인비중%"] > C["외국인비중%"].quantile(.80)]     # 외국인 비중 상위 20% 지역
X = StandardScaler().fit_transform(C[["외국인CV","관광업종%","생필품업종%","청년2030%"]])
for k in (2,3):
    lab = KMeans(k, n_init=20, random_state=0).fit_predict(X)
    print(f"  k={k} 실루엣 {silhouette_score(X, lab):.3f}")
C["군집"] = KMeans(2, n_init=20, random_state=0).fit_predict(X)
print("\n  [군집 프로파일]"); print(C.groupby("군집").agg(["mean","size"]).round(1).iloc[:, [0,1,2,4,6,8]])
for g, sub in C.groupby("군집"):
    print(f"\n  군집{g} ({len(sub)}곳): {', '.join(sub.nlargest(8,'외국인비중%').index)}")

# ══ 후보5: 외식/내식 비율 변화가 지역별로 갈리는가 ════════════════════════
line("[후보5] 외식/내식 대체 — 지역별 긴축 속도 분산")
cp = df.pivot_table(index=["REGION","T"], columns="CAT", values="AMT", aggfunc="sum").fillna(0)
cp["ratio"] = cp["외식"]/cp["유통"]
def sl(s):
    y = np.log(s.values); x = np.arange(len(y)); return np.polyfit(x, y, 1)[0]*100
rr = cp.groupby("REGION").ratio.agg(sl).rename("외식내식추세%").loc[R.index]
nat = sl(df.groupby(["T","CAT"]).AMT.sum().unstack().eval("외식/유통"))
print(f"  전국 외식/내식 비율 추세: 월 {nat:+.2f}%")
print(f"  지역 분산: 표준편차 {rr.std():.2f}%p | p10 {rr.quantile(.1):+.2f} p90 {rr.quantile(.9):+.2f}")
print(f"  전국 추세 대비 지역 분산비 = {rr.std()/abs(nat):.1f}배")
print(f"  → {'지역 고유 신호 충분 (PASS)' if rr.std() > abs(nat)*0.8 else '전국 동조. 지역분석 불가 (FAIL)'}")
print(f"\n  긴축(내식전환) 빠른 지역: {', '.join(rr.nsmallest(5).index)}")
print(f"  외식 회복 빠른 지역   : {', '.join(rr.nlargest(5).index)}")

# ══ 후보6: 변동성이 지역 고유인가, 업종 구성 탓인가 ═══════════════════════
line("[후보6] 계절 변동성의 분산분해")
bz  = df.pivot_table(index=["REGION","TP_BUZ_NM"], values="AMT", aggfunc="sum")
w   = bz.AMT/bz.groupby("REGION").AMT.transform("sum")           # 지역별 업종 가중치
nb  = df.groupby(["TP_BUZ_NM","T"]).AMT.sum()
nbi = (nb/nb.groupby("TP_BUZ_NM").transform("mean")).rename("idx") # 전국 업종별 월지수
syn = (w.reset_index().merge(nbi.reset_index(), on="TP_BUZ_NM")
         .assign(v=lambda d: d.AMT*d.idx).groupby(["REGION","T"]).v.sum())
syn_cv = syn.groupby("REGION").agg(lambda s: s.std()/s.mean()*100).rename("업종설명CV").loc[R.index]
V = pd.DataFrame({"실제CV": R.cv, "업종설명CV": syn_cv}).assign(고유CV=lambda d: d.실제CV-d.업종설명CV)
print(f"  실제 CV 평균 {V.실제CV.mean():.2f}% | 업종구성으로 설명되는 CV 평균 {V.업종설명CV.mean():.2f}%")
print(f"  corr(실제, 업종설명) = {V.실제CV.corr(V.업종설명CV):+.3f} "
      f"| 업종구성 설명력 R² = {V.실제CV.corr(V.업종설명CV)**2:.3f}")
print(f"  → {'변동성 대부분이 지역 고유 요인 (PASS)' if V.실제CV.corr(V.업종설명CV)**2 < 0.4 else '업종구성 탓. 지표 무효 (FAIL)'}")
print(f"\n  지역고유 변동성 TOP6:"); print(V.nlargest(6,"고유CV").round(2))

# ══ 후보7: 객단가가 관광 프록시를 걷어내고도 살아남는가 ═══════════════════
line("[후보7] 한식 객단가의 소득 프록시 타당성")
K = R[["한식객단가","외국인%","cv","청년2030%","HHI"]].dropna()
y  = np.log(K.한식객단가); Xr = np.c_[np.ones(len(K)), K[["외국인%","cv"]].values]
bta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
res = y - Xr@bta
r2 = 1 - res.var()/y.var()
print(f"  log(한식객단가) ~ 외국인비중 + 변동성 :  R² = {r2:.3f}")
print(f"  → 관광 프록시가 설명하는 비율 {r2*100:.0f}%, 잔차(고유 소비수준) {100-r2*100:.0f}%")
print(f"  → {'관광으로 설명 안 됨. 독립 신호 (PASS)' if r2 < 0.25 else '관광 프록시에 불과 (FAIL)'}")
K["잔차서열"] = res
print(f"\n  관광보정 후 객단가 상위: {', '.join(K.nlargest(6,'잔차서열').index)}")
print(f"  관측 상위(보정 전)     : {', '.join(K.nlargest(6,'한식객단가').index)}")
ov = len(set(K.nlargest(15,'잔차서열').index) & set(K.nlargest(15,'한식객단가').index))
print(f"  상위15 중복 {ov}/15 → 보정 후에도 서열 {'유지' if ov>=9 else '크게 변동'}")
