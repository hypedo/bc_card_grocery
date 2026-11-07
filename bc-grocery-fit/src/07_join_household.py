# -*- coding: utf-8 -*-
"""07_join_household.py — [검증1] 행안부 주민등록 1인세대 데이터 결합
출처: 행정안전부 주민등록 인구통계 '세대원수별 세대수' (2026.01~06)
주의: 주민등록 CSV는 일반시를 시 단위로 집계(수원시 영통구 ✗ / 수원시 ○).
      BC데이터를 동일 수준으로 올려 결합키를 맞춘다."""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet"); E=pd.read_csv(OUT/"cell_events_v2.csv")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

def jkey(r):
    """BC REGION → 주민등록 결합키"""
    p=r.split()
    if p[0]=="세종특별자치시": return "세종특별자치시"
    if len(p)>=3 and p[1].endswith("시"): return f"{p[0]} {p[1]}"   # 수원시 영통구 → 수원시
    return r
df["JK"]=df.REGION.map(jkey)

h=pd.read_csv("ext/households_raw.csv",thousands=",",encoding="cp949")
h["JK"]=h["행정구역"].str.replace(r"\s*\(\d+\)$","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
M=[f"2026년{m:02d}월" for m in range(1,7)]
h=h[h.JK.isin(set(df.JK))].copy()
for m in M: h[f"one_{m[-3:-1]}"]=h[f"{m}_1인세대"]/h[f"{m}_전체세대"]*100
h=h.drop_duplicates(subset="JK",keep="first")   # 세종: 시도행·시군구행 중복
H=h.set_index("JK")

line("1. 결합 진단")
ours=set(df.JK); print(f"  BC 결합키 {len(ours)}개 (원래 시군구 {df.REGION.nunique()}개) | 매칭 {len(set(H.index)&ours)}개")
miss=sorted(ours-set(H.index)); print(f"  미매칭 {len(miss)}곳: {', '.join(miss) if miss else '없음'}")
cov=df[df.JK.isin(H.index)].AMT.sum()/df.AMT.sum()*100
print(f"  매칭 지역이 전체 소비의 {cov:.1f}% 커버")

line("2. 결정적 비교 — 인구구조는 이 속도를 못 낸다")
no=(h[[f"{m}_1인세대" for m in M]].sum().values/h[[f"{m}_전체세대" for m in M]].sum().values*100)
print("  전국 1인세대 비율: "+" → ".join(f"{v:.2f}%" for v in no)+f"   (6개월 {no[-1]-no[0]:+.2f}%p)")
CH=["대형할인점","슈퍼마켓","편의점"]
n=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index="YM",columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
s=n.div(n.sum(axis=1),axis=0)*100
d_mart=s["대형할인점"].iloc[-1]-s["대형할인점"].iloc[0]
print(f"  대형할인점 비중:   {s['대형할인점'].iloc[0]:.2f}% → {s['대형할인점'].iloc[-1]:.2f}%   (6개월 {d_mart:+.2f}%p)")
print(f"\n  → 채널 변화가 1인세대 변화보다 {abs(d_mart)/(no[-1]-no[0]):.0f}배 빠름")
print(f"  → 1인세대 비율이 5.05%p 움직이려면 현재 속도로 {5.05/((no[-1]-no[0])*2):.0f}년 필요")

line("3. 수준 상관 vs 변화 상관")
agg=df[df.TP_BUZ_NM.isin(CH)].groupby(["JK","TP_BUZ_NM","T"],as_index=False).AMT.sum()
big=df.groupby("JK").AMT.sum(); big=set(big[big>3e10].index)&set(H.index)
a=agg[agg.JK.isin(big)]
lv=a.pivot_table(index="JK",columns="TP_BUZ_NM",values="AMT",aggfunc="sum"); lv=lv.div(lv.sum(axis=1),axis=0)*100
p=a.pivot_table(index=["JK","T"],columns="TP_BUZ_NM",values="AMT",aggfunc="sum").fillna(0)
p=p[(p>0).all(axis=1)]; sh=p.div(p.sum(axis=1),axis=0)
tt=np.log((sh["편의점"]+sh["슈퍼마켓"])/sh["대형할인점"]).groupby("JK").agg(
   lambda x: np.polyfit(np.arange(len(x)),x.values,1)[0]*100 if len(x)==6 else np.nan).dropna()
kn=df[df.GENDER_CD!="x"]; ag=kn.pivot_table(index="JK",columns="AGE",values="AMT",aggfunc="sum")
ag=ag.div(ag.sum(axis=1),axis=0)*100
idx=sorted(set(tt.index)&set(H.index)&big)
A=pd.DataFrame({"1인세대비율":H.loc[idx,"one_01"],"1인세대변화%p":H.loc[idx,"one_06"]-H.loc[idx,"one_01"],
                "편의점비중":lv.loc[idx,"편의점"],"대형할인점비중":lv.loc[idx,"대형할인점"],
                "청년2030":(ag["20대"]+ag["30대"]).loc[idx],"전환속도":tt.loc[idx]})
print(f"  대상 {len(A)}개 지역\n"); print(A.corr().round(3).to_string())
tr={jkey(x) for x in E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].REGION}
B=A[~A.index.isin(tr)]
r1,r2=A["1인세대비율"].corr(A.편의점비중), A["1인세대비율"].corr(A.전환속도)
print(f"\n  ① 1인세대비율(수준) ↔ 편의점비중(수준) = {r1:+.3f}  R²={r1**2*100:.0f}%  ← 팀원 직관의 맞는 절반")
print(f"  ② 1인세대비율(수준) ↔ 전환속도(변화)   = {r2:+.3f}  R²={r2**2*100:.0f}%  ← 1인가구화 가설")
print(f"  ③ 1인세대변화 ↔ 전환속도               = {A['1인세대변화%p'].corr(A.전환속도):+.3f}")
print(f"  ④ 폐점 {len(A)-len(B)}곳 제외 후 ②     = {B['1인세대비율'].corr(B.전환속도):+.3f}")
print(f"\n  1인세대 비율 TOP5: {', '.join(A.nlargest(5,'1인세대비율').index)}")
print(f"  1인세대 비율 BOT5: {', '.join(A.nsmallest(5,'1인세대비율').index)}")
A.to_csv(OUT/"region_household.csv",encoding="utf-8-sig"); print(f"\n[save] {OUT/'region_household.csv'}")
