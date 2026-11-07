# -*- coding: utf-8 -*-
"""06_channel.py — [후보8] 장보기 채널 이동 검정"""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet"); E=pd.read_csv(OUT/"cell_events_v2.csv")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
CH=["대형할인점","슈퍼마켓","편의점"]
r=df[df.TP_BUZ_NM.isin(CH)]

# ── 1. 전국 채널 구성비가 6개월간 실제로 움직이는가 ────────────────────────
line("1. 전국 3채널 구성비 추이 (금액 기준)")
n=r.pivot_table(index="YM",columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
sh=(n.div(n.sum(axis=1),axis=0)*100)
print(sh.round(2).to_string())
print("\n  6개월 총 변화(%p):"); print((sh.iloc[-1]-sh.iloc[0]).round(2).to_string())
for c in CH:
    b=np.polyfit(range(6),np.log(n[c].values),1)[0]*100
    print(f"  {c:6s} 금액 추세 월 {b:+.2f}%")
print("\n  건수 기준 구성비:")
nc=r.pivot_table(index="YM",columns="TP_BUZ_NM",values="CNT",aggfunc="sum")
print((nc.div(nc.sum(axis=1),axis=0)*100).round(2).to_string())

# ── 2. 지역별 채널 이동 추세 — 신호인가 잡음인가 (성별 분할 신뢰도) ────────
line("2. 신뢰도 검정 — 같은 지역을 남성/여성 표본으로 따로 추정하면 일치하는가")
def ch_trend(sub):
    p=sub.pivot_table(index=["REGION","T"],columns="TP_BUZ_NM",values="AMT",aggfunc="sum").fillna(0)
    p=p[(p>0).all(axis=1)]
    s=p.div(p.sum(axis=1),axis=0)
    # 편의점+슈퍼 대비 대형할인점 비율의 로그추세 = '근거리 전환 속도'
    lr=np.log((s["편의점"]+s["슈퍼마켓"])/s["대형할인점"])
    g=lr.groupby("REGION")
    return g.agg(lambda x: np.polyfit(np.arange(len(x)),x.values,1)[0]*100 if len(x)==6 else np.nan).dropna()
big=df.groupby("REGION").AMT.sum(); big=set(big[big>3e10].index)
tm=ch_trend(r[(r.GENDER_CD=="1")&r.REGION.isin(big)])
tf=ch_trend(r[(r.GENDER_CD=="2")&r.REGION.isin(big)])
tall=ch_trend(r[r.REGION.isin(big)])
j=pd.concat([tm.rename("남성"),tf.rename("여성"),tall.rename("전체")],axis=1).dropna()
rel=j.남성.corr(j.여성)
print(f"  대상 {len(j)}개 시군구 | 근거리전환 추세 평균 월 {j.전체.mean():+.2f}% (표준편차 {j.전체.std():.2f})")
print(f"  corr(남성표본 추정, 여성표본 추정) = {rel:+.3f}")
print(f"  스피어만-브라운 보정 신뢰도 = {2*rel/(1+rel):.3f}")
print(f"  → {'지역 고유 신호 존재 (PASS)' if rel>0.4 else '표본 잡음이 지배 (FAIL)'}")

# ── 3. 대형할인점 폐점 이벤트가 이 지표를 만들어내는가 ─────────────────────
line("3. 교란 검정 — '채널 이동'이 실은 대형할인점 폐점 아닌가")
tr=set(E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].REGION)
j["폐점지역"]=j.index.isin(tr)
print(f"  폐점 발생 지역 {j.폐점지역.sum()}곳: 근거리전환 추세 월 {j[j.폐점지역].전체.mean():+.2f}%")
print(f"  그 외    {(~j.폐점지역).sum()}곳: 근거리전환 추세 월 {j[~j.폐점지역].전체.mean():+.2f}%")
print(f"  차이 {j[j.폐점지역].전체.mean()-j[~j.폐점지역].전체.mean():+.2f}%p")
top=j.nlargest(10,"전체"); print(f"\n  전환속도 TOP10 중 폐점지역 {top.폐점지역.sum()}곳: {', '.join(top.index[:6])}")
print(f"  → {'폐점이 지표를 지배. 1인가구화로 해석 불가' if top.폐점지역.sum()>=4 else '폐점과 독립적'}")

# ── 4. 채널 구성비(수준)는 인구구조와 맞는가 ───────────────────────────────
line("4. 채널 구성비 '수준'이 연령구조와 정합적인가 (수준 vs 변화)")
kn=df[df.GENDER_CD!="x"]
ag=kn.pivot_table(index="REGION",columns="AGE",values="AMT",aggfunc="sum").fillna(0)
ag=ag.div(ag.sum(axis=1),axis=0)*100
lv=r.pivot_table(index="REGION",columns="TP_BUZ_NM",values="AMT",aggfunc="sum").fillna(0)
lv=lv.div(lv.sum(axis=1),axis=0)*100
L=pd.DataFrame({"편의점비중":lv["편의점"],"대형할인점비중":lv["대형할인점"],"슈퍼비중":lv["슈퍼마켓"],
                "청년2030":ag["20대"]+ag["30대"],"고령5060":ag["50대"]+ag["60대이상"]}).loc[sorted(big)]
L["전환추세"]=j.전체
print(L.corr().round(2).to_string())
print(f"\n  편의점 비중 TOP6: {', '.join(L.nlargest(6,'편의점비중').index)}")
print(f"  대형할인점 비중 TOP6: {', '.join(L.nlargest(6,'대형할인점비중').index)}")
