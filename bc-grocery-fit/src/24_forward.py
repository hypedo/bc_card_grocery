# -*- coding: utf-8 -*-
"""24_forward.py — 사전 산정 R의 다음 달 비용 검증
팀원 산출물이 6월분뿐이어서 못 했던 검증을, 원자료로 집단 패널을 직접 만들어 수행한다.
절차: 5월 시점 정보로 R 산정(비용 중립) → 그 R을 고정해 6월 적용 → 실제 지급률 확인"""
import pandas as pd, numpy as np
CH=["4004","4020","4010"]; NM={"4004":"대형할인점","4020":"슈퍼마켓","4010":"편의점"}
df=pd.read_csv("ABP_CONTEST_DATA.csv",encoding="utf-8-sig",dtype=str)
df.columns=[c.strip().upper() for c in df.columns]
d=df[df.TP_BUZ_NO.isin(CH)&df.GENDER_CD.isin(["1","2"])&df.AGE_CD.isin(list("123456"))].copy()
d["AMT"]=pd.to_numeric(d.AMT); d=d[d.AMT>0]
d["G"]=d.SIDO_NM+"|"+d.CCG_NM+"|"+d.GENDER_CD+"|"+d.AGE_CD
d["M"]=d.STRD_YYMM.astype(int)
P=d.pivot_table(index=["G","M"],columns="TP_BUZ_NO",values="AMT",aggfunc="sum").fillna(0.0)
for c in CH:
    if c not in P: P[c]=0.0
P=P[CH]; P["tot"]=P.sum(axis=1); P=P[P.tot>0]
line=lambda t: print(f"\n{'='*76}\n{t}\n{'='*76}")

def shares(months):
    """해당 월들의 합계 기준 채널 구성비 (집단별)"""
    x=P[P.index.get_level_values("M").isin(months)].groupby(level="G")[CH].sum()
    return x.div(x.sum(axis=1),axis=0)
def realized(month):
    x=P[P.index.get_level_values("M")==month].groupby(level="G")[CH].sum()
    return x.div(x.sum(axis=1),axis=0), x.sum(axis=1)

R_FLAT=0.05; FLAT=R_FLAT/3
def payrate(w, act, amt):           # 금액가중 지급률
    return np.average((act.values*w.values).sum(axis=1), weights=amt.values)

line("검증 설계")
print("  ① 5월 시점: 직전 3개월(2~4월) 구성으로 가중, 5월 실현으로 비용 중립 R 산정")
print("  ② 그 R을 고정해 6월 적용: 직전 3개월(3~5월) 구성으로 가중, 6월 실현으로 지급률 계산")
print("  ③ 6월에 사후적으로 맞춘 R과 비교")

w5=shares([202602,202603,202604]); a5,m5=realized(202605)
w6=shares([202603,202604,202605]); a6,m6=realized(202606)
g=w5.index.intersection(a5.index).intersection(w6.index).intersection(a6.index)
w5,a5,m5,w6,a6,m6=[x.loc[g] for x in (w5,a5,m5,w6,a6,m6)]
print(f"\n  공통 집단 {len(g):,}개")

T5=payrate(pd.DataFrame(FLAT,index=g,columns=CH),a5,m5)
R_pre=T5/payrate(w5,a5,m5)*1.0
print(f"\n[①] 5월 비용 중립 R = {R_pre*100:.2f}%   (5월 기본안 지급률 {T5*100:.4f}%)")

T6=payrate(pd.DataFrame(FLAT,index=g,columns=CH),a6,m6)
p6_pre=payrate(w6*R_pre,a6,m6)
R_post=T6/payrate(w6,a6,m6)
p6_post=payrate(w6*R_post,a6,m6)
line("결과 — 미리 정한 R을 다음 달에 적용했을 때")
print(f"  6월 기본안 지급률(비교 기준)      {T6*100:.4f}%")
print(f"  ① 5월에 정한 R={R_pre*100:.2f}% 적용   {p6_pre*100:.4f}%   기본안 대비 {(p6_pre/T6-1)*100:+.2f}%")
print(f"  ③ 6월에 사후로 맞춘 R={R_post*100:.2f}%  {p6_post*100:.4f}%   기본안 대비 {(p6_post/T6-1)*100:+.2f}%")
tot6=m6.sum()
print(f"\n  6월 결제액 {tot6/1e8:,.0f}억 기준 지급 총액")
print(f"   기본안 {T6*tot6/1e8:,.1f}억  |  사전 R 적용 {p6_pre*tot6/1e8:,.1f}억  "
      f"(차이 {(p6_pre-T6)*tot6/1e8:+,.1f}억)")
line("참고 — 반대 방향(4월에 정해 5월 적용)")
w4=shares([202601,202602,202603]); a4,m4=realized(202604)
gg=w4.index.intersection(a4.index).intersection(g)
T4=payrate(pd.DataFrame(FLAT,index=gg,columns=CH),a4.loc[gg],m4.loc[gg])
R4=T4/payrate(w4.loc[gg],a4.loc[gg],m4.loc[gg])
p5=payrate(w5.loc[gg]*R4,a5.loc[gg],m5.loc[gg]); T5g=payrate(pd.DataFrame(FLAT,index=gg,columns=CH),a5.loc[gg],m5.loc[gg])
print(f"  4월에 정한 R={R4*100:.2f}% → 5월 지급률 {p5*100:.4f}% (기본안 {T5g*100:.4f}%, {(p5/T5g-1)*100:+.2f}%)")
