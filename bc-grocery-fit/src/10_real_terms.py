# -*- coding: utf-8 -*-
"""10_real_terms.py — [검증3] 소비자물가 보정, 명목 vs 실질
출처: 통계청 소비자물가동향 (2020=100)  2026.01 = 118.03 / 2026.06 = 119.99
      2026.06 전년동월대비 +3.2%, 2026.01 +2.0%"""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
CPI_1, CPI_6 = 118.03, 119.99
infl6 = CPI_6/CPI_1 - 1
infl_m = (CPI_6/CPI_1)**(1/5) - 1          # 1월→6월은 5개월 간격
print(f"[물가] 2026.01 {CPI_1} → 2026.06 {CPI_6}  |  6개월 누적 {infl6*100:+.2f}%  월평균 {infl_m*100:+.2f}%")

def tr(s):                                  # 로그 선형 월추세(%)
    return np.polyfit(range(len(s)), np.log(s.values), 1)[0]*100

line("1. 명목 vs 실질 — 업종별 월 추세")
rows=[]
for b in ["대형할인점","슈퍼마켓","편의점","일반한식","서양음식","스넥","제과점"]:
    g=df[df.TP_BUZ_NM==b].groupby("YM").agg(AMT=("AMT","sum"),CNT=("CNT","sum"))
    g["UNIT"]=g.AMT/g.CNT
    rows.append((b, tr(g.AMT), tr(g.AMT)-infl_m*100, tr(g.CNT), tr(g.UNIT), tr(g.UNIT)-infl_m*100))
T=pd.DataFrame(rows,columns=["업종","금액(명목)","금액(실질)","건수","객단가(명목)","객단가(실질)"]).set_index("업종")
print(T.round(2).to_string())

line("2. 핵심 — 대형할인점 장바구니의 실질 축소")
h=df[df.TP_BUZ_NM=="대형할인점"].groupby("YM").agg(AMT=("AMT","sum"),CNT=("CNT","sum"))
u1,u6=h.AMT.iloc[0]/h.CNT.iloc[0], h.AMT.iloc[-1]/h.CNT.iloc[-1]
real=(u6/u1)/(CPI_6/CPI_1)-1
print(f"  객단가 명목: {u1:,.0f}원 → {u6:,.0f}원  ({(u6/u1-1)*100:+.2f}%)")
print(f"  객단가 실질(2026.01 물가 기준): {u1:,.0f}원 → {u6/(CPI_6/CPI_1):,.0f}원  ({real*100:+.2f}%)")
print(f"\n  → 6개월 만에 실질 장바구니 {abs(real)*100:.1f}% 축소")
c1,c6=h.CNT.iloc[0],h.CNT.iloc[-1]
print(f"  → 방문 건수도 {(c6/c1-1)*100:+.1f}%  (손님·장바구니 동시 축소)")

line("3. 편의점은 반대로 움직였나")
c=df[df.TP_BUZ_NM=="편의점"].groupby("YM").agg(AMT=("AMT","sum"),CNT=("CNT","sum"))
cu1,cu6=c.AMT.iloc[0]/c.CNT.iloc[0], c.AMT.iloc[-1]/c.CNT.iloc[-1]
print(f"  편의점 객단가 명목 {cu1:,.0f}원 → {cu6:,.0f}원 ({(cu6/cu1-1)*100:+.2f}%) | "
      f"실질 {((cu6/cu1)/(CPI_6/CPI_1)-1)*100:+.2f}%")
print(f"  편의점 건수 {(c.CNT.iloc[-1]/c.CNT.iloc[0]-1)*100:+.1f}%")
print(f"\n  → 대형할인점: 건수 {(c6/c1-1)*100:+.1f}% / 객단가 실질 {real*100:+.1f}%")
print(f"     편의점    : 건수 {(c.CNT.iloc[-1]/c.CNT.iloc[0]-1)*100:+.1f}% / 객단가 실질 {((cu6/cu1)/(CPI_6/CPI_1)-1)*100:+.1f}%")
print("     → '큰 장보기 1회'가 '작은 장보기 여러 회'로 분해되는 패턴과 일치")

line("4. 전체 소비의 실질 성장")
t=df.groupby("YM").AMT.sum()
print(f"  11업종 전체 명목 월 {tr(t):+.2f}% | 실질 월 {tr(t)-infl_m*100:+.2f}%")
print(f"  장보기 3채널 합계 명목 월 {tr(df[df.TP_BUZ_NM.isin(['대형할인점','슈퍼마켓','편의점'])].groupby('YM').AMT.sum()):+.2f}% "
      f"| 실질 월 {tr(df[df.TP_BUZ_NM.isin(['대형할인점','슈퍼마켓','편의점'])].groupby('YM').AMT.sum())-infl_m*100:+.2f}%")
