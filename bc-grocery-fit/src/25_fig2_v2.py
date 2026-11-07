# -*- coding: utf-8 -*-
"""25_fig2_v2.py — 그림2 재작성
기존 그림은 '월 추세(로그 선형회귀 기울기)'였고 본문 표는 '1월 대비 6월 변화율'이라
같은 쪽에서 지표가 달랐다. 표와 동일한 기준으로 통일하고 핵심 3채널만 표시한다."""
import pandas as pd, numpy as np, pathlib, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT=pathlib.Path("out/fig"); df=pd.read_parquet("out/clean.parquet")
S1,S2,S3="#2a78d6","#eb6834","#1baf7a"
POS,NEG="#2a78d6","#e34948"
SURF,INK,INK2,MUTED="#fcfcfb","#0b0b0b","#52514e","#898781"; GRID,BASE="#e1e0d9","#c3c2b7"
plt.rcParams.update({"font.family":"Apple SD Gothic Neo","axes.unicode_minus":False,
 "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,"text.color":INK,
 "axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,"axes.edgecolor":BASE,
 "axes.linewidth":.8,"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
CPI=119.99/118.03
rows=[]
for b,col in [("대형할인점",S1),("슈퍼마켓",S3),("편의점",S2)]:
    g=df[df.TP_BUZ_NM==b].groupby("T").agg(A=("AMT","sum"),C=("CNT","sum")); u=g.A/g.C
    rows.append((b,(g.C.iloc[-1]/g.C.iloc[0]-1)*100,((u.iloc[-1]/u.iloc[0])/CPI-1)*100,col))
fig,ax=plt.subplots(figsize=(7.2,4.4))
ax.axhline(0,color=BASE,lw=1,zorder=1); ax.axvline(0,color=BASE,lw=1,zorder=1)
for b,x,y,col in rows:
    ax.scatter(x,y,s=190,color=col,zorder=3,edgecolor=SURF,linewidth=2)
    ax.annotate(f"{b}\n건수 {x:+.1f}% · 객단가 {y:+.1f}%",(x,y),
                xytext=(0,-16 if b=="슈퍼마켓" else 16),textcoords="offset points",
                ha="center",va="top" if b=="슈퍼마켓" else "bottom",
                fontsize=10,color=INK,linespacing=1.45,fontweight="bold")
ax.set_xlim(-22,27); ax.set_ylim(-13.5,4.5)
ax.text(.975,.04,"건수 ↑ · 객단가 ↓",transform=ax.transAxes,ha="right",va="bottom",
        fontsize=10,color=POS,fontweight="bold")
ax.text(.025,.04,"건수 ↓ · 객단가 ↓",transform=ax.transAxes,ha="left",va="bottom",
        fontsize=10,color=NEG,fontweight="bold")
ax.set_xlabel("결제 건수 변화율 (1월 대비 6월, %)")
ax.set_ylabel("실질 건당 금액 변화율 (%)")
ax.grid(color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)
ax.text(0,1.155,"결제 건수와 건당 실질 결제액의 방향",transform=ax.transAxes,
        fontsize=13,fontweight="bold",color=INK,va="top")
ax.text(0,1.045,"세 채널 모두 건당 금액이 줄었고, 대형할인점만 건수까지 감소했다",
        transform=ax.transAxes,fontsize=10.5,color=INK2,va="top")
fig.savefig(OUT/"fig4_다운사이징.png",dpi=220,bbox_inches="tight"); plt.close(fig)
print("  saved fig4_다운사이징.png (1월 대비 6월 변화율 기준)")
for b,x,y,_ in rows: print(f"   {b:6s} 건수 {x:+.1f}%  실질객단가 {y:+.1f}%")
