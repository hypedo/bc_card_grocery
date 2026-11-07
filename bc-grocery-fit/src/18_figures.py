# -*- coding: utf-8 -*-
"""18_figures.py — 요약서용 그림 4종
팔레트: dataviz 기준 팔레트(검증 통과). 인쇄물이라 라이트 모드만 사용."""
import pandas as pd, numpy as np, pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
OUT=pathlib.Path("out/fig"); OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_parquet("out/clean.parquet"); E=pd.read_csv("out/cell_events_v2.csv")

# ── 팔레트 ─────────────────────────────────────────────────────────────────
S1,S2,S3="#2a78d6","#eb6834","#1baf7a"        # 카테고리 1·2·3
POS,NEG="#2a78d6","#e34948"                   # 발산: 파랑↔빨강
SURF,INK,INK2,MUTED="#fcfcfb","#0b0b0b","#52514e","#898781"
GRID,BASE="#e1e0d9","#c3c2b7"
plt.rcParams.update({
 "font.family":"Apple SD Gothic Neo","axes.unicode_minus":False,
 "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
 "text.color":INK,"axes.labelcolor":INK2,"xtick.color":MUTED,"ytick.color":MUTED,
 "axes.edgecolor":BASE,"axes.linewidth":.8,"font.size":11,
 "axes.spines.top":False,"axes.spines.right":False})
def head(ax,t,sub):
    """제목·부제를 축 위에 겹치지 않게 배치"""
    ax.text(0,1.155,t,transform=ax.transAxes,fontsize=13,fontweight="bold",color=INK,va="top")
    ax.text(0,1.045,sub,transform=ax.transAxes,fontsize=10.5,color=INK2,va="top")

def save(fig,name):
    fig.savefig(OUT/name,dpi=220,bbox_inches="tight"); plt.close(fig)
    print(f"  saved {name}")

CH=["대형할인점","슈퍼마켓","편의점"]
MON=["1월","2월","3월","4월","5월","6월"]

# ── 그림1. 3채널 구성 변화 ─────────────────────────────────────────────────
n=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index="T",columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
sh=n.div(n.sum(axis=1),axis=0)*100
fig,ax=plt.subplots(figsize=(7.2,4.2))
for c,col in zip(["슈퍼마켓","편의점","대형할인점"],[S3,S2,S1]):
    ax.plot(range(6),sh[c],color=col,lw=2,marker="o",ms=6,label=c,zorder=3,
            markeredgecolor=SURF,markeredgewidth=1.5)
    ax.annotate(f"{sh[c].iloc[-1]:.1f}%",(5,sh[c].iloc[-1]),xytext=(9,0),
                textcoords="offset points",color=col,fontweight="bold",va="center",fontsize=11)
    ax.annotate(f"{sh[c].iloc[0]:.1f}%",(0,sh[c].iloc[0]),xytext=(-9,0),
                textcoords="offset points",color=col,va="center",ha="right",fontsize=10)
ax.set_xticks(range(6)); ax.set_xticklabels(MON); ax.set_xlim(-.85,5.9)
ax.yaxis.set_major_formatter(PercentFormatter()); ax.grid(axis="y",color=GRID,lw=.8,zorder=0)
ax.set_axisbelow(True); ax.set_ylabel("3채널 내 금액 비중")
head(ax,"3채널 내 금액 비중의 변화","2026년 상반기 · 대형할인점 -5.05%p, 편의점 +4.88%p (집계 기준)")
ax.legend(frameon=False,loc="center left",bbox_to_anchor=(0,.46),fontsize=10,labelcolor=INK2)
save(fig,"fig1_채널이동.png")

# ── 그림2. 연령별 이탈 ─────────────────────────────────────────────────────
AG=["20대이하","20대","30대","40대","50대","60대이상"]
def tr(s): return np.polyfit(range(6),np.log(s.values),1)[0]*100
rows=[]
for a in AG:
    d=df[(df.AGE==a)]
    rows.append([tr(d[d.TP_BUZ_NM=="대형할인점"].groupby("T").AMT.sum()),
                 tr(d[d.TP_BUZ_NM=="편의점"].groupby("T").AMT.sum())])
R=pd.DataFrame(rows,index=AG,columns=["대형할인점","편의점"])
fig,ax=plt.subplots(figsize=(7.2,4.2)); x=np.arange(6); w=.36
ax.bar(x-w/2-.01,R.대형할인점,w,color=S1,label="대형할인점",zorder=3)
ax.bar(x+w/2+.01,R.편의점,w,color=S2,label="편의점",zorder=3)
for i,(a,b) in enumerate(zip(R.대형할인점,R.편의점)):
    ax.text(i-w/2-.01,a-.35,f"{a:.1f}",ha="center",va="top",fontsize=9.5,color=S1,fontweight="bold")
    ax.text(i+w/2+.01,b+.3,f"+{b:.1f}",ha="center",va="bottom",fontsize=9.5,color=S2,fontweight="bold")
ax.axhline(0,color=BASE,lw=1,zorder=2)
ax.set_xticks(x); ax.set_xticklabels(AG); ax.set_ylabel("월평균 증감률 (%)")
ax.grid(axis="y",color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True); ax.set_ylim(-9.8,9.6)
head(ax,"연령대별 월평균 증감률","관측된 전 연령대가 같은 방향이며 속도만 다르다")
ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.0),
          fontsize=10,labelcolor=INK2,columnspacing=1.6)
save(fig,"fig2_연령별이탈.png")

# ── 그림3. DID 업종 서열 (5월 제외 정본) ───────────────────────────────────
K=3
cell=df.groupby(["REGION","TP_BUZ_NM","T"],as_index=False).AMT.sum()
nb=cell.groupby(["TP_BUZ_NM","T"]).AMT.sum()
cell=cell.join((nb/nb.groupby("TP_BUZ_NM").transform("mean")).rename("b"),on=["TP_BUZ_NM","T"])
cell["ADJ"]=cell.AMT/cell.b
tr_=set(E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].REGION)
has=set(cell[cell.TP_BUZ_NM=="대형할인점"].REGION)
ev=set(zip(E[E.EVENT].REGION,E[E.EVENT].TP_BUZ_NM))
o=cell[(cell.TP_BUZ_NM!="대형할인점")&cell.REGION.isin(has)]
o=o[~pd.Series([(r,b) in ev for r,b in zip(o.REGION,o.TP_BUZ_NM)],index=o.index)]
s=o.pivot_table(index=["REGION","TP_BUZ_NM"],columns="T",values="ADJ")
s=s[s.notna().all(axis=1)&(s>0).all(axis=1)]
y=np.log(s[[3,5]].mean(axis=1)/s[[0,1,2]].mean(axis=1))*100
d=y.rename("c").reset_index(); d["T_"]=d.REGION.isin(tr_)
did=(d[d.T_].groupby("TP_BUZ_NM").c.mean()-d[~d.T_].groupby("TP_BUZ_NM").c.mean())
did=did[["슈퍼마켓","스넥","제과점","편의점","일반한식","중국음식","서양음식","일식회집"]].sort_values()
fig,ax=plt.subplots(figsize=(7.2,4.4))
ax.barh(did.index,did.values,color=[POS if v>0 else NEG for v in did.values],height=.62,zorder=3)
for i,v in enumerate(did.values):
    ax.text(v+(.12 if v>0 else -.12),i,f"{v:+.2f}%p",va="center",ha="left" if v>0 else "right",
            fontsize=10,fontweight="bold",color=POS if v>0 else NEG)
ax.axvline(0,color=BASE,lw=1,zorder=2); ax.set_xlim(-2.2,5.6)
ax.grid(axis="x",color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)
ax.set_xlabel("이중차분 순효과 (%p)"); ax.tick_params(axis="y",labelcolor=INK)
head(ax,"폐점 시군구에서 업종별로 다르게 나타난 변화","외부 확인 33곳 vs 대조군 · 이중차분 순효과 (5월 제외 기준)")
save(fig,"fig3_업종서열.png")

# ── 그림4. 건수 × 실질객단가 2×2 ───────────────────────────────────────────
INFL=(119.99/118.03)**(1/5)-1
pts=[]
for b in ["대형할인점","슈퍼마켓","편의점","일반한식","서양음식","스넥","제과점","중국음식","일식회집"]:
    g=df[df.TP_BUZ_NM==b].groupby("T").agg(A=("AMT","sum"),C=("CNT","sum"))
    pts.append((b,tr(g.C),tr(g.A/g.C)-INFL*100))
P=pd.DataFrame(pts,columns=["업종","건수","객단가"]).set_index("업종")
fig,ax=plt.subplots(figsize=(7.2,4.8))
ax.axhline(0,color=BASE,lw=1); ax.axvline(0,color=BASE,lw=1)
ax.scatter(P.건수,P.객단가,s=110,color=S1,zorder=3,edgecolor=SURF,linewidth=1.5)
# 라벨 충돌을 피하려고 업종별로 위치를 지정한다 (dx, dy, 정렬)
LAB={"대형할인점":(9,0,"left","center"),   "슈퍼마켓":(0,-13,"center","top"),
     "편의점":(9,2,"left","center"),       "일반한식":(0,9,"center","bottom"),
     "서양음식":(-9,0,"right","center"),   "스넥":(9,0,"left","center"),
     "제과점":(-9,0,"right","center"),     "중국음식":(9,0,"left","center"),
     "일식회집":(9,0,"left","center")}
for b,r in P.iterrows():
    dx,dy,ha,va=LAB[b]
    ax.annotate(b,(r.건수,r.객단가),xytext=(dx,dy),textcoords="offset points",
                fontsize=10,color=INK,ha=ha,va=va)
ax.set_xlim(-3.4,6.2); ax.set_ylim(-3.6,1.6)
for xy,t,c in [((.97,.03),"건수 ↑ · 객단가 ↓",POS),((.03,.03),"건수 ↓ · 객단가 ↓",NEG)]:
    ax.text(*xy,t,transform=ax.transAxes,ha="right" if xy[0]>.5 else "left",va="bottom",
            fontsize=10,color=c,fontweight="bold",linespacing=1.4)
ax.set_xlabel("결제 건수 월추세 (%)"); ax.set_ylabel("실질 객단가 월추세 (%)")
ax.grid(color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)
head(ax,"결제 건수와 건당 실질 결제액의 방향","스넥을 제외한 모든 업종에서 실질 객단가 하락 · 대형할인점만 건수도 감소")
save(fig,"fig4_다운사이징.png")
print("\n[수치 확인]")
print(f"  그림1 대형할인점 {sh['대형할인점'].iloc[0]:.2f}% → {sh['대형할인점'].iloc[-1]:.2f}%")
print(f"  그림3 DID: "+", ".join(f"{k} {v:+.2f}" for k,v in did.sort_values(ascending=False).items()))
