# -*- coding: utf-8 -*-
"""13_did_recheck.py — 5월 교란이 DID/DDD를 오염시켰는지 재검정
문제: 사전 T0-2(1~3월) / 사후 T3-5(4~6월) 분할에 5월이 사후에만 들어간다.
전국 업종×월 보정은 전국 평균 5월 효과만 제거하고, 지역별 5월 민감도 차이는 남는다.
대책: (a) 처치·대조군의 5월 민감도 비교 (b) 5월을 통째로 빼고 재계산"""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet"); E=pd.read_csv(OUT/"cell_events_v2.csv")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
K=3
cell=df.groupby(["REGION","TP_BUZ_NM","T"],as_index=False).AMT.sum()
nb=cell.groupby(["TP_BUZ_NM","T"]).AMT.sum()
cell=cell.join((nb/nb.groupby("TP_BUZ_NM").transform("mean")).rename("bidx"),on=["TP_BUZ_NM","T"])
cell["ADJ"]=cell.AMT/cell.bidx
tr=set(E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].REGION)
has=set(cell[cell.TP_BUZ_NM=="대형할인점"].REGION); ctrl=has-tr

m=df.groupby(["REGION","T"]).AMT.sum().unstack()
may=(m[4]/m[[0,1,2,3,5]].mean(axis=1)*100)
line("1. 처치군과 대조군의 5월 민감도가 다른가")
a,b=may[list(tr&has)],may[list(ctrl)]
print(f"  처치군 {len(a)}곳 {a.mean():.1f}  |  대조군 {len(b)}곳 {b.mean():.1f}  |  차이 {a.mean()-b.mean():+.1f}")
rng=np.random.default_rng(2); v=may[list(has)].values; n=len(a)
null=[(lambda p:p[:n].mean()-p[n:].mean())(v[rng.permutation(len(v))]) for _ in range(20000)]
p=np.mean(np.abs(null)>=abs(a.mean()-b.mean()))
print(f"  순열검정 p = {p:.4f} → {'두 군의 5월 민감도가 다름, 교란 위험' if p<0.05 else '두 군이 비슷함, DID는 상쇄됨'}")

def run(cols_pre, cols_post, label):
    o=cell[(cell.TP_BUZ_NM!="대형할인점")&cell.REGION.isin(has)].copy()
    ev=set(zip(E[E.EVENT].REGION,E[E.EVENT].TP_BUZ_NM))
    o=o[~pd.Series([(r,bz) in ev for r,bz in zip(o.REGION,o.TP_BUZ_NM)],index=o.index)]
    s=o.pivot_table(index=["REGION","TP_BUZ_NM"],columns="T",values="ADJ")
    s=s[s.notna().all(axis=1)&(s>0).all(axis=1)]
    y=np.log(s[cols_post].mean(axis=1)/s[cols_pre].mean(axis=1))*100
    d=y.rename("chg").reset_index(); d["TREAT"]=d.REGION.isin(tr).astype(int)
    d["GRP"]=np.where(d.TP_BUZ_NM.isin(["슈퍼마켓","편의점"]),"유통","외식")
    pv=d.pivot_table(index="GRP",columns="TREAT",values="chg",aggfunc="mean")
    g=pv[1]-pv[0]; ddd=g["유통"]-g["외식"]
    regs=d.REGION.unique(); nt=len(tr&set(regs)); rg=np.random.default_rng(11); nl=[]
    for _ in range(10000):
        t=set(regs[rg.permutation(len(regs))[:nt]]); mm=d.REGION.isin(t)
        q=d.assign(TR=mm).pivot_table(index="GRP",columns="TR",values="chg",aggfunc="mean")
        gg=q[True]-q[False]; nl.append(gg["유통"]-gg["외식"])
    pp=np.mean(np.abs(nl)>=abs(ddd))
    byb=d[d.TREAT==1].groupby("TP_BUZ_NM").chg.mean()-d[d.TREAT==0].groupby("TP_BUZ_NM").chg.mean()
    return label,g["유통"],g["외식"],ddd,pp,byb

line("2. 5월 포함 vs 5월 제외 — DDD 재계산")
rows=[run([0,1,2],[3,4,5],"기존: 사전 1~3월 / 사후 4~6월 (5월 포함)"),
      run([0,1,2],[3,5],  "수정: 사전 1~3월 / 사후 4·6월 (5월 제외)"),
      run([0,1],  [3,5],  "보수: 사전 1~2월 / 사후 4·6월 (5월 제외)")]
print(f"{'설정':42s} {'유통DID':>8s} {'외식DID':>8s} {'DDD':>8s} {'p':>8s}")
for lab,u,d_,ddd,pp,_ in rows:
    print(f"{lab:42s} {u:+8.2f} {d_:+8.2f} {ddd:+8.2f} {pp:8.4f} {'★' if pp<0.05 else ''}")

line("3. 5월 제외 시 업종별 DID")
byb=rows[1][5].sort_values(ascending=False)
print(byb.round(2).to_string())
print(f"\n  기존(5월포함) 업종별:"); print(rows[0][5].sort_values(ascending=False).round(2).to_string())
