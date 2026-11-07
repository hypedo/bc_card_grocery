# -*- coding: utf-8 -*-
"""23_v5.py — 4차 심사 지적 반영
(1) α 혼합을 '지급액 혼합'으로 재정의해 재현 가능하게
(2) 시범 대상 선정을 사전 정보만으로 (미래 정보 배제)
(3) 대상군에서 R 재검토
(4) 최다 채널 단일 우대 방식과의 비교"""
import pandas as pd, numpy as np, pathlib
D=pathlib.Path("/Users/tfmoty/Downloads/catboost_june_tuned_output (1)")
d=pd.read_csv(D/"june_predictions.csv"); CH=["4004","4020","4010"]
act=d[[f"actual_share_{c}" for c in CH]].to_numpy()
p3 =d[[f"baseline_3m_{c}"  for c in CH]].to_numpy()
pl =d[[f"baseline_last_{c}" for c in CH]].to_numpy()
amt=d.target_total_amt.to_numpy()
W=lambda r,m=None: np.average(r if m is None else r[m], weights=amt if m is None else amt[m])
pay=lambda w:(act*w).sum(axis=1); S=300_000
line=lambda t: print(f"\n{'='*76}\n{t}\n{'='*76}")
R_FLAT=0.05; flat_w=np.full(3,R_FLAT/3); fp=pay(flat_w); T=W(fp)

line("1. α 혼합 재정의 — 혜택률을 섞으면 지급액도 정확히 선형으로 섞인다")
Rw=T/W(pay(p3))
print(f"  가중안 R_w = {Rw*100:.2f}% (단독 적용 시 비용 중립)")
print(f"  혼합 혜택률  w_c = α × 1.667% + (1-α) × (R_w × 직전3개월비중_c)")
print(f"\n  {'α':>5s} {'손해집단':>7s} {'평균감소액':>10s} {'하위5%':>9s} {'평균이득':>9s} {'지급률':>9s}")
for a in [0.0,0.25,0.5,0.75,0.85]:
    w = a*flat_w + (1-a)*(Rw*p3)
    df=(pay(w)-fp)*S; dn=df<-1e-9
    print(f"  {a:5.2f} {dn.mean()*100:6.1f}% {df[dn].mean():9.0f}원 {np.percentile(df,5):8.0f}원 "
          f"{df[df>0].mean():8.0f}원 {W(pay(w))*100:8.4f}%")
print("  → 손실이 (1-α)에 정확히 비례한다. 이전 표는 '비중을 섞고 R을 재정규화'한 값이라 달랐다")

line("2. 시범 대상 선정 — 사전 정보만 사용")
ex_ante = 0.5*np.abs(pl-p3).sum(axis=1)     # 직전월 vs 직전3개월 구성차 (6월 이전에 계산 가능)
post_hoc= 0.5*np.abs(act-p3).sum(axis=1)    # 당월 vs 직전3개월 (사후에만 관측)
print(f"  사전 지표와 사후 실제 변화폭의 상관 = {np.corrcoef(ex_ante,post_hoc)[0,1]:+.3f}")
stable = ex_ante <= np.median(ex_ante)      # 하위 50% = 구성이 안정적인 집단
print(f"  안정 집단(사전 기준 하위 50%) {stable.sum()}개 / {len(d)}개")
print(f"  이 집단의 사후 실제 변화폭 평균 {post_hoc[stable].mean():.4f} vs 나머지 {post_hoc[~stable].mean():.4f}")

line("3. 시범 대상군에서 R 재검토")
T_s = W(fp, stable)
Rw_s = T_s/W(pay(p3), stable)
print(f"  전체 기준 R_w = {Rw*100:.2f}%  →  안정 집단 기준 R_w = {Rw_s*100:.2f}%")
for nm,R,m in [("전체 R을 그대로 적용",Rw,stable),("대상군에서 재산정",Rw_s,stable)]:
    w=R*p3; df=(pay(w)-fp)[m]*S
    print(f"  {nm:20s} 지급률 {W(pay(w),m)*100:.4f}% (기준 {T_s*100:.4f}%) | "
          f"총혜택 증가 {(df>1e-9).mean()*100:.1f}% | 손해 평균 {df[df<0].mean():.0f}원")

line("4. 비례 가중 vs 최다 채널 단일 우대")
top3=np.zeros_like(p3); top3[np.arange(len(d)),p3.argmax(axis=1)]=1   # 사전 기준 최다 채널
designs={}
designs["통합 동일률"]=flat_w*np.ones_like(p3)
designs["최다 채널 단일 우대"]=(T/W(pay(top3)))*top3
designs["비중 비례 가중"]=Rw*p3
print(f"  {'설계':18s} {'지급률':>9s} {'증가집단':>8s} {'손해평균':>9s} {'하위5%':>9s} {'급변집단':>9s} {'안정집단':>9s}")
hi=post_hoc>=np.percentile(post_hoc,75)
for nm,w in designs.items():
    df=(pay(w)-fp)*S; dn=df<-1e-9
    print(f"  {nm:18s} {W(pay(w))*100:8.4f}% {(df>1e-9).mean()*100:7.1f}% "
          f"{(df[dn].mean() if dn.any() else 0):8.0f}원 {np.percentile(df,5):8.0f}원 "
          f"{df[hi].mean():+8.0f}원 {df[stable].mean():+8.0f}원")
print("\n  → 단일 우대는 이득도 손실도 크고, 비례 가중은 중간. 구성이 급변할 때 차이를 본다")
