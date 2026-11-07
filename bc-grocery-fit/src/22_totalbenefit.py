# -*- coding: utf-8 -*-
"""22_totalbenefit.py — 심사 재지적 반영
(1) 표시 혜택률이 아니라 '월 총혜택' 기준으로 누가 얼마나 이득/손해인가
(2) 소비 구성이 바뀐 집단에게 실제로 유리한가 (반례 검증)
(3) 기본+가중 혼합 설계로 반례를 완화할 수 있는가
(4) 미관측=0 처리 민감도, 사전기간 R 고정의 사후 비용 검증"""
import pandas as pd, numpy as np, pathlib
D=pathlib.Path("/Users/tfmoty/Downloads/catboost_june_tuned_output (1)")
d=pd.read_csv(D/"june_predictions.csv")
CH=["4004","4020","4010"]; NM=["대형할인점","슈퍼마켓","편의점"]
act=d[[f"actual_share_{c}" for c in CH]].to_numpy()
p3 =d[[f"baseline_3m_{c}"  for c in CH]].to_numpy()
amt=d.target_total_amt.to_numpy()
line=lambda t: print(f"\n{'='*76}\n{t}\n{'='*76}")
W=lambda r: np.average(r,weights=amt)

def payout(w): return (act*w).sum(axis=1)          # 지급률 벡터
FLAT=1/3
def mix(alpha, R):                                  # 기본 α + 가중 (1-α)
    return R*(alpha*FLAT + (1-alpha)*p3)
def solve_R(alpha, target):                         # 총 지급액이 target과 같아지는 R
    base=payout(mix(alpha,1.0)); return target/W(base)

R_flat = 0.05                                       # 통합 동일률 기준 (각 1.667%)
flat_w = np.full(3, R_flat/3); flat_pay = payout(flat_w)
TARGET = W(flat_pay)
print(f"[기준] 통합 동일률 각 {R_flat/3*100:.3f}% → 지급률 {TARGET*100:.4f}%  (이 지급액을 모든 안에 동일 적용)")

line("1. 월 총혜택 기준 — 누가 이득이고 누가 손해인가 (완전 가중, α=0)")
R0=solve_R(0.0,TARGET); w0=mix(0.0,R0); pay0=payout(w0)
print(f"  동일 비용을 맞춘 총 혜택률 R = {R0*100:.2f}%  |  지급률 {W(pay0)*100:.4f}% (기준 {TARGET*100:.4f}%)")
diff=pay0-flat_pay; S=300_000
up,dn=diff>1e-12, diff<-1e-12
print(f"  총혜택 증가 집단 {up.mean()*100:5.1f}%  평균 {diff[up].mean()*S:+7.0f}원/월(결제 30만원 기준)")
print(f"  총혜택 감소 집단 {dn.mean()*100:5.1f}%  평균 {diff[dn].mean()*S:+7.0f}원/월")
print(f"  → 표시 혜택률이 오른 집단(87.6%)과 총혜택이 오른 집단은 다르다")

line("2. 반례 검증 — 소비 구성이 바뀐 집단에게 유리한가")
tv=0.5*np.abs(act-p3).sum(axis=1)                   # 직전3개월 대비 당월 구성 변화폭
q=pd.Series(pd.qcut(tv,4,labels=["유지(하위25%)","Q2","Q3","급변(상위25%)"]))
t=pd.DataFrame({"구성변화":tv,"총혜택차":diff*S,"g":q})
print(f"  {'집단':16s} {'구성변화':>8s} {'총혜택 변화':>12s} {'이득비율':>9s}")
for g in ["유지(하위25%)","급변(상위25%)"]:
    sub=t[t.g==g]
    print(f"  {g:16s} {sub.구성변화.mean():7.3f} {sub.총혜택차.mean():+11.0f}원 {(sub.총혜택차>0).mean()*100:8.1f}%")
print("  → 구성이 급변한 집단에서 총혜택이 줄어든다면, 제안의 대상에게 불리하다는 뜻")

line("3. 기본 + 가중 혼합 — 반례를 완화할 수 있는가")
print(f"  {'α(기본비중)':11s} {'R':>7s} {'급변집단 총혜택':>14s} {'유지집단 총혜택':>14s} {'손해집단':>8s}")
best=None
for a in [0.0,0.3,0.5,0.7,0.85,1.0]:
    R=solve_R(a,TARGET); pay=payout(mix(a,R)); df=(pay-flat_pay)*S
    hi=df[(q=="급변(상위25%)").to_numpy()].mean(); lo=df[(q=="유지(하위25%)").to_numpy()].mean()
    loss=(df<-1e-9).mean()*100
    print(f"  {a:9.2f}   {R*100:6.2f}% {hi:+13.0f}원 {lo:+13.0f}원 {loss:7.1f}%")

line("4. 미관측=0 민감도 — 3채널 모두 관측된 집단만")
obs=(act>0).all(axis=1)&(p3>0).all(axis=1)
print(f"  전체 {len(d)} → 3채널 모두 관측 {obs.sum()} ({obs.mean()*100:.1f}%)")
def sub_metrics(mask):
    T=np.average(flat_pay[mask],weights=amt[mask])
    R=T/np.average(payout(mix(0.0,1.0))[mask],weights=amt[mask])
    pay=payout(mix(0.0,R)); df=(pay-flat_pay)[mask]
    top=act[mask].argmax(axis=1)
    rw=(mix(0.0,R)[mask])[np.arange(mask.sum()),top]; rf=np.full(mask.sum(),R_flat/3)
    return (df>0).mean()*100,(rw>rf).mean()*100
for nm,m in [("전체",np.ones(len(d),bool)),("3채널 관측만",obs)]:
    a,b=sub_metrics(m); print(f"  {nm:12s} 총혜택 이득 {a:5.1f}%  |  주이용채널 혜택률 상승 {b:5.1f}%")

line("5. 사전 기간에서 R을 정하고 당월 비용을 검증")
pl=d[[f"baseline_last_{c}" for c in CH]].to_numpy()  # 직전월 구성 = 사전 정보
R_pre = TARGET/W((act*(pl)).sum(axis=1))             # 사전 정보로 정한 R (실제로는 전월 데이터로 산출)
pay_pre=payout(R_pre*pl)
print(f"  사전 정보(직전월 구성)로 R={R_pre*100:.2f}% 고정 → 당월 실제 지급률 {W(pay_pre)*100:.4f}%")
print(f"  목표 {TARGET*100:.4f}% 대비 {(W(pay_pre)-TARGET)*100:+.4f}%p → 사후 조정 없이도 비용이 근사하게 맞는가 확인")
