# -*- coding: utf-8 -*-
"""21_cost_check.py — 심사 지적 반영: 혜택 '예산 고정'의 정의와 비용 검증
질문1. 앞선 시뮬레이션의 '예산 고정'은 할인율의 합이지 지급액이 아니다. 지급액은 얼마나 늘어나는가?
질문2. 지급액을 동일하게 맞추면 자동 가중은 여전히 유리한가?
질문3. 세 채널 통합(동일률)만으로 충분한가, 가중이 추가로 무엇을 주는가?"""
import pandas as pd, numpy as np, pathlib
D=pathlib.Path("/Users/tfmoty/Downloads/catboost_june_tuned_output (1)")
d=pd.read_csv(D/"june_predictions.csv")
CH=["4004","4020","4010"]; NM=["대형할인점","슈퍼마켓","편의점"]
act=d[[f"actual_share_{c}" for c in CH]].to_numpy()
p3 =d[[f"baseline_3m_{c}"  for c in CH]].to_numpy()
amt=d.target_total_amt.to_numpy()
line=lambda t: print(f"\n{'='*76}\n{t}\n{'='*76}")
B=0.05                      # 할인율의 합 (3채널 가중 합)
W=lambda r: np.average(r,weights=amt)

line("1. '예산 고정'의 정의 — 할인율 합을 고정해도 지급액은 같지 않다")
flat_w = np.full(3, B/3)                       # 통합 동일률
flat_pay = (act*flat_w).sum(axis=1)            # 실제 구성에 적용한 지급률
wt_w   = B*p3                                  # 구성 비례 가중
wt_pay = (act*wt_w).sum(axis=1)
print(f"  통합 동일률 (각 {B/3*100:.3f}%)   지급률 {W(flat_pay)*100:.4f}%")
print(f"  구성 비례 가중              지급률 {W(wt_pay)*100:.4f}%")
print(f"  차이 {(W(wt_pay)-W(flat_pay))*100:+.4f}%p  →  같은 비용이 아니라 비용이 늘어난다")
S=300_000
print(f"\n  월 결제 {S:,}원 기준 1인당 지급액: "
      f"{W(flat_pay)*S:,.0f}원 → {W(wt_pay)*S:,.0f}원 ({(W(wt_pay)-W(flat_pay))*S:+,.0f}원)")
tot=amt.sum()
print(f"  표본 전체 6월 결제액 {tot/1e8:,.0f}억 기준 지급 총액 "
      f"{W(flat_pay)*tot/1e8:,.1f}억 → {W(wt_pay)*tot/1e8:,.1f}억 ({(W(wt_pay)-W(flat_pay))*tot/1e8:+,.1f}억)")

line("2. 지급액을 동일하게 맞추면 — 가중은 무엇을 남기는가")
k = W(flat_pay)/W((act*p3).sum(axis=1))        # 지급률이 같아지도록 가중 예산 재조정
wt_eq_w = k*p3
wt_eq_pay = (act*wt_eq_w).sum(axis=1)
print(f"  재조정 계수 k = {k:.4f}  (할인율 합 {B*100:.1f}% → {k*100:.2f}%)")
print(f"  통합 동일률 지급률 {W(flat_pay)*100:.4f}%  |  동일비용 가중 지급률 {W(wt_eq_pay)*100:.4f}%")
top=act.argmax(axis=1)
r_flat=flat_w[top]; r_wt=wt_eq_w[np.arange(len(d)),top]
print(f"\n  [고객 체감률] 본인이 가장 많이 쓰는 채널에 붙는 혜택률")
print(f"   통합 동일률 {W(r_flat)*100:.3f}%  →  동일비용 가중 {W(r_wt)*100:.3f}%  ({(W(r_wt)-W(r_flat))*100:+.3f}%p)")
print(f"   체감률이 오른 집단 {(r_wt>r_flat).mean()*100:.1f}%  |  내린 집단 {(r_wt<r_flat).mean()*100:.1f}%")

line("3. 통합만으로 충분한가 — 단계별 기여 분해")
cur_w=np.array([0.01,0.01,0.03])               # 현행 유사: 편의점만 우대, 채널별 분리
steps=[("현행형 — 채널별 분리 우대", (act*cur_w).sum(axis=1)),
       ("① 3채널 통합 동일률",        flat_pay),
       ("② 통합 + 구성 비례 가중",    wt_pay),
       ("③ ②를 동일 비용으로 조정",   wt_eq_pay)]
base=None
for nm,r in steps:
    v=W(r); 
    print(f"  {nm:26s} 지급률 {v*100:7.4f}%" + ("" if base is None else f"   직전 대비 {(v-base)*100:+.4f}%p"))
    base=v
print(f"\n  → ①이 효과의 대부분. ②는 비용도 함께 증가. ③은 비용 동일 시 지급률 차이 거의 없음")
