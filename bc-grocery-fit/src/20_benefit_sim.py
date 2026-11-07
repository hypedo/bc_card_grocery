# -*- coding: utf-8 -*-
"""20_benefit_sim.py — 혜택 시뮬레이션
팀원 보고서 7절이 제안한 검증: 예측 오차 개선이 실제 혜택 개선으로 이어지는가?
각 집단에 대해 (a)기준선 (b)예측 (c)완전예측(oracle)으로 패키지를 고르고,
실제 6월 소비 구성에 적용해 실현 혜택률을 비교한다.
※ 혜택률 수치는 팀원 보고서의 예시값이며 제안값이 아니다(탄력성 추정 불가)."""
import pandas as pd, numpy as np, pathlib
D=pathlib.Path("/Users/tfmoty/Downloads/catboost_june_tuned_output (1)")
d=pd.read_csv(D/"june_predictions.csv")
line=lambda t: print(f"\n{'='*76}\n{t}\n{'='*76}")
CH=["4004","4020","4010"]; NM={"4004":"대형할인점","4020":"슈퍼마켓","4010":"편의점"}
PKG={"마트형":{"4004":.03,"4020":.01,"4010":.01},
     "슈퍼형":{"4004":.01,"4020":.03,"4010":.01},
     "편의점형":{"4004":.01,"4020":.01,"4010":.03}}
names=list(PKG)
rate=np.array([[PKG[p][c] for c in CH] for p in names])          # 패키지 × 채널
act=d[[f"actual_share_{c}" for c in CH]].to_numpy()
SRC={"직전월":"baseline_last","최근3개월":"baseline_3m","CatBoost":"prediction","완전예측":None}
amt=d.target_total_amt.to_numpy()

def pick(est): return (est@rate.T).argmax(axis=1)                 # 기대 혜택 최대 패키지
def realized(choice): return (act*rate[choice]).sum(axis=1)       # 실제 구성에 적용한 혜택률

res={}
for k,col in SRC.items():
    est=act if col is None else d[[f"{col}_{c}" for c in CH]].to_numpy()
    ch=pick(est); res[k]=(ch,realized(ch))

line("1. 어떤 패키지가 선택되는가")
t=pd.DataFrame({k:pd.Series(names).reindex(range(3)).values for k in []})
for k,(ch,_) in res.items():
    v=pd.Series(ch).value_counts().reindex(range(3),fill_value=0)
    print(f"  {k:8s} " + " | ".join(f"{names[i]} {v[i]:4d}({v[i]/len(d)*100:4.1f}%)" for i in range(3)))

line("2. 실현 혜택률 — 예측 개선이 혜택 개선으로 이어지는가")
base=res["최근3개월"][1]
print(f"  {'선택 기준':10s} {'단순평균':>10s} {'금액가중':>10s} {'기준선대비':>12s}")
for k,(ch,r) in res.items():
    w=np.average(r,weights=amt)
    diff=(w-np.average(base,weights=amt))*100
    print(f"  {k:10s} {r.mean()*100:9.4f}% {w*100:9.4f}% {diff:+11.4f}%p")

line("3. 추천이 실제로 달라졌는가 (CatBoost vs 최근3개월)")
b_ch,b_r=res["최근3개월"]; p_ch,p_r=res["CatBoost"]
chg=b_ch!=p_ch
print(f"  추천 변경 집단: {chg.sum()} / {len(d)} ({chg.mean()*100:.2f}%)")
if chg.sum():
    g=(p_r-b_r)[chg]
    print(f"  변경된 집단 중 혜택 개선 {(g>0).sum()}곳 / 악화 {(g<0).sum()}곳 / 동일 {(g==0).sum()}곳")
    print(f"  변경 집단 평균 효과 {g.mean()*100:+.4f}%p | 변경 집단 금액 비중 {amt[chg].sum()/amt.sum()*100:.2f}%")
line("4. 달성 가능 상한 대비")
o=np.average(res["완전예측"][1],weights=amt); b=np.average(base,weights=amt); p=np.average(p_r,weights=amt)
print(f"  기준선 {b*100:.4f}%  →  CatBoost {p*100:.4f}%  →  완전예측 {o*100:.4f}%")
gap=o-b
print(f"  이론적 개선 여지 {gap*100:.4f}%p 중 CatBoost가 달성 {(p-b)/gap*100 if gap else 0:.1f}%")
line("5. 예측 오차와 혜택의 관계")
for k,col in [("최근3개월","baseline_3m"),("CatBoost","prediction")]:
    est=d[[f"{col}_{c}" for c in CH]].to_numpy()
    print(f"  {k:8s} MAE {np.abs(est-act).mean()*100:.4f}%p | 실현혜택 {np.average(res[k][1],weights=amt)*100:.4f}%")
print("\n  → 오차는 줄었는데 혜택은 얼마나 움직였는지 위 표로 판단")

# ── 6. 이산 선택 vs 연속 가중 ───────────────────────────────────────────────
# 위 결과의 원인 가설: '3개 패키지 중 택1'이라는 이산 구조가 예측 정밀도를 흡수한다.
# 요약서 5장이 제안한 '가중치 자동 갱신'은 연속 배분이므로 정밀도가 그대로 반영되어야 한다.
line("6. 이산 패키지 선택 vs 연속 가중 배분")
BUD=0.05                                     # 총 혜택 예산(3채널 가중 합) — 수준이 아니라 구조 비교용
def cont(est): return (act*(BUD*est)).sum(axis=1)
flat=(act*(BUD/3)).sum(axis=1)
rows=[("균등 배분(무예측)",flat)]
for k,col in [("직전월","baseline_last"),("최근3개월","baseline_3m"),("CatBoost","prediction")]:
    rows.append((k,cont(d[[f"{col}_{c}" for c in CH]].to_numpy())))
rows.append(("완전예측",cont(act)))
b=np.average(rows[2][1],weights=amt)         # 최근3개월 기준
print(f"  {'배분 방식':16s} {'금액가중 혜택률':>14s} {'최근3개월 대비':>14s}")
for k,r in rows:
    w=np.average(r,weights=amt)
    print(f"  {k:16s} {w*100:13.4f}% {(w-b)*100:+13.4f}%p")
disc_gain=np.average(res['CatBoost'][1],weights=amt)-np.average(res['최근3개월'][1],weights=amt)
cont_gain=np.average(rows[3][1],weights=amt)-b
print(f"\n  이산 선택에서 CatBoost 이득  {disc_gain*100:+.4f}%p")
print(f"  연속 가중에서 CatBoost 이득  {cont_gain*100:+.4f}%p")
print(f"  → 연속 배분이 {cont_gain/disc_gain:.0f}배 크게 반영" if disc_gain>0 else "")
