# -*- coding: utf-8 -*-
"""05_did.py — 자연실험: 대형할인점 붕괴가 주변 업종에 미친 영향 (이중차분)
처치군: 대형할인점 계단형 급감이 탐지된 시군구
대조군: 대형할인점이 존재하고 그런 급감이 없는 시군구
결과변수: 같은 시군구의 다른 업종 (전국 업종×월 지수로만 보정 — 지역보정은 순환이라 금지)"""
import pandas as pd, numpy as np, pathlib
OUT = pathlib.Path("out")
df = pd.read_parquet(OUT/"clean.parquet")
E  = pd.read_csv(OUT/"cell_events_v2.csv")
line = lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

cell = df.groupby(["REGION","TP_BUZ_NM","T"], as_index=False).AMT.sum()
nb   = cell.groupby(["TP_BUZ_NM","T"]).AMT.sum()
cell = cell.join((nb/nb.groupby("TP_BUZ_NM").transform("mean")).rename("bidx"), on=["TP_BUZ_NM","T"])
cell["ADJ"] = cell.AMT/cell.bidx                       # 전국 업종×월 보정만

# ── 처치군 정의 ────────────────────────────────────────────────────────────
tr = E[(E.TP_BUZ_NM=="대형할인점") & (E.유형=="폐점의심")]
has = set(cell[cell.TP_BUZ_NM=="대형할인점"].REGION)
treat = set(tr.REGION); ctrl = has - treat
print(f"[설계] 대형할인점 보유 시군구 {len(has)}곳 = 처치군 {len(treat)} + 대조군 {len(ctrl)}")
print(f"       처치 시점 분포 k: {dict(tr.k.value_counts().sort_index())}  (k=폐점 직후 첫 시점)")
K = int(tr.k.median()); print(f"       공통 처치시점 k={K} → 사전 T<{K}, 사후 T>={K}")

# ── 사전 추세 평행성 검정 (DID의 전제) ─────────────────────────────────────
line("1. 평행추세 검정 — 처치 전(T=0,1,2) 두 군의 다른 업종 추세가 같은가")
oth = cell[cell.TP_BUZ_NM!="대형할인점"].copy()
oth["G"] = np.where(oth.REGION.isin(treat), "처치", np.where(oth.REGION.isin(ctrl), "대조", None))
oth = oth.dropna(subset=["G"])
pre = oth[oth["T"] < K].groupby(["G","T"]).ADJ.sum().unstack()
pre = pre.div(pre.iloc[:,0], axis=0)*100
print(pre.round(2).to_string())
sl = {g: np.polyfit(range(pre.shape[1]), np.log(r.values), 1)[0]*100 for g,r in pre.iterrows()}
print(f"\n  사전 추세: 처치군 {sl['처치']:+.2f}%/월, 대조군 {sl['대조']:+.2f}%/월, 차이 {sl['처치']-sl['대조']:+.2f}%p")
print(f"  → {'평행추세 성립. DID 유효' if abs(sl['처치']-sl['대조'])<1.0 else '평행추세 위배. DID 부적합'}")

# ── DID 추정 ───────────────────────────────────────────────────────────────
line("2. 이중차분 — 대형할인점 붕괴 후 같은 지역 '다른 업종'은 어떻게 됐나")
def did(sub, label):
    t = sub.groupby(["G","T"]).ADJ.sum().unstack()
    pre_m, post_m = t.loc[:, t.columns<K].mean(axis=1), t.loc[:, t.columns>=K].mean(axis=1)
    ch = (post_m/pre_m-1)*100
    return label, ch.get("처치",np.nan), ch.get("대조",np.nan), ch.get("처치",np.nan)-ch.get("대조",np.nan)

rows = [did(oth, "다른 업종 전체")]
for b in ["슈퍼마켓","편의점","일반한식","서양음식","스넥","제과점","중국음식","일식회집"]:
    rows.append(did(oth[oth.TP_BUZ_NM==b], f"  └ {b}"))
D = pd.DataFrame(rows, columns=["대상","처치군 변화%","대조군 변화%","DID(순효과%p)"]).set_index("대상")
print(D.round(2).to_string())

# 대형할인점 자체 변화(처치 강도 확인)
hm = cell[cell.TP_BUZ_NM=="대형할인점"].copy()
hm["G"] = np.where(hm.REGION.isin(treat),"처치",np.where(hm.REGION.isin(ctrl),"대조",None))
h = hm.dropna(subset=["G"]).groupby(["G","T"]).ADJ.sum().unstack()
hc = (h.loc[:,h.columns>=K].mean(axis=1)/h.loc[:,h.columns<K].mean(axis=1)-1)*100
print(f"\n  [처치 강도] 대형할인점 자체: 처치군 {hc['처치']:+.1f}% vs 대조군 {hc['대조']:+.1f}%")

# ── 지역별 이질성 + 통계적 유의성(순열검정) ────────────────────────────────
line("3. 지역별 효과 + 순열검정")
def reg_eff(regions):
    s = oth[oth.REGION.isin(regions)].groupby(["REGION","T"]).ADJ.sum().unstack()
    return ((s.loc[:,s.columns>=K].mean(axis=1)/s.loc[:,s.columns<K].mean(axis=1))-1)*100
et, ec = reg_eff(treat), reg_eff(ctrl)
obs = et.mean()-ec.mean()
allr = list(treat|ctrl); pool = pd.concat([et,ec])
rng = np.random.default_rng(0)
null = [pool.iloc[rng.permutation(len(pool))[:len(treat)]].mean() -
        pool.iloc[rng.permutation(len(pool))[len(treat):]].mean() for _ in range(2000)]
p = (np.abs(null) >= abs(obs)).mean()
print(f"  관측 DID {obs:+.2f}%p | 순열검정 p = {p:.3f} "
      f"({'유의 (p<0.05)' if p<0.05 else '유의하지 않음'})")
print(f"  처치군 {len(et)}곳 중 다른 업종이 오히려 성장한 곳 {(et>ec.mean()).sum()}곳")
print("\n  [효과가 가장 부정적인 지역 TOP5]"); print(et.nsmallest(5).round(1).to_string())
print("\n  [폐점에도 상권이 버틴 지역 TOP5]"); print(et.nlargest(5).round(1).to_string())
