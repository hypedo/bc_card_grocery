# -*- coding: utf-8 -*-
"""04_events.py — 점포 개·폐점 이벤트 셀 탐지 및 제외
계절보정 시계열에서 '단일 계단형 구조변화'를 찾아 개·폐점 이벤트로 판정한다.
근거: 대형할인점은 시군구당 점포가 1~2개뿐이라 1개 점포 개폐가 계단으로 나타나지만,
      편의점은 수백 개라 같은 크기의 계단이 생길 수 없다. 업종별 검출률이 이 논리를 검증한다."""
import pandas as pd, numpy as np, pathlib
OUT = pathlib.Path("out")
df = pd.read_parquet(OUT/"clean.parquet")
line = lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

# ── 1. 계절보정 패널 (전국 업종×월 지수로 나눠 5월 특수·설 제거) ──────────
cell = df.groupby(["REGION","TP_BUZ_NM","T"], as_index=False).agg(AMT=("AMT","sum"), CNT=("CNT","sum"))
nb   = df.groupby(["TP_BUZ_NM","T"]).AMT.sum()
idx  = (nb / nb.groupby("TP_BUZ_NM").transform("mean")).rename("nidx").reset_index()
cell = cell.merge(idx, on=["TP_BUZ_NM","T"])
cell["ADJ"] = cell.AMT / cell.nidx                       # 계절보정 금액

piv  = cell.pivot_table(index=["REGION","TP_BUZ_NM"], columns="T", values="ADJ")
size = cell.groupby(["REGION","TP_BUZ_NM"]).AMT.sum()
piv  = piv[(piv.notna().sum(axis=1) == 6) & (size.reindex(piv.index) > 6e8)]   # 6개월 완전 + 6억↑
print(f"[panel] 분석 대상 셀 {len(piv):,}개 (지역×업종)")

# ── 2. 단일 계단 탐지 ──────────────────────────────────────────────────────
def step_scan(y):
    """6시점 로그 시계열에서 최적 단일 분할점과 계단 크기·선명도를 반환"""
    ly = np.log(y)
    best = (0.0, np.nan, np.inf)                          # (delta, k, 내부산포)
    for k in range(1, 6):
        a, b = ly[:k], ly[k:]
        delta = np.median(b) - np.median(a)
        spread = np.sqrt((a.var(ddof=0)*len(a) + b.var(ddof=0)*len(b)) / 6)  # 구간내 표준편차
        if abs(delta) > abs(best[0]):
            best = (delta, k, spread)
    return best

rows = []
for (reg, bz), y in piv.iterrows():
    d, k, sp = step_scan(y.values.astype(float))
    rows.append((reg, bz, d, k, sp, abs(d)/(sp+1e-9)))
E = pd.DataFrame(rows, columns=["REGION","TP_BUZ_NM","delta","k","spread","선명도"]).set_index(["REGION","TP_BUZ_NM"])
E["변화율%"] = (np.exp(E.delta)-1)*100

# 판정: 계단 30%↑ 이고 계단이 잡음 대비 뚜렷(선명도 3↑)
E["EVENT"] = (E.delta.abs() > np.log(1.30)) & (E.선명도 > 3)
E["유형"]  = np.where(~E.EVENT, "정상", np.where(E.delta < 0, "폐점의심", "개점의심"))
print(f"[detect] 이벤트 셀 {E.EVENT.sum():,}개 ({E.EVENT.mean()*100:.1f}%) "
      f"— 폐점의심 {(E.유형=='폐점의심').sum()} / 개점의심 {(E.유형=='개점의심').sum()}")

# ── 3. 타당성 검증: 업종별 검출률이 '점포 밀도' 가설과 맞는가 ──────────────
line("3. 업종별 이벤트 검출률 — 점포 밀도 가설 검증")
v = (E.reset_index().groupby("TP_BUZ_NM")
       .agg(셀수=("EVENT","size"), 이벤트=("EVENT","sum"))
       .assign(검출률=lambda d: d.이벤트/d.셀수*100).sort_values("검출률", ascending=False))
v["6개월건수만"] = (df.groupby("TP_BUZ_NM").CNT.sum()/1e4).round(0)
print(v.round(1).to_string())
print("\n  → 점포수가 적은 업종(대형할인점)일수록 검출률이 높아야 가설이 성립")

line("4. 탐지된 주요 이벤트")
ev = E[E.EVENT].join(size.rename("AMT6")).sort_values("변화율%")
print("\n[폐점의심 TOP10]")
print(ev.head(10)[["변화율%","k","선명도"]].round(1).to_string())
print("\n[개점의심 TOP8]")
print(ev.tail(8)[["변화율%","k","선명도"]].round(1).to_string())

E.to_csv(OUT/"cell_events.csv", encoding="utf-8-sig")

# ── 5. 이벤트 제외가 지표에 미치는 영향 ────────────────────────────────────
line("5. 이벤트 셀 제외 전후 — 지표 변화")
bad = set(E[E.EVENT].index)
cell["EVENT"] = list(zip(cell.REGION, cell.TP_BUZ_NM))
cell["EVENT"] = cell.EVENT.isin(bad)
print(f"  이벤트 셀 금액 비중: 전체의 {cell[cell.EVENT].AMT.sum()/cell.AMT.sum()*100:.2f}%")

def region_cv(c):
    r = c.groupby(["REGION","T"]).AMT.sum().unstack()
    r = r[r.sum(axis=1) > 2e10]
    return (r.std(axis=1)/r.mean(axis=1)*100)
cv_b, cv_a = region_cv(cell), region_cv(cell[~cell.EVENT])
j = pd.concat([cv_b.rename("제외전"), cv_a.rename("제외후")], axis=1).dropna()
print(f"\n  [지역 월변동성 CV] 평균 {j.제외전.mean():.2f}% → {j.제외후.mean():.2f}% "
      f"| 순위상관 {j.제외전.corr(j.제외후, method='spearman'):.3f}")
print("  변동성이 가장 크게 줄어든 지역(= 폐점이 변동성을 만들던 곳):")
print((j.제외전-j.제외후).nlargest(5).round(2).to_string())

def ratio_trend(c):
    RET = {"대형할인점","편의점","슈퍼마켓"}
    c = c.assign(CAT=np.where(c.TP_BUZ_NM.isin(RET), "유통", "외식"))
    p = c.pivot_table(index=["REGION","T"], columns="CAT", values="AMT", aggfunc="sum").fillna(0)
    p = p[(p>0).all(axis=1)]
    return p.eval("외식/유통").groupby("REGION").agg(
        lambda s: np.polyfit(np.arange(len(s)), np.log(s.values), 1)[0]*100 if len(s)==6 else np.nan)
rt_b, rt_a = ratio_trend(cell), ratio_trend(cell[~cell.EVENT])
k = pd.concat([rt_b.rename("제외전"), rt_a.rename("제외후")], axis=1).dropna()
k = k.loc[k.index.intersection(j.index)]
print(f"\n  [외식/내식 비율 추세] 순위상관 {k.제외전.corr(k.제외후, method='spearman'):.3f}")
print("  '외식 회복 1위'가 제외 전후로 어떻게 바뀌는가:")
print("   제외 전:", ", ".join(k.nlargest(5,'제외전').index))
print("   제외 후:", ", ".join(k.nlargest(5,'제외후').index))
cell[~cell.EVENT].to_parquet(OUT/"panel_noevent.parquet", index=False)
print(f"\n[save] {OUT/'cell_events.csv'}, {OUT/'panel_noevent.parquet'}")
