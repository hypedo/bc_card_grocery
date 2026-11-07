# -*- coding: utf-8 -*-
"""04b_events2.py — 이중보정(업종×지역) 후 이벤트 재탐지
1차 실패 원인: 전국 업종지수만 보정하면 관광지 고유 계절성(울릉·고성 6월 성수기)이
'개점'으로 오인됨. → 해당 지역 전체의 월 패턴을 leave-one-out으로 함께 보정한다.
보정 후 남는 것 = 같은 지역 다른 업종은 안 움직이는데 이 업종만 계단 = 진짜 점포 이벤트."""
import pandas as pd, numpy as np, pathlib
OUT = pathlib.Path("out")
df = pd.read_parquet(OUT/"clean.parquet")
line = lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

cell = df.groupby(["REGION","TP_BUZ_NM","T"], as_index=False).AMT.sum()

# 1) 전국 업종×월 지수
nb  = cell.groupby(["TP_BUZ_NM","T"]).AMT.sum()
bidx = (nb/nb.groupby("TP_BUZ_NM").transform("mean")).rename("bidx")
# 2) 지역×월 지수 — 자기 자신 제외(leave-one-out)해야 순환참조가 없음
reg_tot = cell.groupby(["REGION","T"]).AMT.sum().rename("rtot")
cell = cell.join(reg_tot, on=["REGION","T"]).join(bidx, on=["TP_BUZ_NM","T"])
cell["loo"] = cell.rtot - cell.AMT                       # 같은 지역 '다른 업종' 합계
loo_mean = cell.groupby(["REGION","TP_BUZ_NM"]).loo.transform("mean")
cell["ridx"] = cell.loo / loo_mean                       # 지역 고유 월 패턴
cell["ADJ2"] = cell.AMT / (cell.bidx * cell.ridx)        # 이중보정

piv  = cell.pivot_table(index=["REGION","TP_BUZ_NM"], columns="T", values="ADJ2")
size = cell.groupby(["REGION","TP_BUZ_NM"]).AMT.sum()
piv  = piv[(piv.notna().sum(axis=1)==6) & (piv>0).all(axis=1) & (size.reindex(piv.index)>6e8)]

def step_scan(y):
    ly = np.log(y); best = (0.0, np.nan, np.inf)
    for k in range(1,6):
        a,b = ly[:k], ly[k:]
        d = np.median(b)-np.median(a)
        sp = np.sqrt((a.var(ddof=0)*len(a)+b.var(ddof=0)*len(b))/6)
        if abs(d) > abs(best[0]): best = (d,k,sp)
    return best

E = pd.DataFrame([(r,b,*step_scan(y.values.astype(float))) for (r,b),y in piv.iterrows()],
                 columns=["REGION","TP_BUZ_NM","delta","k","spread"]).set_index(["REGION","TP_BUZ_NM"])
E["선명도"] = E.delta.abs()/(E.spread+1e-9); E["변화율%"] = (np.exp(E.delta)-1)*100
E["EVENT"] = (E.delta.abs() > np.log(1.30)) & (E.선명도 > 3)
E["유형"] = np.where(~E.EVENT,"정상", np.where(E.delta<0,"폐점의심","개점의심"))

old = pd.read_csv(OUT/"cell_events.csv").set_index(["REGION","TP_BUZ_NM"])
line("1. 1차(전국보정) vs 2차(이중보정) 탐지 결과")
print(f"  1차: 이벤트 {old.EVENT.sum():3d}개 (폐점 {(old.유형=='폐점의심').sum():2d} / 개점 {(old.유형=='개점의심').sum():2d})")
print(f"  2차: 이벤트 {E.EVENT.sum():3d}개 (폐점 {(E.유형=='폐점의심').sum():2d} / 개점 {(E.유형=='개점의심').sum():2d})")
j = old[["EVENT"]].join(E[["EVENT"]], lsuffix="_1", rsuffix="_2").dropna()
print(f"  1차에서만 잡힘 {(j.EVENT_1&~j.EVENT_2).sum()}개 | 공통 {(j.EVENT_1&j.EVENT_2).sum()}개 | 2차에서만 {(~j.EVENT_1&j.EVENT_2).sum()}개")

line("2. 검증 — 1차 '개점의심' 상위(관광지 의심)가 2차에서 어떻게 되는가")
susp = ["경상남도 고성군","경상북도 울릉군","충청남도 태안군","전북특별자치도 장수군",
        "대구광역시 군위군","경상북도 영양군","강원특별자치도 고성군"]
chk = old[old.EVENT & old.index.get_level_values(0).isin(susp)].join(
      E[["변화율%","EVENT","유형"]], lsuffix="_1차", rsuffix="_2차")
print(chk[["변화율%_1차","변화율%_2차","EVENT_2차"]].round(1).to_string())
surv = chk.EVENT_2차.sum()
print(f"\n  → 관광지 의심 {len(chk)}건 중 2차 생존 {surv}건 "
      f"({'대부분 제거됨 — 보정 성공' if surv <= len(chk)*0.3 else '여전히 남음 — 추가 보정 필요'})")

line("3. 업종별 검출률 (이중보정 후)")
v = (E.reset_index().groupby("TP_BUZ_NM").agg(셀수=("EVENT","size"), 이벤트=("EVENT","sum"))
       .assign(검출률=lambda d: d.이벤트/d.셀수*100).sort_values("검출률",ascending=False))
print(v.round(1).to_string())

line("4. 최종 이벤트 목록")
ev = E[E.EVENT].sort_values("변화율%")
print("[폐점의심]"); print(ev[ev.delta<0][["변화율%","k","선명도"]].round(1).to_string())
print("\n[개점의심]"); print(ev[ev.delta>0][["변화율%","k","선명도"]].round(1).to_string())
E.to_csv(OUT/"cell_events_v2.csv", encoding="utf-8-sig")
print(f"\n[save] {OUT/'cell_events_v2.csv'}")
