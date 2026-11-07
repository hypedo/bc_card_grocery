# -*- coding: utf-8 -*-
"""02_eda.py — 1차 EDA + 후보 아이디어 판별(go/no-go) 지표"""
import pandas as pd, numpy as np, pathlib
pd.set_option("display.width", 200)
OUT = pathlib.Path("out")
df = pd.read_parquet(OUT/"clean.parquet")
line = lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

# ── 0. 기본 분포 ───────────────────────────────────────────────────────────
line("0. 기본 분포")
b = (df.groupby("TP_BUZ_NM").agg(금액조=("AMT","sum"), 건수만=("CNT","sum"))
       .assign(금액조=lambda d: d.금액조/1e12, 건수만=lambda d: d.건수만/1e4)
       .assign(객단가=lambda d: d.금액조*1e12/(d.건수만*1e4), 금액비중=lambda d: d.금액조/d.금액조.sum()*100)
       .sort_values("금액조", ascending=False))
print(b.round({"금액조":2,"건수만":0,"객단가":0,"금액비중":1}))

m = df.groupby("YM").agg(금액조=("AMT","sum"), 건수억=("CNT","sum"))
m["금액조"] /= 1e12; m["건수억"] /= 1e8; m["객단가"] = m.금액조*1e12/(m.건수억*1e8)
m["전월비%"] = m.금액조.pct_change()*100
print("\n[월별]"); print(m.round(2))

# ── 1. 계절보정: 전국 동월·동업종 점유율(share) ─────────────────────────────
# 지역×업종의 '전국 동업종 동월 합계 대비 점유율' → 계절·전국추세가 상쇄됨
p = df.groupby(["REGION","TP_BUZ_NM","T"], as_index=False).agg(AMT=("AMT","sum"), CNT=("CNT","sum"))
nat = p.groupby(["TP_BUZ_NM","T"])["AMT"].transform("sum")
p["SHARE"] = p.AMT / nat
p["UNIT"]  = p.AMT / p.CNT

def slope(g, col):           # 6시점 로그 선형추세 기울기(월 %) — 단일월 이상치에 강건
    y = np.log(g[col].replace(0, np.nan)); x = g["T"]
    ok = y.notna()
    if ok.sum() < 4: return np.nan
    return np.polyfit(x[ok], y[ok], 1)[0] * 100

sizes = p.groupby(["REGION","TP_BUZ_NM"])["AMT"].sum()
big   = sizes[sizes > 6e8].index                      # 6개월 6억원↑ 셀만 (소액 셀 노이즈 제거)
g = p.set_index(["REGION","TP_BUZ_NM"]).loc[big].reset_index()
tr = (g.groupby(["REGION","TP_BUZ_NM"])
        .apply(lambda x: pd.Series({"g_share": slope(x,"SHARE"), "g_amt": slope(x,"AMT"),
                                    "g_cnt": slope(x,"CNT"),   "g_unit": slope(x,"UNIT"),
                                    "amt": x.AMT.sum()}), include_groups=False)
        .dropna())
line("1. 계절보정 효과 — 원시 성장률 vs 점유율 기반 성장률")
print(f"분석 셀 {len(tr):,}개 (지역×업종, 6개월 6억원 이상)")
print(f"  원시 금액추세  월 {tr.g_amt.mean():+.2f}% (표준편차 {tr.g_amt.std():.2f}) "
      f"| p10 {tr.g_amt.quantile(.1):+.1f} p90 {tr.g_amt.quantile(.9):+.1f}")
print(f"  점유율 추세    월 {tr.g_share.mean():+.2f}% (표준편차 {tr.g_share.std():.2f}) "
      f"| p10 {tr.g_share.quantile(.1):+.1f} p90 {tr.g_share.quantile(.9):+.1f}")
print(f"  → 전국 공통요인 제거 후 분산 {(1-tr.g_share.std()/tr.g_amt.std())*100:.0f}% 감소")

# ── 2. [후보2 판별] 쇠퇴 분해: 건수축 vs 객단가축이 독립인가 ────────────────
line("2. [후보2] 객단가 분해 — 건수 성장 vs 객단가 성장의 독립성")
r = tr.g_cnt.corr(tr.g_unit)
print(f"  corr(건수추세, 객단가추세) = {r:+.3f}")
print(f"  → {'독립적. 2축 분해 유효 (PASS)' if abs(r)<0.35 else '같이 움직임. 분해 의미 약함 (FAIL)'}")
q = tr.query("amt > 3e9")
dec = q[q.g_share < q.g_share.quantile(.25)]          # 점유율 하락 하위 25% = 쇠퇴 셀
dec = dec.assign(유형=np.where(dec.g_cnt < dec.g_unit, "이탈형(손님이탈)", "위축형(객단가하락)"))
print(f"\n  쇠퇴 셀 {len(dec):,}개 분해: {dict(dec.유형.value_counts())}")
print("\n  [이탈형 상위5]"); print(dec[dec.유형.str.startswith("이탈")].nsmallest(5,"g_cnt")[["g_share","g_cnt","g_unit"]].round(2))
print("\n  [위축형 상위5]"); print(dec[dec.유형.str.startswith("위축")].nsmallest(5,"g_unit")[["g_share","g_cnt","g_unit"]].round(2))

# ── 3. 지역 단위 지표 패널 ─────────────────────────────────────────────────
reg = df.groupby(["REGION","T"], as_index=False).agg(AMT=("AMT","sum"), CNT=("CNT","sum"))
rt  = reg.groupby("REGION").agg(amt=("AMT","sum"),
                                cv=("AMT", lambda s: s.std()/s.mean()*100))   # 월 변동계수
main = rt[rt.amt > 3e10].index                                                # 6개월 300억↑ 지역

kn = df[df.GENDER_CD != "x"]                       # x는 분모에서 제외(지역 편차 때문)
gs = kn.pivot_table(index="REGION", columns="GENDER", values="AMT", aggfunc="sum").fillna(0)
gs = gs.div(gs.sum(1), axis=0) * 100
ag = kn.pivot_table(index="REGION", columns="AGE", values="AMT", aggfunc="sum").fillna(0)
ag = ag.div(ag.sum(1), axis=0) * 100
cs = df.pivot_table(index="REGION", columns="CAT", values="AMT", aggfunc="sum").fillna(0)
bz = df.pivot_table(index="REGION", columns="TP_BUZ_NM", values="AMT", aggfunc="sum").fillna(0)
bzs = bz.div(bz.sum(1), axis=0)

R = pd.DataFrame({
    "amt조": rt.amt/1e12, "cv": rt.cv,
    "외국인%": gs["외국인"], "청년2030%": ag["20대"]+ag["30대"], "고령5060%": ag["50대"]+ag["60대이상"],
    "외식내식비": cs["외식"]/cs["유통"], "HHI": (bzs**2).sum(1),
    "객단가": df.groupby("REGION").AMT.sum()/df.groupby("REGION").CNT.sum(),
}).loc[main]
R["한식객단가"] = (df[df.TP_BUZ_NM=="일반한식"].groupby("REGION").AMT.sum() /
                   df[df.TP_BUZ_NM=="일반한식"].groupby("REGION").CNT.sum())
line(f"3. 지역 지표 패널 ({len(R)}개 주요 시군구) — 상관행렬")
print(R.drop(columns="amt조").corr().round(2))
R.to_csv(OUT/"region_metrics.csv", encoding="utf-8-sig")
tr.to_csv(OUT/"cell_trends.csv", encoding="utf-8-sig")
print(f"\n[save] {OUT/'region_metrics.csv'}, {OUT/'cell_trends.csv'}")
