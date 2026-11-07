# -*- coding: utf-8 -*-
"""01_clean.py — 원본 CSV 정규화 + 품질 진단 + 파생변수 생성"""
import pandas as pd, numpy as np, pathlib

RAW = pathlib.Path("ABP_CONTEST_DATA.csv")
OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)

GENDER = {"1": "남성", "2": "여성", "3": "외국인", "x": "미상"}
AGE    = {"1": "20대이하", "2": "20대", "3": "30대", "4": "40대",
          "5": "50대", "6": "60대이상", "x": "미상"}
RETAIL = {"대형할인점", "편의점", "슈퍼마켓"}          # 식료품 유통
DINE   = {"일반한식", "갈비전문점", "한정식", "일식회집",
          "중국음식", "서양음식", "스넥", "제과점"}     # 외식
SPARSE = {"갈비전문점", "한정식"}                      # 희소 업종(전국 6개월 100억 미만)

df = pd.read_csv(RAW, encoding="utf-8-sig", dtype=str)
df.columns = [c.strip().upper() for c in df.columns]
print(f"[load] {len(df):,}행 {list(df.columns)}")

# --- 1) 정규화 -------------------------------------------------------------
df["TP_BUZ_NM"] = df["TP_BUZ_NM"].str.replace(r"\s+", "", regex=True)  # "편 의 점" → "편의점"
df["GENDER_CD"] = df["GENDER_CD"].str.lower().str.strip()
df["AGE_CD"]    = df["AGE_CD"].str.lower().str.strip()
df["REGION"]    = df["SIDO_NM"].str.strip() + " " + df["CCG_NM"].str.strip()  # 복합키(중구·남구 중복 방지)
df["YM"]        = df["STRD_YYMM"].astype(int)
df["T"]         = df.groupby("YM", sort=True).ngroup()                        # 0~5 시점 인덱스
for c in ("AMT", "CNT"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["GENDER"] = df["GENDER_CD"].map(GENDER)
df["AGE"]    = df["AGE_CD"].map(AGE)
df["CAT"]    = np.where(df["TP_BUZ_NM"].isin(RETAIL), "유통",
               np.where(df["TP_BUZ_NM"].isin(DINE), "외식", "기타"))
df["SPARSE"] = df["TP_BUZ_NM"].isin(SPARSE)
df["UNIT"]   = df["AMT"] / df["CNT"]        # 객단가

assert df["CAT"].ne("기타").all(), "미분류 업종 존재"
assert df[["AMT", "CNT"]].notna().all().all(), "AMT/CNT 결측"

# --- 2) 품질 진단 ----------------------------------------------------------
print("\n[진단] 차원")
for c in ("YM", "SIDO_NM", "REGION", "TP_BUZ_NM", "GENDER", "AGE"):
    print(f"  {c:12s} {df[c].nunique():>6,}개")

# 2-1. 객단가 이상치: 하한은 물리적으로 불가능 → 오류. 상한은 희소 고가업종일 수 있어 보존.
df["FLAG_LOW"]  = df["UNIT"] < 1_000       # 1천원 미만 = 집계 오류
df["FLAG_HIGH"] = df["UNIT"] > 300_000     # 30만원 초과 = 검토 대상(삭제 안 함)
low = df[df["FLAG_LOW"]]
print(f"\n[진단] 객단가 1천원 미만(오류 판정·제거 대상) {len(low)}건")
for _, r in low.iterrows():
    print(f"   {r.YM} {r.REGION} {r.TP_BUZ_NM} {r.GENDER}{r.AGE} "
          f"AMT={r.AMT:,.0f} CNT={r.CNT:,.0f} 객단가={r.UNIT:,.0f}원")
hi = df[df["FLAG_HIGH"]]
print(f"\n[진단] 객단가 30만원 초과(보존·플래그만) {len(hi)}건 "
      f"— 업종 구성: {dict(hi['TP_BUZ_NM'].value_counts())}")

# 2-2. 성별 미상(x) 비중의 지역 편차 → 비중 계산 시 분모에서 제외해야 하는 근거
na_share = (df.assign(na=np.where(df.GENDER_CD == "x", df.AMT, 0))
              .groupby("REGION").agg(na=("na", "sum"), tot=("AMT", "sum")))
na_share["pct"] = na_share.na / na_share.tot * 100
print(f"\n[진단] 성별미상(x) 비중 — 전국 {df.loc[df.GENDER_CD=='x','AMT'].sum()/df.AMT.sum()*100:.1f}%, "
      f"지역 중앙값 {na_share.pct.median():.1f}%, 최대 {na_share.pct.max():.1f}%")
print("  상위 6개 지역(제휴카드 회원정보 수집 편차로 추정):")
for reg, r in na_share.sort_values("pct", ascending=False).head(6).iterrows():
    print(f"   {reg:22s} {r.pct:5.1f}%")

# --- 3) 정제본 저장 --------------------------------------------------------
clean = df[~df["FLAG_LOW"]].copy()
print(f"\n[clean] {len(df):,} → {len(clean):,}행 (제거 {len(df)-len(clean)}행)")
print(f"        총 금액 {clean.AMT.sum()/1e12:.2f}조원 / 총 건수 {clean.CNT.sum()/1e8:.2f}억건")

keep = ["YM","T","SIDO_NM","CCG_NM","REGION","TP_BUZ_NM","CAT","SPARSE",
        "GENDER_CD","GENDER","AGE_CD","AGE","AMT","CNT","UNIT","FLAG_HIGH"]
clean[keep].to_parquet(OUT/"clean.parquet", index=False)
print(f"[save] {OUT/'clean.parquet'}")
