# -*- coding: utf-8 -*-
"""12_opportunity.py — 앵커스토어 공백 상권 기회 모델 (1단계: 이질성 구조 확인)
질문: 대형마트가 빠진 자리를 어떤 업종이 흡수하는가, 그리고 그건 상권마다 다른가?
종속변수 = 처치군 23곳 × 8업종의 흡수효과(사후/사전 로그변화, 전국 업종패턴 보정)
독립변수 = 업종 + 상권 특성(1인세대·연령·규모·업종다양성·채널믹스)"""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet"); E=pd.read_csv(OUT/"cell_events_v2.csv")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
K=3
HP={"서울특별시 중랑구","서울특별시 송파구","서울특별시 노원구","부산광역시 해운대구","부산광역시 사상구",
 "부산광역시 영도구","대구광역시 달서구","인천광역시 서구","인천광역시 남동구","인천광역시 연수구",
 "인천광역시 미추홀구","경기도 하남시","경기도 고양시 덕양구","경기도 수원시 팔달구","경기도 부천시 소사구",
 "경기도 성남시 분당구","경기도 남양주시","경기도 고양시 일산서구","경기도 포천시","충청남도 계룡시",
 "전북특별자치도 김제시","전북특별자치도 익산시","전라남도 목포시","전라남도 순천시","경상북도 경산시",
 "경상북도 포항시 북구","경상북도 포항시 남구","경상남도 김해시","경상남도 창원시 마산회원구",
 "경상남도 밀양시","경상남도 사천시","경상남도 진주시","경상남도 창원시 진해구"}
DET=set(E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].REGION)
TREAT=HP|DET                      # 실제 확인 + 데이터 탐지 둘 다 처치군으로

# ── 결과변수: 업종별 사후/사전 변화 (전국 업종×월 보정) ────────────────────
cell=df.groupby(["REGION","TP_BUZ_NM","T"],as_index=False).AMT.sum()
nb=cell.groupby(["TP_BUZ_NM","T"]).AMT.sum()
cell=cell.join((nb/nb.groupby("TP_BUZ_NM").transform("mean")).rename("bidx"),on=["TP_BUZ_NM","T"])
cell["ADJ"]=cell.AMT/cell.bidx
has=set(cell[cell.TP_BUZ_NM=="대형할인점"].REGION)
o=cell[(cell.TP_BUZ_NM!="대형할인점")&cell.REGION.isin(has)]
s=o.pivot_table(index=["REGION","TP_BUZ_NM"],columns="T",values="ADJ")
s=s[(s.notna().all(axis=1))&(s>0).all(axis=1)]
y=np.log(s.loc[:,s.columns>=K].mean(axis=1)/s.loc[:,s.columns<K].mean(axis=1))*100
D=y.rename("chg").reset_index()
D["TREAT"]=D.REGION.isin(TREAT).astype(int)
D=D[D.TP_BUZ_NM.isin(["슈퍼마켓","편의점","일반한식","서양음식","스넥","제과점","중국음식","일식회집"])]
# 업종 고정효과 = 대조군 평균 → 처치군 초과분만 남김
base=D[D.TREAT==0].groupby("TP_BUZ_NM").chg.mean()
D["효과"]=D.chg-D.TP_BUZ_NM.map(base)
T=D[D.TREAT==1].copy()
print(f"[표본] 처치 시군구 {T.REGION.nunique()}곳 × 업종 = {len(T)}셀 "
      f"(실제확인 {len(HP&has)} + 탐지 {len(DET-HP)})")

line("1. 업종별 평균 흡수효과 — 어디에 기회가 있나")
g=T.groupby("TP_BUZ_NM").효과.agg(["mean","std","size"]).sort_values("mean",ascending=False)
g["양(+)비율%"]=T.groupby("TP_BUZ_NM").효과.apply(lambda x:(x>0).mean()*100)
print(g.round(2).to_string())
print(f"\n  → 평균이 높아도 표준편차가 크면 '어디서나 되는 기회'가 아니라 '상권을 타는 기회'")

# ── 상권 특성 ──────────────────────────────────────────────────────────────
H=pd.read_csv("ext/households_raw.csv",thousands=",",encoding="cp949")
H["JK"]=H["행정구역"].str.replace(r"\s*\(\d+\)$","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
H=H.drop_duplicates(subset="JK").set_index("JK")
def jkey(r):
    p=r.split()
    if p[0]=="세종특별자치시": return "세종특별자치시"
    if len(p)>=3 and p[1].endswith("시"): return f"{p[0]} {p[1]}"
    return r
one=(H["2026년01월_1인세대"]/H["2026년01월_전체세대"]*100)
kn=df[df.GENDER_CD!="x"]
ag=kn.pivot_table(index="REGION",columns="AGE",values="AMT",aggfunc="sum"); ag=ag.div(ag.sum(axis=1),axis=0)*100
bz=df.pivot_table(index="REGION",columns="TP_BUZ_NM",values="AMT",aggfunc="sum").fillna(0)
bzs=bz.div(bz.sum(axis=1),axis=0)
ch=df[df.TP_BUZ_NM.isin(["대형할인점","슈퍼마켓","편의점"])].pivot_table(index="REGION",columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
X=pd.DataFrame({
  "1인세대%":pd.Series({r:one.get(jkey(r),np.nan) for r in bz.index}),
  "청년2030%":ag["20대"]+ag["30대"], "고령60%":ag["60대이상"],
  "규모log":np.log(df.groupby("REGION").AMT.sum()),
  "업종다양성":1-(bzs**2).sum(axis=1),
  "마트의존도%":(ch["대형할인점"]/ch.sum(axis=1)*100),
})
T=T.join(X,on="REGION").dropna()
line("2. 흡수효과를 좌우하는 상권 특성 (처치군 내 상관)")
print(T[["효과","1인세대%","청년2030%","고령60%","규모log","업종다양성","마트의존도%"]].corr()["효과"].drop("효과").round(3).to_string())
line("3. 지역 단위 총 흡수력 — 누가 버텼고 누가 못 버텼나")
R=T.groupby("REGION").효과.mean().sort_values()
RX=X.loc[R.index]
print("  [흡수 실패 하위5]"); print(R.head(5).round(2).to_string())
print("\n  [흡수 성공 상위5]"); print(R.tail(5).round(2).to_string())
print(f"\n  지역 흡수력과 상권특성 상관:")
print(pd.concat([R.rename("흡수력"),RX],axis=1).corr()["흡수력"].drop("흡수력").round(3).to_string())
T.to_csv(OUT/"opportunity_cells.csv",encoding="utf-8-sig")
pd.concat([R.rename("흡수력"),RX],axis=1).to_csv(OUT/"opportunity_regions.csv",encoding="utf-8-sig")
print(f"\n[save] {OUT/'opportunity_cells.csv'}, {OUT/'opportunity_regions.csv'}")
