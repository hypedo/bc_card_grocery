# -*- coding: utf-8 -*-
"""14_matrix.py — 앵커스토어 공백 취약도 × 업종 기회 매트릭스
예측모델이 아니라 규칙 기반 진단 도구. 학습표본 부족·5월 교란 문제를 우회한다.
축1 마트의존도 = 대형할인점 / 장보기3채널
축2 근린성숙도 = 편의점+슈퍼 1인당 침투(건수 기준) 의 전국 백분위
취약 = 의존도 높고 근린 약함 → 앵커 상실 시 수요를 받아줄 그릇이 없음"""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
CH=["대형할인점","슈퍼마켓","편의점"]

H=pd.read_csv("ext/households_raw.csv",thousands=",",encoding="cp949")
H["JK"]=H["행정구역"].str.replace(r"\s*\(\d+\)$","",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
H=H.drop_duplicates(subset="JK").set_index("JK")
def jkey(r):
    p=r.split()
    if p[0]=="세종특별자치시": return "세종특별자치시"
    if len(p)>=3 and p[1].endswith("시"): return f"{p[0]} {p[1]}"
    return r
df["JK"]=df.REGION.map(jkey)
hh=H["2026년01월_전체세대"]; one=H["2026년01월_1인세대"]/hh*100

# 시군구 단위(BC 원해상도) 지표 — 세대수는 결합키 단위라 시 단위 공유분을 배분 없이 참고만
amt=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index="REGION",columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
cnt=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index="REGION",columns="TP_BUZ_NM",values="CNT",aggfunc="sum")
amt=amt[amt.notna().all(axis=1)]; cnt=cnt.loc[amt.index]
M=pd.DataFrame(index=amt.index)
M["마트의존도%"]=amt["대형할인점"]/amt.sum(axis=1)*100
M["근린건수비중%"]=(cnt["편의점"]+cnt["슈퍼마켓"])/cnt.sum(axis=1)*100
M["JK"]=[jkey(r) for r in M.index]
M["세대수"]=M.JK.map(hh); M["1인세대%"]=M.JK.map(one)
M["근린침투"]=(cnt["편의점"]+cnt["슈퍼마켓"])/M["세대수"]      # 세대당 근린 결제건수(시 단위 세대 공유 → 상대비교용)
M["소비규모억"]=df.groupby("REGION").AMT.sum()/1e8
M=M.dropna()

# 전국 백분위 → 2×2
M["의존도P"]=M["마트의존도%"].rank(pct=True)*100
M["근린P"]=M["근린침투"].rank(pct=True)*100
def cls(r):
    hi_dep, hi_nb = r.의존도P>=50, r.근린P>=50
    if hi_dep and not hi_nb: return "A. 취약 (의존高·근린弱)"
    if hi_dep and hi_nb:     return "B. 완충 (의존高·근린强)"
    if not hi_dep and hi_nb: return "C. 전환완료 (의존低·근린强)"
    return "D. 저활성 (의존低·근린弱)"
M["유형"]=M.apply(cls,axis=1)

line("1. 전국 시군구 유형 분포")
g=M.groupby("유형").agg(시군구수=("마트의존도%","size"), 마트의존도=("마트의존도%","mean"),
                      근린침투=("근린침투","mean"), 일인세대=("1인세대%","mean"), 소비규모억=("소비규모억","mean"))
print(g.round(2).to_string())

line("2. A유형(취약) — 앵커 상실 시 받아줄 그릇이 없는 상권")
A=M[M.유형.str.startswith("A")].nlargest(15,"마트의존도%")
print(A[["마트의존도%","근린침투","1인세대%","소비규모억"]].round(2).to_string())

line("3. 검증 — 실제 폐점이 일어난 곳은 어느 유형이었나")
E=pd.read_csv(OUT/"cell_events_v2.csv")
HP={"서울특별시 중랑구","부산광역시 해운대구","부산광역시 사상구","부산광역시 영도구","대구광역시 달서구",
 "인천광역시 서구","인천광역시 남동구","인천광역시 연수구","인천광역시 미추홀구","경기도 하남시",
 "경기도 고양시 덕양구","경기도 수원시 팔달구","경기도 부천시 소사구","경기도 성남시 분당구","경기도 남양주시",
 "경기도 고양시 일산서구","경기도 포천시","충청남도 계룡시","전북특별자치도 김제시","전북특별자치도 익산시",
 "전라남도 목포시","전라남도 순천시","경상북도 경산시","경상북도 포항시 북구","경상북도 포항시 남구",
 "경상남도 김해시","경상남도 창원시 마산회원구","경상남도 밀양시","경상남도 사천시","경상남도 진주시",
 "경상남도 창원시 진해구","서울특별시 송파구","서울특별시 노원구"}
M["폐점"]=M.index.isin(HP)
v=M.groupby("유형").폐점.agg(["sum","size"]); v["폐점률%"]=v["sum"]/v["size"]*100
print(v.round(1).to_string())
print(f"\n  → 폐점은 유형을 가리지 않는다(원인이 기업 재무구조). 매트릭스는 '폐점 예측'이 아니라")
print(f"     '폐점이 났을 때 버틸 수 있는가'를 읽는 도구로만 쓸 것.")

line("4. 유형별 권장 창업 업종 — DID 서열(5월 제외 기준) 적용")
DID={"슈퍼마켓":4.28,"스넥":2.82,"제과점":2.30,"편의점":1.54,"일반한식":0.85,
     "중국음식":0.69,"서양음식":-0.56,"일식회집":-0.75}
print("  대형마트 공백 시 업종별 순효과(%p, 전국 처치군 평균)")
for k,v_ in DID.items(): print(f"   {k:6s} {v_:+5.2f}")
print("\n  A유형(취약)  → 슈퍼마켓·스넥 우선. 대체 수요를 받을 채널 자체가 부족")
print("  B유형(완충)  → 제과점·편의점. 기본 채널은 이미 있어 차별화 업종이 유리")
print("  C유형(전환완료) → 신규 진입 실익 낮음. 기존 점포 객단가 방어가 과제")
print("  D유형(저활성) → 상권 규모 자체가 작아 창업 권장 어려움")
M.drop(columns="JK").to_csv(OUT/"vulnerability_matrix.csv",encoding="utf-8-sig")
print(f"\n[save] {OUT/'vulnerability_matrix.csv'}")
