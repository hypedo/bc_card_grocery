# -*- coding: utf-8 -*-
"""15_matrix_v2.py — 취약도 매트릭스 v2 (외부 분모 제거)
v1 오류: 근린침투 = 근린건수/세대수 인데 세대수가 시 단위(수원시)·건수가 구 단위(장안구)라
         다구도시가 33% 수준으로 과소평가됨. → 외부 분모를 쓰지 않는 내부 비율로 교체."""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
CH=["대형할인점","슈퍼마켓","편의점"]
HP={"서울특별시 중랑구","부산광역시 해운대구","부산광역시 사상구","부산광역시 영도구","대구광역시 달서구",
 "인천광역시 서구","인천광역시 남동구","인천광역시 연수구","인천광역시 미추홀구","경기도 하남시",
 "경기도 고양시 덕양구","경기도 수원시 팔달구","경기도 부천시 소사구","경기도 성남시 분당구","경기도 남양주시",
 "경기도 고양시 일산서구","경기도 포천시","충청남도 계룡시","전북특별자치도 김제시","전북특별자치도 익산시",
 "전라남도 목포시","전라남도 순천시","경상북도 경산시","경상북도 포항시 북구","경상북도 포항시 남구",
 "경상남도 김해시","경상남도 창원시 마산회원구","경상남도 밀양시","경상남도 사천시","경상남도 진주시",
 "경상남도 창원시 진해구","서울특별시 송파구","서울특별시 노원구"}

amt=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index="REGION",columns="TP_BUZ_NM",values="AMT",aggfunc="sum")
cnt=df[df.TP_BUZ_NM.isin(CH)].pivot_table(index="REGION",columns="TP_BUZ_NM",values="CNT",aggfunc="sum")
amt=amt[amt.notna().all(axis=1)]; cnt=cnt.loc[amt.index]
M=pd.DataFrame(index=amt.index)
M["마트의존도%"]=amt["대형할인점"]/amt.sum(axis=1)*100          # 금액 기준 앵커 의존
M["근린빈도%"]=(cnt["편의점"]+cnt["슈퍼마켓"])/cnt.sum(axis=1)*100  # 건수 기준 근린 이용 습관 (내부비율)
M["소비규모억"]=df.groupby("REGION").AMT.sum()/1e8
M["다구도시"]=[len(i.split())>=3 and i.split()[1].endswith("시") for i in M.index]
line("1. 지표 교체 후 다구도시 편향이 사라졌는가")
for c in ["마트의존도%","근린빈도%"]:
    a,b=M[M.다구도시][c].mean(),M[~M.다구도시][c].mean()
    print(f"  {c:10s} 다구 {a:6.2f} | 단일 {b:6.2f} | 비율 {a/b*100:5.0f}%")
print("  → 100%에 가까우면 편향 없음")

M["의존도P"]=M["마트의존도%"].rank(pct=True)*100
M["근린P"]=M["근린빈도%"].rank(pct=True)*100
M["유형"]=np.where(M.의존도P>=50,
        np.where(M.근린P>=50,"B. 완충 (의존高·근린高)","A. 취약 (의존高·근린低)"),
        np.where(M.근린P>=50,"C. 전환완료 (의존低·근린高)","D. 저활성 (의존低·근린低)"))
M["폐점"]=M.index.isin(HP)
line("2. 유형 분포")
g=M.groupby("유형").agg(시군구수=("폐점","size"),마트의존도=("마트의존도%","mean"),
      근린빈도=("근린빈도%","mean"),소비규모억=("소비규모억","mean"),폐점수=("폐점","sum"))
g["폐점률%"]=g.폐점수/g.시군구수*100
print(g.round(2).to_string())

line("3. A유형 폐점 집중은 지표 수정 후에도 남는가")
a,o=M[M.유형.str.startswith("A")],M[~M.유형.str.startswith("A")]
obs=a.폐점.mean()-o.폐점.mean()
rng=np.random.default_rng(4); v=M.폐점.values.astype(float); n=len(a)
null=[(lambda p:p[:n].mean()-p[n:].mean())(v[rng.permutation(len(v))]) for _ in range(20000)]
p=np.mean(np.abs(null)>=abs(obs))
print(f"  A유형 {a.폐점.sum()}/{len(a)} = {a.폐점.mean()*100:.1f}%  |  그외 {o.폐점.sum()}/{len(o)} = {o.폐점.mean()*100:.1f}%")
print(f"  차이 {obs*100:+.1f}%p, p = {p:.4f} → {'유의' if p<0.05 else '유의하지 않음'}")
print(f"  규모: A유형 {a.소비규모억.mean():.0f}억 vs 그외 {o.소비규모억.mean():.0f}억")
print(f"  다구도시 비중: A유형 {a.다구도시.mean()*100:.0f}% vs 그외 {o.다구도시.mean()*100:.0f}%")

line("4. A유형(취약) 상위 15곳")
print(M[M.유형.str.startswith("A")].nlargest(15,"마트의존도%")[["마트의존도%","근린빈도%","소비규모억","폐점"]].round(2).to_string())
M.drop(columns="다구도시").to_csv(OUT/"vulnerability_matrix_v2.csv",encoding="utf-8-sig")
print(f"\n[save] {OUT/'vulnerability_matrix_v2.csv'}")
