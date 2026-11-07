# -*- coding: utf-8 -*-
"""11_retune.py — 폐점 탐지기 임계값 재조정
검증2에서 드러난 두 실패: (a) 크기필터 6억이 계룡시(3.4억) 같은 실제 폐점을 탈락시킴
(b) 계단 30% 임계가 다점포 지역(하남 -20%, 송파 -18%)의 폐점을 놓침.
정답(홈플러스 33곳)을 기준으로 임계값별 정밀도·재현율을 그려 최적점을 찾는다."""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
HP={"서울특별시 중랑구","서울특별시 송파구","서울특별시 노원구","부산광역시 해운대구","부산광역시 사상구",
 "부산광역시 영도구","대구광역시 달서구","인천광역시 서구","인천광역시 남동구","인천광역시 연수구",
 "인천광역시 미추홀구","경기도 하남시","경기도 고양시 덕양구","경기도 수원시 팔달구","경기도 부천시 소사구",
 "경기도 성남시 분당구","경기도 남양주시","경기도 고양시 일산서구","경기도 포천시","충청남도 계룡시",
 "전북특별자치도 김제시","전북특별자치도 익산시","전라남도 목포시","전라남도 순천시","경상북도 경산시",
 "경상북도 포항시 북구","경상북도 포항시 남구","경상남도 김해시","경상남도 창원시 마산회원구",
 "경상남도 밀양시","경상남도 사천시","경상남도 진주시","경상남도 창원시 진해구"}

cell=df[df.TP_BUZ_NM=="대형할인점"].groupby(["REGION","T"],as_index=False).AMT.sum()
nat=cell.groupby("T").AMT.sum(); nidx=(nat/nat.mean())
cell["ADJ"]=cell.AMT/cell["T"].map(nidx)
piv=cell.pivot_table(index="REGION",columns="T",values="ADJ")
size=cell.groupby("REGION").AMT.sum()
piv=piv[piv.notna().all(axis=1)&(piv>0).all(axis=1)]

def scan(y):
    ly=np.log(y); best=(0.0,np.nan,np.inf)
    for k in range(1,6):
        a,b=ly[:k],ly[k:]
        d=np.median(b)-np.median(a)
        sp=np.sqrt((a.var(ddof=0)*len(a)+b.var(ddof=0)*len(b))/6)
        if abs(d)>abs(best[0]): best=(d,k,sp)
    return best
S=pd.DataFrame([(r,*scan(v.values.astype(float))) for r,v in piv.iterrows()],
               columns=["REGION","delta","k","spread"]).set_index("REGION")
S["선명도"]=S.delta.abs()/(S.spread+1e-9); S["규모"]=size.reindex(S.index)
S["변화율%"]=(np.exp(S.delta)-1)*100

line("1. 임계값 격자 탐색 — 정답은 홈플러스 폐점 33곳")
rows=[]
for szmin in [0, 1e8, 3e8, 6e8]:
    for step in [0.15,0.20,0.25,0.30]:
        for sh in [2.0,2.5,3.0]:
            sub=S[(S.규모>szmin)]
            det=set(sub[(sub.delta< -np.log(1+step))&(sub.선명도>sh)].index)
            if not det: continue
            tp=len(det&HP); pre=tp/len(det); rec=tp/len(HP&set(S.index))
            f1=2*pre*rec/(pre+rec) if pre+rec else 0
            rows.append((szmin/1e8,step,sh,len(det),tp,pre*100,rec*100,f1*100))
G=pd.DataFrame(rows,columns=["최소규모(억)","계단","선명도","탐지수","적중","정밀도%","재현율%","F1"])
print("  [F1 상위 10개 조합]")
print(G.nlargest(10,"F1").round(1).to_string(index=False))
print(f"\n  현재 설정(6억/30%/3.0): " +
      G.query("`최소규모(억)`==6 and 계단==0.30 and 선명도==3.0")[["탐지수","적중","정밀도%","재현율%","F1"]].round(1).to_string(index=False))

line("2. 권장 설정 적용 결과")
b=G.nlargest(1,"F1").iloc[0]
sub=S[S.규모>b["최소규모(억)"]*1e8]
det=set(sub[(sub.delta< -np.log(1+b.계단))&(sub.선명도>b.선명도)].index)
print(f"  설정: 최소규모 {b['최소규모(억)']:.0f}억 / 계단 {b.계단*100:.0f}% / 선명도 {b.선명도}")
print(f"  탐지 {len(det)}곳 중 홈플러스 확인 {len(det&HP)}곳 "
      f"(정밀도 {b['정밀도%']:.0f}% · 재현율 {b['재현율%']:.0f}% · F1 {b.F1:.0f})")
print(f"\n  ✅ 새로 회수된 지역: {', '.join(sorted(det&HP - {'부산광역시 영도구','부산광역시 해운대구','서울특별시 중랑구','경기도 포천시','전북특별자치도 김제시','경상남도 창원시 마산회원구'}))}")
print(f"\n  [탐지 목록] ●=홈플러스 확인")
for r in sorted(det, key=lambda x: S.loc[x,"변화율%"]):
    print(f"   {'●' if r in HP else '○'} {r:24s} {S.loc[r,'변화율%']:+7.1f}%  규모 {S.loc[r,'규모']/1e8:7.1f}억  k={int(S.loc[r,'k'])}")
S.to_csv(OUT/"mart_step_scan.csv",encoding="utf-8-sig")
