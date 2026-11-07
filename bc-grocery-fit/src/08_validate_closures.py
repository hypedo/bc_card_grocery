# -*- coding: utf-8 -*-
"""08_validate_closures.py — [검증2] 대형할인점 폐점의 외부 검증
외부 출처: 나무위키 '홈플러스/지점' + 블로그 종합 — 홈플러스 2026.5.10 영업중단 / 6.4 폐점 확정 37개점
※ 출처가 위키·블로그이며 목록 간 불일치가 있어 '참고 검증'으로만 사용한다."""
import pandas as pd, numpy as np, pathlib
OUT=pathlib.Path("out"); df=pd.read_parquet(OUT/"clean.parquet"); E=pd.read_csv(OUT/"cell_events_v2.csv")
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")

# 홈플러스 2026 폐점 점포 → 소재 시군구 (나무위키 기준)
HP={
 "서울특별시 중랑구":["면목점","신내점"], "서울특별시 송파구":["잠실점"], "서울특별시 노원구":["중계점"],
 "부산광역시 해운대구":["부산반여점","센텀시티점"], "부산광역시 사상구":["서부산점"], "부산광역시 영도구":["영도점"],
 "대구광역시 달서구":["상인점"], "인천광역시 서구":["가좌점"], "인천광역시 남동구":["인천논현점"],
 "인천광역시 연수구":["인천송도점","인천연수점"], "인천광역시 미추홀구":["인천숭의점"],
 "경기도 하남시":["경기하남점"], "경기도 고양시 덕양구":["고양터미널점"], "경기도 수원시 팔달구":["동수원점"],
 "경기도 부천시 소사구":["부천소사점"], "경기도 성남시 분당구":["분당오리점"], "경기도 남양주시":["진접점"],
 "경기도 고양시 일산서구":["킨텍스점"], "경기도 포천시":["포천송우점"], "충청남도 계룡시":["계룡점"],
 "전북특별자치도 김제시":["김제점"], "전북특별자치도 익산시":["익산점"], "전라남도 목포시":["목포점"],
 "전라남도 순천시":["순천풍덕점"], "경상북도 경산시":["경산점"], "경상북도 포항시 북구":["죽도점"],
 "경상북도 포항시 남구":["포항점"], "경상남도 김해시":["김해점"], "경상남도 창원시 마산회원구":["마산점"],
 "경상남도 밀양시":["밀양점"], "경상남도 사천시":["삼천포점"], "경상남도 진주시":["진주점"],
 "경상남도 창원시 진해구":["진해점"],
}
have=set(df[df.TP_BUZ_NM=="대형할인점"].REGION)
hp={k:v for k,v in HP.items() if k in have}
print(f"[목록] 홈플러스 폐점 시군구 {len(HP)}곳 중 대형할인점 데이터 보유 {len(hp)}곳 "
      f"(미보유 {sorted(set(HP)-have)})")

det=set(E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].REGION)
line("1. 우리 탐지 결과 vs 홈플러스 실제 폐점 목록")
hit=det&set(hp); only_det=det-set(HP); only_hp=set(hp)-det
print(f"  ✅ 양쪽 모두({len(hit)}): {', '.join(sorted(hit))}")
print(f"\n  ⚠️ 우리만 탐지({len(only_det)}): {', '.join(sorted(only_det))}")
print(f"\n  ⚠️ 홈플러스만({len(only_hp)}): {', '.join(sorted(only_hp))}")
print(f"\n  정밀도(우리 탐지 중 홈플러스 확인) {len(hit)}/{len(det)} = {len(hit)/len(det)*100:.0f}%")
print(f"  재현율(홈플러스 중 우리가 탐지) {len(hit)}/{len(hp)} = {len(hit)/len(hp)*100:.0f}%")

# ── 2. 타이밍 — 홈플러스는 5월 중단인데 우리 탐지는 언제인가 ────────────────
line("2. 타이밍 검증 — 홈플러스 영업중단은 2026.5.10")
k=E[(E.TP_BUZ_NM=="대형할인점")&(E.유형=="폐점의심")].set_index("REGION").k
print(f"  우리 탐지 k 분포: {dict(k.value_counts().sort_index())}   (k=하락 시작 시점, k=4→202605, k=5→202606)")
print(f"  홈플러스 확인 지역의 k: {dict(k[list(hit)].value_counts().sort_index())}")
print("  ※ 5월 중단이면 k=4 또는 5여야 함")

# ── 3. 통계 검정 — 목록 의존 없이: 홈플러스 지역이 실제로 더 떨어졌나 ───────
line("3. 통계 검정 — 홈플러스 폐점 시군구의 5~6월 대형할인점 급감")
m=df[df.TP_BUZ_NM=="대형할인점"].groupby(["REGION","T"]).AMT.sum().unstack()
m=m[(m.notna().all(axis=1))&(m.min(axis=1)>0)]
nat=m.sum(); nidx=nat/nat.mean()
adj=m.div(nidx,axis=1)                                        # 전국 대형할인점 월 패턴 보정
chg=np.log(adj[[4,5]].mean(axis=1)/adj[[0,1,2,3]].mean(axis=1))*100   # 5~6월 vs 1~4월
g=pd.Series(np.where(chg.index.isin(hp),"홈플러스폐점","그외"),index=chg.index)
print(f"  홈플러스 폐점 {(g=='홈플러스폐점').sum()}곳: {chg[g=='홈플러스폐점'].mean():+.2f}%  (중앙값 {chg[g=='홈플러스폐점'].median():+.2f}%)")
print(f"  그 외 {(g=='그외').sum()}곳:        {chg[g=='그외'].mean():+.2f}%  (중앙값 {chg[g=='그외'].median():+.2f}%)")
obs=chg[g=="홈플러스폐점"].mean()-chg[g=="그외"].mean()
rng=np.random.default_rng(3); n=(g=="홈플러스폐점").sum(); v=chg.values
null=[(lambda p: p[:n].mean()-p[n:].mean())(v[rng.permutation(len(v))]) for _ in range(20000)]
p=np.mean(np.abs(null)>=abs(obs))
print(f"\n  차이 {obs:+.2f}%p | 순열검정 p = {p:.4f} → {'유의 (p<0.05)' if p<0.05 else '유의하지 않음'}")
print(f"\n  홈플러스 지역 하락폭 TOP8:"); print(chg[g=="홈플러스폐점"].nsmallest(8).round(1).to_string())
pd.DataFrame({"chg_5_6월%":chg,"그룹":g}).to_csv(OUT/"closure_validation.csv",encoding="utf-8-sig")
