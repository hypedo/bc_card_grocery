# -*- coding: utf-8 -*-
"""17_final_model.py — 최종 모델: 마트 이탈 위험 예측 (2변수)
설계: 피처 1~3월 / 종속변수 4~6월 (누출 없음), 3채널 내 비중 기반 (5월 교란 회피)
결과: 구조변수 11개는 기여 없음 → 2변수가 최선. 단순함이 곧 결론이다."""
import pandas as pd, numpy as np, pathlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import spearmanr
import io, contextlib
OUT=pathlib.Path("out")
g={}; 
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("src/16_predict.py").read().split("# ── 베이스라인")[0],g)
X,y,sh=g["X"],g["y"],g["sh"]
line=lambda t: print(f"\n{'='*78}\n{t}\n{'='*78}")
F=["마트비중","마트비중추세"]
cv=KFold(5,shuffle=True,random_state=0)
pred=pd.Series(cross_val_predict(LinearRegression(),X[F],y,cv=cv),index=y.index)
r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()

line("최종 모델 성능 (5-fold 교차검증, n=197)")
print(f"  R² = {r2:+.3f} | Spearman = {spearmanr(y,pred).statistic:+.3f} | MAE = {(y-pred).abs().mean():.2f}%p")
rng=np.random.default_rng(1)
null=[1-((y.values[rng.permutation(len(y))]-pred.values)**2).sum()/((y-y.mean())**2).sum() for _ in range(10000)]
print(f"  순열검정 p = {np.mean(np.array(null)>=r2):.4f}")
print(f"\n  [비교] 전국평균 R²=0.000 | 단순 지속예측 R²=-0.383 | 13변수 R²=+0.249")
print(f"  → 단순 지속예측은 평균보다도 나쁘다. 최근 추세를 그대로 믿으면 틀린다.")

m=LinearRegression().fit(X[F],y)
print(f"\n  회귀식:  변화(%p) = {m.intercept_:+.3f} {m.coef_[0]:+.4f}×마트비중 {m.coef_[1]:+.4f}×마트비중추세")
print(f"  해석: 마트비중 10%p 높을수록 이후 3개월 {m.coef_[0]*10:+.2f}%p 더 이탈")

line("전국 시군구 마트 이탈 위험 예측 — 상위 15곳")
full=pd.Series(m.predict(X[F]),index=y.index)
R=pd.DataFrame({"1~3월 마트비중%":X.마트비중,"1~3월 추세":X.마트비중추세,
                "예측 4~6월 변화%p":full,"실제%p":y,"오차":y-full}).round(2)
print(R.nsmallest(15,"예측 4~6월 변화%p").to_string())

line("검증 — 예측 위험 상위 25%가 실제로 더 빠졌는가")
q=full.quantile(.25)
hi,lo=y[full<=q],y[full>q]
print(f"  위험군 {len(hi)}곳 실제 {hi.mean():+.2f}%p | 그 외 {len(lo)}곳 {lo.mean():+.2f}%p | 차이 {hi.mean()-lo.mean():+.2f}%p")
v=y.values; n=len(hi); rg=np.random.default_rng(5)
nl=[(lambda p:p[:n].mean()-p[n:].mean())(v[rg.permutation(len(v))]) for _ in range(20000)]
print(f"  순열검정 p = {np.mean(np.abs(nl)>=abs(hi.mean()-lo.mean())):.4f}")

line("구조변수가 왜 필요 없는가 — 이것이 결론이다")
print("  1인세대·연령믹스·업종다양성·규모를 넣어도 예측이 나아지지 않는다(오히려 과적합).")
print("  즉 다운사이징은 상권 특성을 가리지 않는다. 마트 의존도가 높으면 어디든 빠진다.")
print("  이는 전 연령 동시 이탈(20대이하 -7.4% ~ 60대이상 -1.6%)과 같은 이야기이며,")
print("  1인가구화 가설 기각을 예측 모델 쪽에서 한 번 더 뒷받침한다.")
R.to_csv(OUT/"risk_prediction.csv",encoding="utf-8-sig")
print(f"\n[save] {OUT/'risk_prediction.csv'}")
