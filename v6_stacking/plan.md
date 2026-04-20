# v6_stacking 설계 계획서

## 개요

v5_ann 파이프라인 위에 **Stacking Ensemble** (ANN + XGBoost + LightGBM + Ridge meta-learner)을 구축해 R² 추가 부스트를 검증합니다. v6_chemberta와 동일 조건에서 실행해 비교하는 것이 목표.

## 차별점 (v5 대비 유일한 차이)

v5 파이프라인을 그대로 유지하면서 base 모델 2개 추가 + meta-learner 추가:
- v5 ResidualMLP는 **base 모델 #1** (5-fold OOF + 5-seed test)
- **XGBoost**: VT 후 504차원 입력, 5-fold OOF (base #2)
- **LightGBM**: VT 후 504차원 입력, 5-fold OOF (base #3)
- **Ridge meta-learner**: OOF 3개로 학습, Test 예측 합성

## v5와 동일한 부분 (Fair Comparison)
- 데이터 (2117, RS=42), Train/Test split, Stratified/Scaffold 이중 CV
- 피처: Morgan + MACCS + RDKit → VT(0.01) → MI top-300 (ANN용)
- 모델 아키텍처 및 학습: ResidualMLP(256×3), Mixup α=0.3 p=0.5, WD 5e-4, EPOCHS 200 PATIENCE 25
- 5-seed Snapshot [42, 53, 65, 79, 93]
- 평가 지표, 시각화, 시간 예산 ≤ 30분

## 예상 성과
- **Test R²: 0.78~0.82** (v5 ensemble 0.7536 대비 +0.03~0.07)
- **Test MAE: 26~30 K** (v5 32.0 대비 개선)
- Ridge 계수로 base 모델 상대 중요도 확인

## 논문 근거
- Sushko et al. 2011 (JCIM) — ADMET stacking
- Tetko et al. — OCHEM 플랫폼의 표준 방식
- Kaggle Merck 2012 우승 솔루션 — stacking + DNN

## 예상 실행 시간
v5 1.1분 + XGB 5-fold (~2분) + LGB 5-fold (~2분) + Ridge 즉시 = **~5~6분**
