# v7_ann 설계 계획서 — Stacking + Optuna HPO + LightGBM Meta

## 개요

v6_stacking(Ridge meta, R² 0.7754) 위에 **2가지 업그레이드**를 적용해 R² 0.80+ 돌파를 목표로 합니다.

## v6_stacking 대비 차별점 (3가지)

### 1. XGBoost/LightGBM Optuna HPO
- 현재 v6_stacking은 고정 하이퍼파라미터 (기본값 + 일부 수동 설정)
- v7: **Optuna TPE 30 trial × 3-fold**로 각 모델별 최적 탐색
- Search space:
```
n_estimators:  {500, 1000, 2000}
max_depth:     {4, 6, 8, 10}
learning_rate: loguniform(0.01, 0.1)
subsample:     uniform(0.7, 1.0)
colsample_bytree: uniform(0.7, 1.0)
min_child_weight: {1, 3, 5}   # XGB
num_leaves: {15, 31, 63, 127}  # LGB
```
- Objective: `mean(CV R²) - 0.5*std(CV R²)` (robust)

### 2. LightGBM Meta-Learner
- Ridge (선형) → LightGBM(depth=3, 100 estimators)
- Nonlinear 결합으로 base 모델 간 복잡한 상호작용 포착
- 과적합 방지: depth 제한 + early stopping

### 3. CatBoost 추가 (선택, 기본 포함)
- 4번째 base model로 CatBoost 추가
- Tree 계열이지만 XGB/LGB와 미세하게 다른 inductive bias

## v5/v6와 동일한 부분
- 데이터, split, Stratified + Scaffold CV
- 피처: Morgan + MACCS + RDKit → VT + MI top-300 (ANN)
- 모델: ResidualMLP(256×3, proj LN + head LN)
- Mixup α=0.3, p=0.5
- Adam lr=1e-3, WD=5e-4, EPOCHS 200, PATIENCE 25
- 5-seed Snapshot [42, 53, 65, 79, 93]

## 예상 성과

| 구성 | 기대 R² | 누적 효과 |
|---|---|---|
| v6_stacking (baseline) | 0.7754 | — |
| + XGB/LGB Optuna HPO | 0.79~0.81 | +0.02~0.04 |
| + LightGBM meta | 0.80~0.82 | +0.01~0.02 |
| + CatBoost base | 0.80~0.83 | +0.01~0.02 |
| **최종 기대** | **0.80~0.83** | **+0.03~0.07** |

## 예상 실행 시간

| 단계 | 소요 |
|---|---|
| v5/v6와 공통 (feature, CV, ANN final) | ~1.5분 |
| Optuna HPO (XGB 30 trial × 3-fold) | ~3~5분 |
| Optuna HPO (LGB 30 trial × 3-fold) | ~3~5분 |
| CatBoost 5-fold OOF | ~2~3분 |
| 최종 OOF 수집 + Meta-learner | ~1분 |
| **총** | **~10~15분** |

30분 예산 내 여유.

## 리스크

- **Optuna HPO 시간**: 복잡한 데이터에서 30 trial 부족할 수 있음. ASHA pruner로 보완.
- **LightGBM meta 과적합**: base가 3개뿐이라 meta는 단순해야 함. depth=3 제한 + early stopping 필수.
- **CatBoost 의존성**: 추가 패키지 필요 (소요 10MB 다운로드).
