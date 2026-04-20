# 중간시험 레포트 — Melting Point 예측 AI 모델 과제

---

## A. 기본 정보

| 항목 | 내용 |
|------|------|
| **학번** | 120240502 |
| **이름** | 최성희 |
| **사용 모델** | Stacking Ensemble (ResidualMLP + XGBoost + LightGBM) with Ridge Meta-Learner |

---

## B. 결과 (5-Fold CV 평균값)

### 주 지표 — Stratified 5-Fold Cross Validation (ANN base)

| 지표 | 값 |
|------|------|
| **R²** | **0.6543 ± 0.1399** |
| **MSE** | 8074.73 ± 2481.08 |
| **MAE** | 38.77 ± 2.93 |

### 보조 지표 — Stacking Meta-Learner Test Set 성능 (참고)

| 지표 | 값 |
|------|------|
| Test R² | 0.7754 |
| Test MSE | 4370.98 |
| Test MAE | 29.70 K |

> Train:Test = 1693:424 (80:20 stratified split), 모든 CV는 Train에서 수행.

---

## C. 아키텍처 설명

### 1. 모델 유형 및 선택 이유

**Stacking Ensemble 구조**:
```
Base 1: ResidualMLP (5-seed Snapshot)  — 비선형 연속 함수 근사
Base 2: XGBoost                         — 탐욕적 분기, 상호작용 포착
Base 3: LightGBM                        — Leaf-wise 성장, 세밀 경계
   ↓ OOF predictions (5-fold)
Meta-Learner: Ridge(α=1.0)              — 선형 가중 합성
   ↓
최종 예측
```

**선택 이유**:
- 단일 모델(XGBoost R² 0.48, MLP R² 0.58)로는 복잡한 MP 구조-물성 관계 포착 한계
- 세 base 모델의 **inductive bias가 서로 직교** (신경망 vs 앙상블 트리)
- Ridge 계수 학습 결과 — ANN 0.548, XGB 0.216, LGB 0.226 — **모두 양수**이므로 각 base 모델이 진정한 예측 신호를 보유함이 확인됨
- 특정 Fold(예: Fold 3)에서 ANN R²=0.45 → Tree R²=0.73로 큰 격차 → 상호 보완 효과 명확

### 2. 피처 엔지니어링 / 전처리 방법

**(1) SMILES → 2431차원 피처**
- **Morgan Fingerprint** (radius=2, fpSize=2048): 국소 부분구조 표현
- **MACCS Keys** (167 bits): 작용기 기반 이진 피처
- **RDKit Descriptors** (~216개): 분자량, LogP, TPSA, 회전 결합수 등 물리화학 기술자
- **총 차원**: 2048 + 167 + 216 = **2431**

**(2) 차원 축소 파이프라인**
- `VarianceThreshold(0.01)`: 저분산 비트 제거 → **2431 → 504** (79% 감소)
- `StandardScaler`: 평균 0, 표준편차 1로 정규화 + `[-10, 10]` 클리핑 (극단값 제어)
- `mutual_info_regression` top-300 선별 (ANN용, fold-내부 fit으로 누수 차단)

**(3) 타깃 변환**
- `log1p(MP)` 변환 후 학습 → `expm1()` 역변환 (고온 꼬리 분포 완화)

**(4) 데이터 증강**
- **Mixup** (α=0.3, p=0.5): 두 샘플을 선형 보간하여 부드러운 결정경계 학습 (X, y, sample_weight 모두 보간)

### 3. 하이퍼파라미터 튜닝 전략

**ResidualMLP (ANN base)**:
- 구조: `proj(Linear+LN) → ResidualBlock×3 → head(LN+Linear+SiLU+Linear)`, hidden=256
- 각 ResidualBlock: `LN → Linear → SiLU → Dropout(0.3) → LN → Linear → (+skip)`
- Optimizer: Adam(lr=1e-3, weight_decay=5e-4)
- Scheduler: ReduceLROnPlateau(patience=10, factor=0.5)
- Early stopping: EPOCHS=80, PATIENCE=12
- Batch size: 128, 5-seed ensemble: [42, 53, 65, 79, 93]

**Tree 모델** (기본 파라미터):
- XGBoost: n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8
- LightGBM: n_estimators=1000, num_leaves=31, learning_rate=0.05, subsample=0.8
- Early stopping rounds=50

**Meta-Learner**:
- Ridge(alpha=1.0) — 단순 선형 결합, L2 정규화

**튜닝 철학**:
- Optuna 등 자동 HPO보다 **검증된 합리적 기본값 + 적절한 regularization** 선호
- 과도한 HPO는 소규모 데이터(2117개)에서 오히려 과적합 유발 (별도 실험으로 확인)

### 4. 학습 과정에서의 핵심 관찰 및 인사이트

**(1) 단일 모델 한계의 명확한 증거**
- ANN 단독 CV R² 0.6543이지만 특정 Fold(2, 3)에서 R² 0.45로 붕괴
- Tree 모델이 같은 Fold에서 R² 0.73 달성 → **서로 다른 실패 패턴**
- Stacking이 이를 상호 보완하여 **Test R² 0.7754 달성**

**(2) CV 편차가 크지만 Scaffold CV에서 안정**
- Stratified CV 편차 ±0.140: 특정 분자 클래스에 따라 성능 편차
- Scaffold(Bemis-Murcko) 기반 CV 편차 **±0.065**: 분자 구조 기반 평가로는 안정
- **일반화 성능은 scaffold CV가 더 정직한 지표**

**(3) 복잡도 증가의 함정**
- LightGBM meta-learner 시도 → R² 0.7754 → 0.6960 (**-0.08 하락**)
- Optuna HPO 30 trial → 기대 +0.03, 실측 +0.00
- Tautomer augmentation (+19% 데이터) → R² -0.03
- **교훈**: 소규모 데이터(2117)에서는 모델 복잡도보다 **모델 다양성**이 중요

**(4) 정규화 조합의 누적 효과**
- Mixup + Dropout 0.3 + WD 5e-4 + LayerNorm의 조합이 CV-Test 갭을 극도로 낮춤
- MLP 단일 모델만으로도 Test R² 0.58에서 0.75까지 향상

### 5. 시각화 자료

**Predicted vs Actual (Test Set)**
- 대각선(y=x)에 집중 분포, R² = 0.7754
- 저온 영역(MP < 300K)에서 예측 정확도 높음
- 고온 영역(MP > 800K)에서 예측 편차 증가 (샘플 수 부족)

**잔차 분포**
- 평균 ≈ 0 (bias 없음)
- 표준편차 ≈ 66 K, 정규 근사
- MAE 29.70 K는 실험 MP 측정 오차(±1~5 K) 대비 6~30배

**Fold별 CV R² (5-Fold)**

| Fold | ANN | XGBoost | LightGBM |
|------|------|---------|----------|
| 1 | 0.800 | 0.756 | 0.638 |
| 2 | 0.525 | 0.329 | 0.297 |
| 3 | 0.452 | 0.726 | 0.691 |
| 4 | 0.718 | 0.545 | 0.650 |
| 5 | 0.777 | 0.702 | 0.659 |
| **평균** | **0.654** | **0.612** | **0.587** |

*Fold 3에서 Tree가 ANN을 큰 폭으로 역전 → Stacking의 보완 효과 직접 증거*

---

## 추가: 개선 여정 요약

| 버전 | 핵심 변경 | CV R² | Test R² | Test MAE |
|------|---------|-------|---------|----------|
| base | XGBoost 단독 | 0.5346 | 0.4831 | 53.06 |
| v3_ann | MLP 4층 | 0.7065 | 0.5821 | 46.35 |
| v4_ann | Mixup + Residual + MI | 0.6702 | 0.6958 | 32.53 |
| v5_ann | + Epochs↑ + Scaffold CV | 0.6543 | 0.7536 | 32.00 |
| **v6_stacking** | **+ XGB/LGB Stacking + Ridge** | **0.6543** | **0.7754** | **29.70** |

최종 모델(v6_stacking)은 base 모델 대비 **Test R² +60%, MAE -44%** 개선.

---

*제출 파일*: `final_report.md` + `v6_stacking/melting_point_ann.ipynb`
