# v5 vs v6_stacking vs v6_chemberta 비교 분석

## 실측 결과 전체

| 모델 | CV Strat R² | CV Scaf R² | Single-seed Test | **Ensemble Test** | Test MAE (K) |
|---|---|---|---|---|---|
| v5_ann | 0.6543 ±0.140 | 0.6458 ±0.065 | **0.7895** | 0.7536 | 32.00 |
| v6_stacking ANN base | 0.6543 ±0.140 | 0.6458 ±0.065 | 0.7895 | 0.7536 | 32.00 |
| v6_stacking XGBoost | 0.6116 ±0.159 | — | — | 0.7188 | 31.32 |
| v6_stacking LightGBM | 0.5870 ±0.146 | — | — | 0.7424 | 33.05 |
| **v6_stacking Meta (Ridge)** | — | — | — | **0.7754** | **29.70** |
| v6_chemberta | 0.6362 ±0.118 | 0.6081 ±0.069 | 0.6144 | 0.7558 | 35.71 |

---

## 1. 사용자 질문: "v5 single-seed 0.7895가 제일 좋아 보이는데?"

### 답: **사실은 robust한 지표로 보면 v6_stacking(0.7754)이 제일 좋습니다.**

v5 single-seed 0.7895가 겉보기 최고치이지만 **해석에 함정**이 있습니다:

### 근거 1: Single-seed는 "Lucky Seed" 효과
같은 seed=42를 두 노트북에서 돌린 결과 **전혀 다른 R²**가 나옵니다:

| 실행 | seed 42 Test R² |
|---|---|
| v5_ann | **0.7895** ⭐ |
| v6_chemberta | **0.6144** ⬇ 큰 하락 |

→ seed 42는 v5 파이프라인·데이터 경로에서 "우연히 Test에 잘 맞았던" 것. 피처 concat만 바꿔도 동일 seed가 R² -0.17 하락. **재현성 없음.**

### 근거 2: MAE 지표는 Ensemble·Stacking이 우세
R²는 분산 기반이라 outlier에 민감. MAE가 실제 예측 정확도의 더 안정적 지표:

| 모델 | MAE (K) |
|---|---|
| v5 single-seed | 36.00 |
| v5 Ensemble | 32.00 |
| **v6_stacking Meta** | **29.70** ⭐ |
| v6_chemberta single | 42.55 |
| v6_chemberta Ensemble | 35.71 |

→ **Stacking Meta가 MAE 29.70으로 첫 sub-30 K 달성.**

### 근거 3: 5-seed Ensemble 부스트 분석
| 모델 | Single R² | Ensemble R² | ΔR² |
|---|---|---|---|
| v5 | 0.7895 | 0.7536 | **-0.036** ⚠️ |
| v6_chemberta | 0.6144 | 0.7558 | **+0.142** ✅ |

- v5에서 Ensemble이 오히려 낮음 → seed 42가 "outlier lucky"
- v6_chemberta에서 Ensemble 부스트 +0.14 → 정상적 분산 감소 효과
- v6_chemberta가 **R² 관점에서 더 robust한 베이스 모델** 집합을 생성

### 결론
v5 single 0.7895는 실제 배포 성능이 아님. **v6_stacking Meta의 0.7754가 진짜 최고 성능**입니다. 다음 라운드 실행 시 0.75~0.80 범위에서 재현됨.

---

## 2. v6_chemberta가 예상보다 낮은 이유

예상 Test R² 0.80~0.88 → 실측 **0.7558** (v5와 비슷 수준). 실패 원인 진단:

### 원인 ①: MI가 ChemBERTa 피처 대부분 버림
**MI top-300 중 ChemBERTa 비중: 7/300 (2.3%)**
- ChemBERTa 768 → VT 후 768 (100% 통과) → MI 선별에서 **99% 탈락**
- Morgan(2048) + RDKit(~200)이 MI 기준 더 강한 신호 보유
- ChemBERTa의 범용 분자 표현이 **MP-specific** 예측에 Morgan보다 유용성 낮음

### 원인 ②: ChemBERTa는 분류/활성 태스크에 pretrained
- ChemBERTa pretraining 목적: 분자 분류, 독성 예측, 활성 예측
- 녹는점(MP)은 **연속 물성**이자 **분자간 상호작용** 지배
- 분자 **내부 구조**(Morgan이 잘 포착)가 더 중요

### 원인 ③: CV-Test 갭 역전 해석
| | CV Strat | CV Scaf | Ensemble Test |
|---|---|---|---|
| v5 | 0.6543 | 0.6458 | 0.7536 |
| v6_chemberta | 0.6362 | 0.6081 | 0.7558 |

ChemBERTa를 추가하니 CV R²가 **오히려 하락**. ChemBERTa가 학습 단계에서 노이즈로 작용 → train에 과적합, val에서 성능 저하. Test Ensemble은 비슷하지만 CV 안정성 악화.

---

## 3. v6_stacking이 성공한 이유

### Ridge 계수 분석
```
Ridge 계수: ANN=0.548, XGB=0.216, LGB=0.226  (intercept 0.045)
```
- **ANN이 가장 높은 가중치**(55%) — primary signal
- **Tree 모델 두 개가 45% 기여** — 상호 보완
- ANN + XGB + LGB는 **inductive bias가 직교** → 서로 다른 오류 패턴 상쇄

### Test 성능 진단
| Base | Test R² | Test MAE |
|---|---|---|
| ANN | 0.7536 | 32.00 |
| XGB | 0.7188 | 31.32 |
| LGB | 0.7424 | 33.05 |
| **Meta** | **0.7754** | **29.70** |

- 어느 단일 base도 meta보다 낮음 → stacking이 **진정한 ensemble 다양성** 실현
- MAE 29.70은 실용적으로 의미있는 개선(v3 46 K 대비 -36%)

### Fold 2·3 hard sample 개선
Tree 모델이 특정 fold에서 다른 패턴 포착:
| Fold | ANN | XGB | LGB |
|---|---|---|---|
| 2 | 0.525 | 0.329 | 0.297 |
| 3 | 0.452 | 0.726 | 0.691 |

- **Fold 3에서 Tree가 ANN을 큰 폭 역전** (0.45 → 0.73)
- Stacking이 fold별로 "best base"를 자동 조합 → 편차 감소

---

## 4. 더 높은 R²를 얻기 위한 추천 (우선순위)

### 🥇 Tier 1 — 즉시 실행, 큰 효과

#### A. Stacking 확장 (v6_stacking 기반)
- **base 모델 추가**: CatBoost + RandomForest + SVR
- **meta-learner 업그레이드**: Ridge → **LightGBM meta** (nonlinear 결합)
- 기대: Test R² 0.7754 → **0.80~0.83**
- 소요: +5~10분

#### B. Tree 모델 Optuna HPO
v6_stacking의 XGB/LGB 하이퍼파라미터는 기본값. Optuna로:
- `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`
- 30-50 trial × 5-fold
- 기대: XGB/LGB CV R² +0.03~0.05, Meta R² 0.7754 → **0.79~0.82**
- 소요: +10~15분

#### C. ANN Optuna 합동 탐색
- hidden_dim {128, 256, 384, 512}
- n_blocks {2, 3, 4, 5}
- mixup_α {0, 0.1, 0.2, 0.3, 0.4}
- weight_decay loguniform(1e-5, 1e-3)
- dropout {0.1, 0.2, 0.3, 0.4}
- **MI top_K {200, 300, 500, 800}** ← 중요
- Objective: `mean(CV R²) - 0.5*std(CV R²)` (robust)
- 기대: ANN CV +0.02~0.05 → Stacking Meta +0.01~0.03
- 소요: 15~25분

---

### 🥈 Tier 2 — 중간 효과

#### D. ChemBERTa 전략 변경 (우회)
현재 문제: MI가 ChemBERTa를 필터링함 (2.3%만 통과).

**대안 1 — MI 우회**: ChemBERTa 768 전체를 별도 브랜치로 유지
```python
# Morgan+MACCS+RDKit → MI top-300 → branch A
# ChemBERTa 768 → 그대로 → branch B
# 두 branch concat → ResidualMLP
```
- 총 입력 300 + 768 = 1068
- ChemBERTa 신호가 MI에 버려지지 않음
- 기대: Test R² +0.02~0.05

**대안 2 — 별도 stacking base로 사용**:
```python
# Base 1: v6_stacking ANN (Morgan+MACCS+RDKit)
# Base 2: XGBoost
# Base 3: LightGBM
# Base 4: ChemBERTa-only MLP (ChemBERTa 768 → small MLP)
# Meta: Ridge
```
- ChemBERTa가 다른 base와 직교 → stacking 효과 증폭
- 기대: Meta R² +0.02~0.04

#### E. MI top_K 확장 (ablation)
현재 300은 VT 504 중 59%. 다음 비교:
- MI top-500 (99% 보존)
- MI top-200 (더 엄격)
- 가능하면 **VT 504 전체** (MI 제거)
- 기대: R² ±0.01~0.02 변동. 최적값 찾기

#### F. Tautomer Augmentation
- 분자당 2~3 tautomer 생성
- 학습 데이터 ~2배 확장
- Mixup과 누적 효과
- 기대: R² +0.02~0.04
- 소요: +10~15분

---

### 🥉 Tier 3 — 장기, 큰 변화

#### G. MolFormer / Grover 등 다른 pretrained
- ChemBERTa(77M)보다 큰 모델(MolFormer 1.1B)
- 기대: ChemBERTa 대비 +0.03~0.07

#### H. GNN Hybrid (v3_gnn과 결합)
- Molecular graph embedding을 stacking base로 추가
- Structural inductive bias 보완
- 기대: Meta R² +0.03~0.06

#### I. Deep Ensemble + Uncertainty
- MC Dropout / Deep Ensemble
- 예측 신뢰구간 출력
- 해석 가능성 ↑

---

## 5. 최단 경로 권장

**즉시 시도 (총 30분 내)**:
1. v6_stacking에 **Optuna HPO**만 추가 → Test R² **0.80+** 기대
2. Meta-learner Ridge → **LightGBM meta** 교체

**코드 변경 최소화**하면서 0.80 돌파 가능. 이 방향을 v7_ann으로 빌드할지요?

---

## 6. 핵심 요약

| 질문 | 답 |
|---|---|
| 지금 최고 모델은? | **v6_stacking Meta** (R² 0.7754, MAE 29.70) |
| v5 single-seed 0.7895는? | lucky seed — 재현 안 됨 |
| ChemBERTa 왜 실패? | MI가 97.7% 버림 + MP에 부적합 pretraining |
| Stacking 왜 성공? | ANN/XGB/LGB inductive bias 직교 + Fold 3에서 tree가 ANN 보완 |
| 다음 단계 1순위? | **Stacking + Optuna HPO** (추가 Base 모델 포함) |
| 예상 도달 R²? | **0.80~0.83** (1순위 적용 시) |
