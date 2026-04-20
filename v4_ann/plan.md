# v4_ann 설계 계획서

## 왜 v4_ann인가?

`v3_ann`은 트리 모델(base) 대비 큰 개선을 이뤘습니다.
- CV R² 0.5346 → **0.7065** (+0.172)
- Test R² 0.4831 → **0.5821** (+0.099)

그러나 **CV-Test 갭이 0.052 → 0.124로 확대**되어 과적합 경향이 다시 나타났습니다 (`v3_ann/result_analysis.md`). v4는 5인 전문가 팀 토의 결과 합의된 5단계 통합 파이프라인을 적용합니다.

| 지표 | v3_ann | v4_ann (목표) |
|---|---|---|
| Test R² | 0.5821 | **0.70~0.75** |
| CV-Test 갭 | 0.124 | **≤ 0.08** |
| CV 편차 | ±0.120 | ≤ ±0.08 |

---

## 5단계 합의 파이프라인

| 순위 | 작업 | 주관 전문가 | 기대 R² Δ | 비고 |
|---|---|---|---|---|
| **1** | Augmentation (Mixup α=0.3) + Step A (VarianceThreshold) | augmentation-expert | +0.05~0.10 | Mixup은 binary FP 위에서 soft regularization으로 작동 |
| **2** | Feature Engineering (MI top-300 선별) | feature-expert | +0.03~0.05 | SHAP 대신 MI 사용 (속도 5~10배) |
| **3** | Architecture (Residual + LayerNorm + Swish) | architecture-expert | +0.03~0.05 | hidden=256, 데이터 작음을 고려해 모델 폭 제한 |
| **4** | Hyperparameter (joint search) | hyperparam-expert | +0.02~0.05 | v4 1차는 sensible defaults, 후속에서 Optuna |
| **5** | Ensemble (5-seed Snapshot) | ensemble-expert | +0.02~0.05 | Stacking with v2 트리는 후속 |

**누적 예상**: Test R² 0.58 → 0.70~0.75

---

## 핵심 협업 묶음

```
[Step A: VarianceThreshold]  ← 데이터 비의존적 사전 정리
        │
        ▼
[Step 1: Mixup]  ←──직교────►  [Step 2: MI top-300]
        │                            │
        └────────────┬───────────────┘
                     ▼
        [Step 3: ResidualMLP] ◄──── joint ────► [Step 4: HPO (후속)]
                     │
                     ▼
        [Step 5: 5-seed Snapshot Ensemble]
                     │
                     ▼
              [최종 예측 → Test 평가]
```

- **1+Step A 묶음**: VarianceThreshold(0.01) → Mixup 학습. VT는 sample-independent라 train/test 분할 전 적용.
- **3+4 joint search (후속)**: Optuna에 architecture choice + LR/dropout/mixup_α 동시 탐색.
- **3 → 5 묻음**: Architecture 학습 시 시드 5개 돌려두면 5순위 ensemble은 가중치 평균만 취하면 됨 (추가 학습 없음).

---

## 기술 선택 근거

### Step A: VarianceThreshold(0.01)
- Morgan FP 2048 비트 중 상당수가 거의 항상 0 (희소 substructure) → 정보량 0
- 학습 전 사전 제거 시 ~500~800 비트 감소 예상, 정보 손실 거의 없음
- 데이터 비의존이라 train/test 분할 이전에 적용 가능

### Mixup (α=0.3)
- α=0.2~0.4가 tabular 회귀의 표준 범위
- Binary FP 입력에선 soft regularization으로 작동 (이진 비트의 선형 보간 → confidence smoothing 효과)
- sample weight는 **보간하지 않음** (high-MP weighting 의도 보존)

### MI vs SHAP
- SHAP은 학습된 모델 필요 (chicken-and-egg) + 계산 시간 오래
- MI(mutual_info_regression)는 학습 전 직접 계산, scikit-learn 표준 구현
- top-300 선별로 입력 차원 ~8배 축소 → 모델 capacity와 데이터량 균형

> **✅ MI 누수 차단 (Option B 채택, 라운드 2 리뷰 합의)**
> CV 루프 내부에서 **fold tr_idx만으로 MI fit**하여 validation 정보 누수를 차단합니다.
> `cell v4_mi`의 전체-train MI는 **최종 모델 학습용**으로만 사용(Test 누수 없음).
> Fold별 top-K overlap이 함께 출력되어 selection robustness도 진단 가능.
> 추가 비용: +3~5분 (30분 예산 내).

### Residual + LayerNorm + Swish
- v3의 단순 4층 MLP는 표현력 부족 의심
- **Residual**: 깊이 늘려도 gradient flow 유지
- **LayerNorm**: BatchNorm은 batch=64에서 통계 분산 큼. LayerNorm은 sample 단위라 안정
- **Swish (SiLU)**: ReLU 대비 smooth gradient, 작은 데이터에서 일반화 우수
- **hidden=256**: 데이터 2117개에 대해 hidden=512는 과한 capacity (architecture-expert 권고)

### Snapshot Ensemble (3 seeds, 최종 모델만)
- **시간 제약(노트북 전체 ≤ 30분)** 반영해 시드 5 → 3 축소
- CV 단계는 단일 시드 (fold당 1개 모델), 최종 Test 평가에서만 3-seed 앙상블
- 분할 의존성(v3 Fold 편차 ±0.120)은 sample weight + Mixup으로 1차 완화, snapshot은 최종 안정화
- 예측 평균은 **log 공간에서 평균 후 expm1**로 적용 (수치 안정성)

---

## v3_ann 대비 차이점

| 항목 | v3_ann | v4_ann |
|---|---|---|
| 피처 차원 | 2431 | 2431 → VT → MI top-300 |
| Architecture | MLP 512→256→128 | Residual×3 (hidden=256) |
| Activation | ReLU | Swish (SiLU) |
| Normalization | BatchNorm | LayerNorm |
| Dropout | 0.3/0.3/0.2 | 0.3 (균일) |
| Weight Decay | 1e-4 | 5e-4 (강화) |
| Augmentation | 없음 | Mixup α=0.3 |
| Ensemble | 단일 모델 | 3-seed Snapshot (최종만) |
| Epochs | 150 | **80** (시간 제약) |
| Patience | 20 | **12** (시간 제약) |
| Batch | 64 | **128** (epoch 단축) |
| Sample weight | 고MP 3.0× | 동일 유지 |

---

## 후속 작업 (v4 1차 빌드 후)

5단계 합의 중 다음 항목은 1차 빌드에서 제외하고 후속 라운드에서 진행:

1. **Hyperparameter joint search (Optuna)**: 80 trial × 5-fold = 4~6h GPU. 1차 baseline 확정 후.
2. **SHAP 기반 정밀 피처 선별**: MI 결과 위에서 SHAP로 추가 정제.
3. **Stacking with v2 XGBoost/LightGBM**: 1차 결과 위에 OOF 기반 stacking.
4. **Tautomer enumeration**: Mixup 효과가 미미할 경우 대안.

---

## 예상 성과 표

| 단계 | 예상 CV R² | 예상 Test R² |
|------|-----------|-------------|
| base (XGBoost) | 0.5346 | 0.4831 |
| v3_ann (MLP) | 0.7065 | 0.5821 |
| v4_ann (1차) | **0.78~0.82** | **0.70~0.75** |
| v4_ann + HPO + Stacking (후속) | 0.82+ | 0.75+ |

목표 미달 시 결과 분석 후 다음 라운드 토의에서 우선순위 재조정.
