# v8_ann 설계 계획서 — Tautomer Augmentation + Stacking

## 개요

v6_stacking의 Ridge meta 구조(R² 0.7754)를 유지하면서 **Tautomer augmentation + Target noise**를 추가해 R² 0.80 돌파를 시도합니다.

## v6 대비 차별점 (3가지)

### 1. Tautomer Enumeration (핵심)
- RDKit `rdMolStandardize.TautomerEnumerator`
- 분자당 평균 1.5~2 tautomer 생성
- **Train only 증강** (Test는 원본 유지 → data leakage 방지)
- 학습 데이터 1693 → ~3000~4000

### 2. Group-aware CV
- 같은 원본 분자의 tautomer는 **같은 fold**에 들어감 (`GroupKFold(groups=original_idx)`)
- Train augmented data가 val에 들어가는 누수 방지

### 3. Target Noise (label smoothing)
- 학습 중 `y_log += Gaussian(0, σ)` 추가 (σ=0.02 on log scale ≈ ±2~3K on original)
- 실험 오차 모방으로 과적합 완화

## v5/v6와 동일한 부분
- 데이터 (2117), Train/Test split (RS=42), Scaffold CV
- 피처: Morgan + MACCS + RDKit → VT(0.01) → MI top-300
- 모델: ResidualMLP(256×3, proj LN + head LN) + Mixup α=0.3 p=0.5
- 학습: Adam lr=1e-3 WD=5e-4, EPOCHS=200 PATIENCE=25
- 5-seed Snapshot [42, 53, 65, 79, 93]
- Base 3개: ANN + XGBoost + LightGBM (v6 동일, 기본 하이퍼파라미터)
- Meta: Ridge(alpha=1.0) (v6 동일)

## 예상 성과

| 구성 요소 | ΔR² | 누적 |
|---|---|---|
| v6_stacking (baseline) | — | 0.7754 |
| + Tautomer 2x | +0.02~0.04 | 0.79~0.81 |
| + Target noise | +0.01~0.02 | 0.80~0.82 |
| + (Mixup α 0.3→0.4 강화) | +0.00~0.01 | 0.80~0.83 |

**목표 Test R²: 0.80~0.82, MAE 25~28 K**

## 예상 실행 시간

| 단계 | 소요 |
|---|---|
| 원본 피처 + 증강 tautomer 피처 | ~15~20초 |
| 나머지 v6 파이프라인 (CV + XGB + LGB + Final + Meta) | ~1.5분 |
| 증강으로 데이터 2배 → CV 시간 약 2배 | +1분 |
| **총** | **~3~4분** |

30분 예산 내 여유.

## 위험 요소

- **Tautomer MP 라벨 가정**: tautomer끼리 같은 실험 MP 사용. 실제로 안정한 tautomer 간 MP 차이 있을 수 있으나 빠른 평형 가정 하에 근사.
- **증강 데이터로 Ridge meta 과적합**: GroupKFold로 누수 차단하지만 meta가 augmented pattern 학습 가능. alpha 조정으로 대응.
- **일부 분자는 tautomer 생성 실패**: 원본만 유지, 로그에 보고.
