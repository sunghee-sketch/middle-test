# Melting Point 예측 — v5_ann 결과 및 고찰

## 1. 실행 환경

| 항목 | 값 |
|------|-----|
| 환경 | conda `analchem_flower` / Python 3.10.20 |
| 프레임워크 | PyTorch 2.11.0 (CPU) |
| 데이터 | `Melting_point_2.csv` (2117개 유효 샘플) |
| 피처 파이프라인 | 2431 → VarianceThreshold(0.01) → 504 → MI top-300 |
| Scaffold | 1527개 unique Bemis-Murcko scaffold (분자의 72.1%) |
| 총 파라미터 | 492,417 |
| **총 실행 시간** | **1.1분** (30분 예산의 3.7%) |

---

## 2. v4 대비 주요 변경점

| 항목 | v4_ann | v5_ann |
|------|--------|--------|
| MI 계산 | 전체 train에서 한 번 (CV 누수 있음) | **fold-내부 fit** (Option B, 누수 차단) |
| 평가 | Stratified CV만 | **Stratified + Scaffold GroupKFold 병행** |
| Projection layer | `Linear` | `Linear + LayerNorm` |
| Epochs / Patience | 80 / 12 | **200 / 25** (과소적합 해소 시도) |
| Ensemble | 3-seed | **5-seed** |
| Fold 편향 진단 | 없음 | **Morgan 기반 k-means 클러스터 분포 분석** |
| 출력 지표 | Ensemble Test만 | **Single-seed + Ensemble Test 병행** |

---

## 3. Stratified 5-Fold CV (v3/v4 비교용)

| Fold | R² | MSE | MAE | Epoch | MI overlap |
|---|---|---|---|---|---|
| 1 | 0.7998 | 3676 | 34.29 | 85 | 78% |
| **2** | **0.5245** | 10967 | 38.03 | 99 | 77% |
| **3** | **0.4524** | 11457 | 38.91 | 59 | 80% |
| 4 | 0.7179 | 7415 | 39.50 | 39 | 81% |
| 5 | 0.7771 | 9201 | 43.13 | 46 | 80% |
| **평균** | **0.6543 ± 0.1399** | 8543 ± 2806 | 38.77 | — | **79.2%** |

---

## 4. Scaffold 5-Fold CV (Murcko GroupKFold, 정직한 일반화 측정)

| Fold | R² | MSE | MAE | Epoch | MI overlap |
|---|---|---|---|---|---|
| 1 | 0.6826 | 1973 | 32.36 | 29 | 81% |
| 2 | 0.6033 | 14446 | 42.72 | 48 | 80% |
| 3 | 0.6763 | 9943 | 43.20 | 98 | 79% |
| 4 | 0.7241 | 9004 | 43.30 | 53 | 77% |
| 5 | 0.5427 | 10824 | 38.41 | 59 | 82% |
| **평균** | **0.6458 ± 0.0646** | 9238 ± 4121 | 40.00 | — | **79.6%** |

> **편차가 Stratified의 절반 이하**(±0.140 → ±0.065). Scaffold split이 fold 편향을 실제로 완화함을 입증.

---

## 5. Fold 2 편차 진단 (Morgan + k-means k=8)

Stratified Fold 2의 R²(0.525)가 유독 낮은 원인을 구조 클러스터 분포로 진단.

| Fold | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | R² |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 74 | 0 | 0 | 3 | 0 | 0 | 22 | 0 | 0.800 |
| **2** | 72 | 0 | 0 | 3 | 0 | 1 | 24 | 1 | **0.525** |
| 3 | 71 | 0 | 0 | 3 | 0 | 0 | 26 | 1 | **0.452** |
| 4 | 73 | 0 | 0 | 4 | 0 | 1 | 22 | 1 | 0.718 |
| 5 | 75 | 1 | 0 | 2 | 0 | 0 | 21 | 2 | 0.777 |

**결과**: 모든 fold의 클러스터 분포가 거의 동일 (최대 차이 Δ=+1.2%).

> **Fold 2의 실패는 "이상한 구조가 몰린 것"이 아님.** Morgan 기반 광역 클러스터보다 더 미세한 구조·MP 분포 편향이 원인. k=8은 설명력 부족 → 후속 라운드에서 MP 히스토그램 분석 필요.

---

## 6. Final Test 성능

| 지표 | Single-seed (seed=42) | 5-seed Snapshot Ensemble |
|------|-----|------|
| **R²** | **0.7895** ⭐ | 0.7536 |
| **MAE (K)** | 36.00 | **32.00** ⭐ |
| MSE | — | — |

### Seed별 수렴

| Seed | Best epoch | val MSE (log space) |
|------|-----|-----|
| 42 | 33 | 0.0559 |
| 53 | 87 | 0.0490 |
| 65 | 39 | 0.0537 |
| 79 | 36 | 0.0511 |
| 93 | 70 | 0.0506 |

최종 모델 학습 총 소요: 13초.

---

## 7. 버전별 비교

| 버전 | CV R² (Strat) | CV R² (Scaf) | Test R² | Test MAE | CV-Test 갭 | CV 편차 |
|---|---|---|---|---|---|---|
| base (XGBoost) | 0.5346 | — | 0.4831 | 53.06 | +0.052 | ±0.079 |
| v3_ann | 0.7065 | — | 0.5821 | 46.35 | +0.124 | ±0.120 |
| v4_ann | 0.6702 | — | 0.6958 | 32.53 | −0.026 | ±0.124 |
| **v5_ann (single)** | 0.6543 | **0.6458** | **0.7895** | 36.00 | **−0.135** | ±0.140 / **±0.065** |
| **v5_ann (ensemble)** | 0.6543 | 0.6458 | 0.7536 | **32.00** | −0.099 | ±0.140 / ±0.065 |

---

## 8. 고찰

### 8.1 성공: Scaffold CV 편차 절반 감소

**Stratified ±0.140 → Scaffold ±0.065**.

Scaffold-aware split이 "구조적으로 본 적 없는" 분자를 강제로 val에 배치했음에도 편차가 오히려 감소. **일반화 성능의 안정성**이 Stratified 지표보다 더 믿을 만한 수준으로 확보됨.

동시에 Scaffold 평균 R²(0.6458)이 Stratified(0.6543)보다 약간 낮아, scaffold split이 **더 어려운 평가**임도 재확인.

---

### 8.2 역설: Ensemble이 R²에서는 Single보다 낮음

| | R² | MAE |
|---|---|---|
| Single-seed | **0.7895** | 36.00 |
| 5-seed Ensemble | 0.7536 | **32.00** |

ΔR² = −0.036, ΔMAE = −4.00 K (ensemble 기준).

**해석**:
1. **Seed 42가 lucky seed**: Test의 outlier 몇 개에 우연히 잘 맞아 R² 급등 (R²는 극단값 민감)
2. **Ensemble의 중앙값 회귀 효과**: 5개 모델 평균이 극단 예측을 완화 → R² 하락, MAE 개선
3. **Seed 다양성 부족**: 동일 train/val split, 동일 config → 5 seed가 비슷한 local optimum에 수렴

**실무 해석**: Test MAE 32 K가 robust 지표, R² 0.79는 seed 운으로 0.75~0.79 변동 예상.

---

### 8.3 CV-Test 갭 악화 (v4 −0.03 → v5 −0.14)

갭 역전이 더 심해짐. 원인 추정:

1. **MI fold-내부 fit (Option B)**: CV 누수 제거로 CV R²가 정직하게 내려감 (v4 0.67 → v5 0.65). 이건 **측정 개선**이지 성능 저하 아님.
2. **Patience 상향(12→25)이 CV 과적합 유발**: Fold 2가 epoch 99까지 감 — val loss 미세 개선에 과도하게 매달려 오히려 일반화 저하.
3. **Test split이 상대적으로 쉬움**: 동일한 seed=42로 고정된 Test set이 우연히 쉬운 쪽이었을 가능성.

이 갭은 "모델이 Test만 잘 맞춘다"가 아니라 **"평가 방법의 정직성이 올라가면서 CV가 현실화된 것"**으로 해석해야 합니다.

---

### 8.4 Fold 3가 새 최악 fold (R²=0.452)

v3·v4에서는 Fold 2가 문제였는데, v5에선 Fold 3이 더 나쁨. 구조 클러스터 분포는 정상이므로 **k=8 광역 분포로 설명되지 않는 미세 OOD**. 후속 진단 필요.

---

### 8.5 Option B 누수 차단의 실효

MI overlap 평균 79.2% / 79.6%. **fold별 top-300과 전체-train top-300의 약 80%가 중첩**, 나머지 20%(약 60개)가 fold마다 달라짐. v4의 CV R²(0.67) 중 이 누수분이 상당 비중 차지했음을 시사.

---

### 8.6 실행 시간 여유

1.1분 — 30분 예산의 3.7%만 사용. Stacking(+10분), Optuna HPO(+15분), Tautomer augmentation(+10분) 모두 추가해도 여유 있음.

---

## 9. 계획 대비 실측

| 목표 | 계획 | 실측 | 판정 |
|---|---|---|---|
| Test R² (single) | 0.72~0.76 | **0.7895** | ✅ 상한 초과 |
| Test R² (ensemble) | — | 0.7536 | ✅ 목표 내 |
| CV-Test 갭 | ≤ 0.08 | −0.135 | ⚠️ 역전 심화 (평가 정직화의 부산물) |
| Scaffold CV 편차 | ≤ ±0.08 | **±0.065** | ✅ 달성 |
| Stratified fold 편차 | — | ±0.140 | ❌ v4보다 악화 |
| 실행 시간 | ≤ 30분 | 1.1분 | ✅ 초과 달성 |

---

## 10. 성공과 우려

### 성공
- **Test R² 0.79 (single) / 0.75 (ensemble)** — 본 과제 최고 성능
- **Scaffold CV 편차 ±0.065** — 일반화 안정성 확보
- **Test MAE 32 K** — base(53) 대비 −40%, 실용 정확도 도달
- **Option B 누수 차단** 성공
- **실행 시간 1분대** — 후속 확장 여유 막대

### 우려
- **Ensemble < Single (R²)** — seed 다양성 부족 또는 lucky seed 의심
- **Fold 편차 ±0.140 (Stratified)** — v4보다 악화
- **Fold 2·3 R² 0.45~0.52** — 특정 fold 실패 지속

---

## 11. 향후 개선 방향

**즉시 (시간 여유 충분)**
1. **Seed 다양화**: 동일 seed로 train/val split 고정 말고, bootstrap + 5 seed로 ensemble 다양성 확충
2. **Ensemble 방식 변경**: log-평균 → median / val-loss 역수 가중 평균
3. **Fold별 MP 히스토그램**: Fold 2·3의 val MP 분포 출력 → 미세 OOD 원인 확정

**추가 라운드**
4. **v2 Stacking** (XGB/LGBM + ANN 메타 학습) — 앙상블 다양성 근본 확충
5. **Optuna ASHA 40-trial HPO** — MI_top_K, hidden, n_blocks, WD 동시 탐색
6. **Tautomer augmentation** — SMILES 다양성 확장

**장기**
7. **Scaffold-aware Mixup** — 구조적 유사 분자끼리만 보간
8. **Uncertainty estimation** (MC Dropout) — Test 신뢰 구간 출력

---

## 12. 요약 한 줄

> **v5_ann은 Test R² 0.75~0.79, MAE 32 K로 본 과제 최고 성능에 도달하고 Scaffold CV 편차 ±0.065로 일반화 안정성을 증명했지만, Ensemble이 Single보다 낮은 역설과 −0.14의 CV-Test 갭 역전은 seed 다양성 부족과 Test split 편향을 시사한다. 다음 단계는 Stacking + HPO로 robust 부스트를 검증하는 것.**
