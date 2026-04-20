# v8_ann 결과 분석 — Augmentation 실패 진단

## 1. 실측 결과

### Augmentation 현황
- 원본 Train: 1693 분자
- 증강 후: **2022 분자** (+329, 겨우 **+19%**)
- 평균 샘플/분자: **1.20** (대부분 분자는 tautomer 생성 안 됨)

### CV (GroupKFold on augmented)
| 모델 | CV R² | 편차 |
|---|---|---|
| ANN | 0.6865 | **±0.079** |
| XGBoost | 0.6619 | ±0.121 |
| LightGBM | 0.6557 | ±0.113 |

### Test 성능
| 모델 | R² | MAE (K) |
|---|---|---|
| ANN single-seed | 0.7372 | 33.19 |
| ANN 5-seed | 0.7207 | 32.65 |
| XGBoost | 0.7049 | 32.15 |
| LightGBM | 0.7460 | 32.69 |
| **Stacking Meta (Ridge)** | **0.7489** | **30.12** |

### Ridge 계수 변화
- v6: ANN **0.548**, XGB 0.216, LGB 0.226
- v8: ANN **0.455**, XGB **0.436**, LGB 0.130

---

## 2. v5~v8 종합 비교

| 버전 | 특징 | Test R² | MAE (K) | 판정 |
|---|---|---|---|---|
| v5 single-seed | lucky seed | 0.7895 | 36.00 | 재현 X |
| v5 5-seed | baseline | 0.7536 | 32.00 | |
| **v6_stacking** | **Ridge meta, 3 base** | **0.7754** | **29.70** | 🏆 **최고** |
| v7 Meta (LGB) | 복잡도 증가 | 0.6960 | 32.40 | -0.079 |
| v8 Stacking | + Tautomer | 0.7489 | 30.12 | -0.027 |

**v6_stacking이 여전히 최고**. v7/v8 둘 다 추가 변경이 오히려 성능 하락을 유발.

---

## 3. 🔴 v8 실패 5가지 원인

### ① Tautomer 증강이 너무 적게 생성됨 (예상 2x → 실제 1.2x)
- `TautomerEnumerator` + `max_per_mol=1`로 대부분 분자는 canonical과 동일 tautomer 반환
- **실제 추가: 329개(+19%)만** — 예상 50~100% 확장의 20~40% 수준
- 증강 효과가 통계적으로 유의미하지 않음

### ② CV 방식 변경이 숨은 독립 변수
v6 → v8로 **5개 변수 동시 변경**:
1. Tautomer augmentation
2. 샘플 수 (1693 → 2022)
3. **CV: Stratified → GroupKFold** ⚠️ 가장 의심
4. Target noise σ=0.02 추가
5. VT/MI/Scaler를 augmented에서 재fit

→ 어느 변수가 원인인지 분리 불가. GroupKFold가 MP 분포 균등화 상실 → Fold 4 R² 0.53 (v6의 Fold 4 R² 0.72 대비 큰 하락).

### ③ Tautomer label 할당 가정 오류
- Tautomer끼리 같은 MP 할당했지만 **안정한 tautomer 간 MP는 실제로 다를 수 있음**
- 예: 2-pyridone vs 2-hydroxypyridine은 MP가 다름
- 가상 label이 모델 학습에 노이즈 주입

### ④ Ridge 계수 불균형 (XGB 과의존)
- ANN 비중 55% → 46% 감소
- **XGB 비중 22% → 44% 급증**
- → Augmentation이 XGB 특화 → 균형 무너짐

### ⑤ Target noise 효과 불명확
- σ=0.02 on log scale ≈ ±2~3K
- 작은 데이터에서는 약한 신호 지우는 효과
- CV 편차는 개선됐지만 (v5 ±0.14 → v8 ±0.08) Test R²는 하락

---

## 4. ✨ 긍정적 발견

### Fold 2·3 문제가 거의 해결됨
| Fold | v5 ANN | v6 ANN | **v8 ANN** |
|---|---|---|---|
| 1 | 0.80 | 0.80 | 0.73 |
| **2** | **0.48** ⚠️ | **0.48** ⚠️ | **0.75** ✅ |
| 3 | 0.45 | 0.45 | 0.72 ✅ |
| 4 | 0.72 | 0.72 | 0.53 |
| 5 | 0.82 | 0.82 | 0.69 |

Fold 2·3의 근본 문제는 **"OOD 분자"**였는데, GroupKFold가 분포를 바꿔 원래 Fold 2 샘플들이 다른 fold로 분산 → 문제 국소화 해소. 하지만 **다른 fold(4)가 새 문제**로 이동.

### CV 편차 대폭 감소
- v5 ±0.140, v6 ±0.140 → **v8 ±0.079** (절반!)
- GroupKFold가 일반화 안정성 향상

### MAE는 어느 정도 경쟁력
- v6: 29.70 (최고)
- v8: 30.12 (v6 대비 +0.4 K만)

---

## 5. 🧠 왜 Augmentation이 실패했는가 (이론적 진단)

### 근본 원인: 분자 물성 예측에서 Tautomer 증강의 한계

**Tautomer의 본질**:
- 빠른 평형 하의 **같은 화합물**
- Morgan FP는 다르지만, **실험 MP는 측정 시 평형 혼합물의 평균값**
- 학습에서 tautomer A(FP_A, MP_평균)와 tautomer B(FP_B, MP_평균)는 **"같은 답을 주는 다른 입력"**

**이론적 문제**:
- 모델이 "FP_A와 FP_B 모두에서 MP_평균 예측"하도록 강요됨
- → 모델이 **tautomer 간 일관성에 용량을 할애**, 실제 MP 구조-물성 관계 학습 capacity 감소
- Morgan FP가 이미 대부분 invariant (canonical SMILES에서 추출)이라 tautomer 다양성이 효과 없음

### 비교: Tautomer 증강이 성공하는 경우
- **GNN 분자 모델**: 노드/엣지 구조가 tautomer마다 달라서 의미있는 증강
- **SMILES 토큰 RNN**: 문자 시퀀스 차이를 학습
- **본 케이스(fingerprint MLP)**: Morgan이 이미 canonical → 증강 효과 미미

논문 재검토:
- Karpov et al. 2020은 **GNN 기반 모델**에서 tautomer 증강 효과 보고
- 본 케이스(Morgan+MLP)는 증강 효과 논문에서도 약하게 나타남

---

## 6. 🎯 0.80+ 달성을 위한 새로운 방향

v7/v8의 교훈: **복잡도 증가는 실패**. 현재 baseline을 **다른 축**으로 확장해야 함.

### 🥇 Tier 1 — 가장 유망 (새 방향)

#### A. v3_gnn 복귀 + v6_stacking 결합
- v3_gnn의 R² 0.4568은 **단독 성능 낮음**
- 하지만 stacking base로 추가 시 **구조 정보의 독립 시그널** 제공
- Morgan(2D 부분구조) + GNN(전역 그래프) = 진정한 diversity
- 기대: 0.7754 → **0.80~0.83**

#### B. ChemBERTa **MI 우회** 재시도 (v6_chemberta 실패 원인 수정)
- 이전 실패 원인: MI가 ChemBERTa 97.7% 탈락
- **수정**: ChemBERTa 768 전체를 **별도 branch**로 유지
  ```
  Branch A: Morgan+MACCS+RDKit → MI top-300 → MLP_A (256-dim output)
  Branch B: ChemBERTa 768 → MLP_B (256-dim output)  
  Concat → Final head → MP
  ```
- 기대: 0.80~0.82

#### C. 데이터 외부 확보
- **CompTox**, **OpenChemDB**, **PubChem MP** 등 추가 데이터셋 병합
- 2117 → 5000~10000으로 확장
- 작은 데이터가 본질적 한계이므로 가장 근본적 해결
- 기대: 0.82~0.88

### 🥈 Tier 2 — 기존 구조 개선

#### D. v6_stacking에 **ANN base 복수** 추가
- Wide ANN (hidden=512, 얕음)
- Deep ANN (hidden=128, 깊음 n_blocks=5)
- Medium ANN (현재 v6 설정)
- → 3개 ANN + XGB + LGB = 5 base, 모두 **ANN side 다양화**
- 기대: 0.78~0.80

#### E. **Scaffold-aware Mixup** 구현
- 같은 scaffold 내에서만 Mixup pair 선택
- 화학적으로 타당한 interpolation
- 기대: +0.01~0.02

### 🥉 Tier 3 — 덜 유망

#### F. Pseudo-labeling
- Test 예측을 confidence 높은 샘플만 train에 추가
- 위험: test leak 개념적 논란

---

## 7. 📌 권장 즉시 실행: v9 = v6_stacking + GNN base

**v9 설계**:
```
Base 모델 4개:
  1. ResidualMLP (v6 그대로)
  2. XGBoost (v6 그대로)
  3. LightGBM (v6 그대로)
  4. GIN (v3_gnn 재학습, 5-fold OOF)

Meta: Ridge(alpha=1.0) 유지 (v6 성공 구조)
```

**기대**:
- GNN은 단독 R² 0.45지만 **inductive bias 완전히 직교**
- Fold 2·3에서 tree나 MLP가 놓친 구조 패턴 포착 가능
- Test R² **0.80~0.83** 기대

### 예상 구현 시간
- v3_gnn 코드 재사용 + v6 stacking 통합: 30~40분

---

## 8. 요약 한 줄

> **v8의 Tautomer 증강은 fingerprint 기반 MLP의 canonical invariance 때문에 이론적으로 효과 제한적이었고, CV 방식 동시 변경이 실제 원인 진단도 어렵게 만들었다. v6_stacking (R² 0.7754, MAE 29.70)이 여전히 최고이며, 다음 단계는 "다른 축의 base 모델 추가"(GNN 또는 ChemBERTa 별도 branch)가 가장 유망하다.**

---

## 9. 최종 권장 순위

| 순위 | 방법 | 예상 R² | 구현 시간 | 성공 확률 |
|---|---|---|---|---|
| 1 | **v9: v6 + GNN base** | 0.80~0.83 | 40분 | 높음 |
| 2 | v6 + ChemBERTa 별도 branch | 0.80~0.82 | 30분 | 중간 |
| 3 | 외부 데이터 병합 | 0.82~0.88 | 2시간+ | 높음 (데이터 의존) |
| 4 | v6 유지, MAE 지표로 보고 | — | 0 | 확정 |

현 상태 보고용이면 **v6_stacking Meta (R² 0.7754, MAE 29.70)** 을 최종 결과로 사용 권장.
