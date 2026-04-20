# Melting Point 예측 모델 — 아키텍처 설명 및 결과 분석

## C. 아키텍처 설명

### C-1. 모델 유형 및 선택 이유

**모델: XGBoost Regressor** (Gradient Boosting 기반 앙상블 트리 모델)

**선택 이유:**
- Morgan Fingerprint는 2048차원의 **희소(sparse) 이진 벡터**로, 트리 기반 모델이 희소 피처를 효과적으로 처리함
- 녹는점과 분자 구조 간 관계는 **비선형적**이며, XGBoost는 비선형 관계 학습에 강함
- 소규모 데이터셋(~2189개)에서 딥러닝보다 **과적합 위험이 낮고 학습 속도가 빠름**
- L1/L2 정규화 내장으로 고차원 피처의 과적합 억제 가능
- QSPR(정량적 구조-물성 관계) 분야에서 검증된 방법론

---

### C-2. 피처 엔지니어링 / 전처리 방법

**피처 구성:**
| 피처 | 차원 | 설명 |
|------|------|------|
| Morgan Fingerprint | 2048-bit | 반경 2, 분자 부분구조 인코딩 |
| RDKit 분자 기술자 | 10종 | MolWt, LogP, HBD, HBA, TPSA, RotBonds, Rings, ArRings, CSP3, HeavyAtoms |
| **합계** | **2058** | |

**전처리 방법:**
- SMILES → RDKit `MolFromSmiles()`로 분자 객체 변환, 파싱 실패 샘플 제외 (3개)
- Morgan Fingerprint: `GetMorganGenerator(radius=2, fpSize=2048)`으로 2048-bit 이진 벡터 생성
- 수치형 기술자 10종을 float32로 변환 후 Morgan FP와 concatenate
- 별도 스케일링 미적용 (트리 모델은 피처 스케일에 불변)

---

### C-3. 하이퍼파라미터 튜닝 전략

**사용 값:**
```
n_estimators    = 1000
learning_rate   = 0.02
max_depth       = 5
subsample       = 0.7
colsample_bytree= 0.5
min_child_weight= 5
reg_alpha       = 1.0   (L1)
reg_lambda      = 5.0   (L2)
```

**결정 근거:**
- `learning_rate=0.02` (낮게 설정): 느리게 학습해 일반화 성능 향상, n_estimators=1000으로 보완
- `max_depth=5`: 깊은 트리로 인한 과적합 방지, 분자 데이터 복잡도 고려
- `subsample=0.7`, `colsample_bytree=0.5`: 행/열 샘플링으로 다양성 확보 및 과적합 억제
- `reg_alpha=1.0`, `reg_lambda=5.0`: 고차원(2058) 피처에서 불필요한 피처 가중치 억제
- 별도 자동 탐색(GridSearch 등) 없이 QSPR 문헌 기반 경험적 설정

---

### C-4. 학습 과정 핵심 관찰 및 인사이트

**1. CV Fold 간 성능 편차**
- R² 범위: 0.4997(Fold 3) ~ 0.6116(Fold 1), 표준편차 ±0.039
- Fold 3에서 MSE가 15541로 급등 → 해당 fold에 고MP 극단값이 집중됐을 가능성

**2. CV vs Test 과적합 확인**
- CV R² 0.5655 → Test R² 0.3570 (격차 0.21)
- Train 데이터에 과적합되어 새로운 데이터 일반화 실패

**3. 고MP 구간 예측 실패**
- Predicted vs Actual 그래프에서 MP > 750K 구간의 산포가 급격히 증가
- 데이터의 75%가 297K 이하에 집중되어 있어 고온 구간 학습 데이터 부족

**4. 잔차 분포 fat tail**
- 잔차 범위 -1500 ~ +1000K로 극단 오차 빈발
- 음수 MP(-348K) 등 물리적으로 비정상적인 데이터가 학습에 악영향

**5. SMILES 데이터 품질 문제**
- 원본 데이터(`Melting_point.csv`)에 `?`로 끝나는 잘못된 SMILES 3개 존재
- `Melting_point_2.csv`로 교체하여 전체 2189개 샘플 활용

---

## 2. 실제 결과

### 데이터셋 현황
- 전체 샘플: 2189개 (SMILES 파싱 실패 3개 제외 → 유효 2186개)
- Train: 1748개 / Test: 438개
- MP 범위: -348 ~ 1870 K, 평균 267.7 K, 표준편차 155.8 K

### 성능 지표

| 구분 | R² | MSE | MAE |
|------|-----|-----|-----|
| 5-Fold CV 평균 | **0.5655 ± 0.039** | 10442 ± 3446 | 42.94 ± 5.08 K |
| Test Set | **0.3570** | 16586 | 51.01 K |

### Fold별 CV 결과
| Fold | R² | MSE | MAE |
|------|-----|-----|-----|
| 1 | 0.6116 | 10759 | 41.87 K |
| 2 | 0.5573 | 12577 | 51.16 K |
| 3 | 0.4997 | 15541 | 45.57 K |
| 4 | 0.5629 | 7014 | 36.36 K |
| 5 | 0.5961 | 6321 | 39.75 K |

---

## 3. 결과 분석 — 왜 이런 성능이 나왔는가

### 핵심 문제: 심각한 과적합 (Overfitting)

> CV R² **0.566** → Test R² **0.357** (차이: **0.209**)

CV와 Test 간 R² 격차가 0.21로 매우 큽니다. 정규화(L1=1.0, L2=5.0)를 적용했음에도 과적합이 발생한 원인:

1. **데이터 분포 불균형**  
   MP 분포가 150~400K에 집중되어 있지만 최대 1870K 극단값이 존재. 모델이 다수 구간에 과적합되고 극단값 예측에 실패.

2. **Predicted vs Actual 그래프 해석**  
   저MP(0~500K) 구간은 비교적 직선에 가깝지만, 고MP(>750K) 구간에서 산포가 크게 벌어짐. 즉, 고온 화합물 예측 능력이 부족.

3. **잔차 분포 문제**  
   잔차 범위가 -1500 ~ +1000K로 매우 넓음. 중심부는 날카롭지만 꼬리가 두꺼워(fat tail) 극단 오차가 빈번.

4. **피처의 근본적 한계**  
   녹는점은 **고체 결정 패킹(crystal packing)** 에 의존하는 특성으로, 2D 분자 구조 피처(Morgan FP)만으로는 포착하기 어려움. 3D 구조 정보(결합 각도, 입체화학)가 없음.

5. **기술자 10종은 부족**  
   현재 MolWt, LogP 등 10종만 사용 — 분자 대칭성, 수소결합 패턴 등 녹는점에 중요한 특성이 누락됨.

---

## 4. 개선 방안

### 4-1. 즉시 적용 가능 (효과 높음)

**RDKit 기술자 확장 (10종 → 200종)**
```python
from rdkit.Chem import Descriptors
desc_list = Descriptors.descList  # ~200종 자동 계산
```
예상 R² 향상: +0.05 ~ +0.10

**이상치(Outlier) 처리**
```python
# IQR 기반 극단값 제거 또는 클리핑
Q1, Q3 = np.percentile(y_all, [25, 75])
IQR = Q3 - Q1
mask = (y_all >= Q1 - 3*IQR) & (y_all <= Q3 + 3*IQR)
```
음수 MP(-348K)와 극고온(>1000K) 샘플이 모델 학습을 방해하고 있음.

### 4-2. 모델 개선

**Early Stopping 추가**
```python
model.fit(X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False)
```
n_estimators=1000이 최적인지 검증되지 않음. Early stopping으로 자동 최적화 가능.

**하이퍼파라미터 탐색**
```python
from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.05],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.4, 0.5, 0.6],
}
```

### 4-3. 모델 고도화 (선택적)

| 방법 | 예상 R² | 난이도 |
|------|---------|--------|
| 기술자 200종 확장 | 0.65~0.70 | 낮음 |
| LightGBM 교체 | 0.60~0.65 | 낮음 |
| 이상치 제거 + 기술자 확장 | 0.70~0.75 | 낮음 |
| Neural Network (MLP) | 0.70~0.80 | 중간 |
| Graph Neural Network | 0.80+ | 높음 |
