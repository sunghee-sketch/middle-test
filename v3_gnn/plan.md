# v3_gnn 설계 계획서

## 왜 GNN(Graph Neural Network)을 선택했는가?

base·v2(트리 앙상블)와 v3_ann(MLP)은 모두 **사전 계산된 벡터 피처**(Morgan Fingerprint, MACCS, RDKit 기술자)에 의존했습니다. 이 방식은 분자의 구조 정보를 **사람이 미리 정의한 규칙**으로 압축하므로, 규칙에 포함되지 않은 구조적 특징은 원천적으로 학습할 수 없습니다.

GNN은 **분자를 원자-결합 그래프로 직접 입력**받아, 모델이 스스로 "어떤 구조가 녹는점에 중요한가"를 학습합니다. 피처 엔지니어링을 모델에 위임하는 접근입니다.

| 비교 항목 | 트리/MLP (벡터 피처) | GNN (그래프) |
|----------|----------------------|--------------|
| 입력 | 사전 계산된 고정 차원 벡터 | 원자-결합 그래프 |
| 피처 엔지니어링 | 사람이 설계 | 모델이 학습 |
| 분자 크기 | 동일 벡터로 압축 | 가변 크기 그래프 그대로 |
| 국소 구조 표현 | FP 비트로 간접 표현 | Message passing으로 직접 |
| 해석성 | 피처 중요도 | 원자 단위 attribution |

---

## 모델 아키텍처

### GIN (Graph Isomorphism Network) 구조

```
SMILES
  ↓  RDKit 파싱
분자 그래프 (원자 노드 + 결합 엣지)
  ↓  노드 피처 7차원, 엣지 피처 1차원
GINConv(7 → 128)    → BatchNorm → ReLU
  ↓
GINConv(128 → 128)  → BatchNorm → ReLU
  ↓
GINConv(128 → 128)  → BatchNorm → ReLU
  ↓
GINConv(128 → 128)  → BatchNorm → ReLU
  ↓
Global Mean Pool ⊕ Global Max Pool  (분자 단위 표현, 256차원)
  ↓
Linear(256 → 128) → ReLU → Dropout(0.3)
  ↓
Linear(128 → 64)  → ReLU
  ↓
Linear(64 → 1)  → log1p(MP) 예측
```

### 왜 GIN인가?

GIN은 **그래프 구조 구별력이 이론적으로 가장 강한 GNN**으로, Weisfeiler-Lehman 그래프 동형성 판별과 동등한 표현력을 가집니다. 분자처럼 미세한 구조 차이(예: ortho vs meta 치환)가 물성에 큰 영향을 주는 문제에 적합합니다.

GCN·GAT 대비 장점:
- **GCN**: 이웃 평균만 사용 → 동형 구조 구별 한계
- **GAT**: 어텐션 가중치 학습 → 파라미터 증가, 소규모 데이터에서 과적합
- **GIN**: `(1+ε)·self + sum(neighbors)` + 학습 가능한 MLP → 표현력·안정성 균형

### Global Mean + Max Pooling 병행

단일 풀링의 한계를 상호 보완합니다.
- **Mean pool**: 분자 전체 평균 특성 (전반적 경향)
- **Max pool**: 가장 돌출된 원자 특성 (특이 구조 감지)
- 두 벡터를 concat하여 256차원 분자 표현 생성

---

## 입력 피처 설계

### 노드(원자) 피처 — 7차원

| # | 피처 | 정규화 | 의미 |
|---|------|--------|------|
| 1 | AtomicNum | /100 | 원소 종류 |
| 2 | Degree | /6 | 결합 수 |
| 3 | FormalCharge | /4 | 형식 전하 |
| 4 | IsInRing | 0/1 | 고리 원자 여부 |
| 5 | IsAromatic | 0/1 | 방향족 여부 |
| 6 | TotalNumHs | /4 | 수소 개수 |
| 7 | Hybridization | /5 | SP/SP2/SP3/SP3D/SP3D2/OTHER |

### 엣지(결합) 피처 — 1차원

| 값 | 결합 종류 |
|----|----------|
| 0/3 | 단일 |
| 1/3 | 이중 |
| 2/3 | 삼중 |
| 3/3 | 방향족 |

> **최소 피처 설계 원칙**: 복잡한 원자 피처(예: 화학적 환경, 부분전하) 대신 기본 속성만 주고, 모델이 message passing으로 나머지를 학습하도록 유도.

---

## 학습 전략

### 타깃 변환
- `log1p(MP)` 로 학습 → 예측 후 `expm1()` 역변환
- v3_ann과 동일. 고온 분포 왜곡 완화.

### 손실 함수
- `F.mse_loss(pred, batch.y)` — 샘플 가중치 **미적용** (v3_ann과 차이)
- GNN은 그래프 단위 학습이라 샘플 가중치 도입이 구현 복잡도 대비 이득이 적음

### 조기 종료 (Early Stopping)
- CV: `patience=15`
- Final: `patience=15`, max 200 epochs

### ReduceLROnPlateau
- 검증 손실 8 epoch 개선 없으면 학습률 절반

### StratifiedKFold
- `pd.qcut(y, q=10)` 10구간 분층 (v3_ann과 동일)

---

## 하이퍼파라미터

| 파라미터 | 값 | 선택 이유 |
|---------|-----|---------|
| GIN layers | 4 | 분자 반경 4 정도까지 구조 인지 (Morgan radius 2와 유사 범위) |
| Hidden dim | 128 | 파라미터 수와 표현력의 균형 |
| Epochs | 100 (CV) / 200 (Final) | Early stopping으로 자동 결정 |
| Batch size | 32 | 그래프 배치는 원자 수 합으로 커지므로 ANN보다 작게 |
| Learning rate | 1e-3 | Adam 기본값 |
| Weight decay | 1e-4 | L2 정규화 |
| Dropout (head) | 0.3 | MLP head 부분만 적용 |

---

## v3_ann 대비 차이점

| 항목 | v3_ann (MLP) | v3_gnn (GIN) |
|------|-------------|-------------|
| 입력 | 2431차원 벡터 | 그래프 (가변 크기) |
| 피처 엔지니어링 | Morgan + MACCS + 기술자 | 원자 7차원 + 결합 1차원 |
| 전처리 | StandardScaler + 클리핑 | 정규화 (/100, /6 등) |
| 샘플 가중치 | 고MP 상위 10%에 3.0 | 없음 |
| Batch size | 64 | 32 |
| 파라미터 수 | 약 130만 | 약 10만 이하 (훨씬 가벼움) |

---

## 예상 성능

| 단계 | 예상 CV R² |
|------|-----------|
| base (XGBoost 튜닝) | 0.5346 (실측) |
| v3_ann (MLP) | 0.7065 (실측) |
| v3_gnn (GIN) | 0.55 ~ 0.70 (예상) |

### 성능 예측 근거

**상방 요인**
- 분자 구조를 직접 학습 → 새로운 구조 패턴 포착 가능성
- 파라미터가 적어 과적합 저항력이 상대적으로 큼
- v3_ann에서 확대됐던 CV-Test 격차(0.124)가 좁혀질 여지

**하방 요인**
- 노드 피처 7차원은 Morgan 2048비트 대비 정보량 부족
- 2117개 샘플은 GNN이 밑바닥부터 구조를 배우기엔 다소 적음
- 샘플 가중치 미적용으로 고MP 영역 성능 저하 가능

---

## 실행 환경

| 항목 | 값 |
|------|-----|
| 환경 | conda `analchem_flower` |
| Python | 3.10.20 |
| PyTorch | 2.11.0 |
| PyTorch Geometric | 2.7.0 |
| 디바이스 | CPU (MPS/CUDA 미사용, 노트북 기본 설정) |

---

## 관전 포인트

1. **CV-Test 격차**: v3_ann(0.124) 대비 좁혀지는가? GNN의 구조적 일반화 능력 확인.
2. **Fold 편차**: v3_ann의 Fold 2(0.48) ↔ Fold 5(0.82) 격차가 어떻게 나타나는가?
3. **고MP 영역**: 샘플 가중치가 없으므로 >1000 K 영역 예측 오차를 중점 관찰.
4. **트리/MLP/GNN의 강점 분화**: 실패 패턴이 달라지면 앙상블/스태킹으로 상호 보완 여지.
