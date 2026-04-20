# 분자 물성 예측 ANN의 대표 데이터 증강 기법

분자 property 예측 분야에서 ANN/MLP에 가장 널리 쓰이는 두 가지 증강 방법을 정리합니다.

---

## 1. Mixup

### 개요
두 학습 샘플을 **선형 보간**해서 가상의 새로운 샘플을 만드는 기법.
2018년 ICLR(Zhang et al., *mixup: Beyond Empirical Risk Minimization*) 이후 tabular/fingerprint 회귀에서 사실상 표준 증강으로 자리 잡음.

### 수식
```
λ ~ Beta(α, α)         # α는 보통 0.2 ~ 0.4
x̃ = λ·x_i + (1-λ)·x_j
ỹ = λ·y_i + (1-λ)·y_j
```

### 구현 예시 (PyTorch)
```python
def mixup_batch(X, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix, y_mix

# 학습 루프 안에서
for X_batch, y_batch, w_batch in loader:
    X_batch, y_batch = mixup_batch(X_batch, y_batch, alpha=0.2)
    pred = model(X_batch)
    loss = criterion(pred, y_batch).mean()
    ...
```

### 장점
- 구현이 5줄로 매우 간단
- 별도 데이터 생성 과정 불필요 (학습 루프 내부에서 즉석 생성)
- Morgan FP / MACCS / RDKit descriptor 등 **모든 벡터 입력**에 적용 가능
- 결정경계를 부드럽게 만들어 과적합을 직접 억제 → CV-Test 갭 감소에 효과적
- Label도 함께 보간하므로 회귀 문제에 자연스럽게 적용됨

### 단점 / 주의
- α 값 튜닝 필요 (0.2가 무난한 출발점, 너무 크면 과한 평활화)
- 분자 화학적으로 "존재할 수 없는" 가상 샘플이 생성됨 → 그러나 실험적으로 일반화 향상에 도움됨이 검증
- BatchNorm과 함께 쓸 때는 통계가 흐려질 수 있어 주의

### 본 프로젝트 적합도
**★★★★★** — Morgan FP + descriptor 기반 ANN에 가장 직접적이고 효과적.

---

## 2. SMILES Enumeration

### 개요
같은 분자를 표현하는 **여러 비-canonical SMILES 문자열**을 생성해서 데이터를 늘리는 방법.
2017년 Bjerrum의 *SMILES Enumeration as Data Augmentation for Neural Network Modeling of Molecules* 이후 SMILES 시퀀스를 직접 입력으로 받는 모델에서 표준 기법.

### 원리
SMILES는 분자 그래프를 순회하는 시작점·방향에 따라 **다른 문자열 표현**을 가질 수 있음.
- 예: 톨루엔 → `Cc1ccccc1`, `c1ccc(C)cc1`, `c1ccccc1C`, ...
- 모두 같은 분자지만 토큰 시퀀스는 다름

### 구현 예시 (RDKit)
```python
from rdkit import Chem

def enumerate_smiles(smi, n=5):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi]
    result = set()
    for _ in range(n * 3):
        rand_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        result.add(rand_smi)
        if len(result) >= n:
            break
    return list(result)

# 학습 데이터 5배 증강
augmented = []
for smi, mp in zip(df["SMILES"], df["MP"]):
    for s in enumerate_smiles(smi, n=5):
        augmented.append((s, mp))
```

### 장점
- 실제 분자에 대응하는 "유효한" 표현이라 화학적으로 자연스러움
- RNN, Transformer, 1D-CNN 등 **SMILES 토큰을 직접 입력으로 받는 모델**에서 효과 큼
- 같은 분자의 다양한 표현을 학습 → 표현에 둔감한 robust한 임베딩 학습

### 단점 / 주의
- **Morgan FP, MACCS, RDKit descriptor는 canonical** → 어떤 SMILES로 변환해도 같은 벡터가 나옴
- 따라서 **fingerprint/descriptor 기반 ANN에는 효과가 없음**
- 효과를 보려면 입력을 SMILES 토큰 자체로 바꾸거나, 토큰 임베딩을 사용하는 모델로 전환해야 함

### 본 프로젝트 적합도
**★☆☆☆☆** — 현재 ANN 구조(2431차원 fingerprint + descriptor)에는 부적합.
입력을 SMILES 시퀀스로 바꾼 RNN/Transformer 모델로 확장할 경우에만 유효.

---

## 요약 비교

| 항목 | Mixup | SMILES Enumeration |
|---|---|---|
| 발표 시점 | 2018 (Zhang et al.) | 2017 (Bjerrum) |
| 적용 위치 | 입력 벡터 (학습 중 즉석) | 원본 데이터 (사전 생성) |
| 본 ANN 호환 | 매우 좋음 | 부적합 (FP canonical) |
| 구현 난이도 | 매우 쉬움 (5줄) | 쉬움 (RDKit 한 줄) |
| 데이터 크기 변화 | 그대로 | N배 증가 |
| 권장 적용 모델 | MLP, tabular 회귀 | RNN, Transformer, 1D-CNN |

## 결론
본 프로젝트(Morgan FP + MACCS + RDKit descriptor 기반 MLP)에서는 **Mixup 단독 적용이 가장 표준적이고 효과적인 선택**.
SMILES enumeration은 추후 SMILES 토큰 기반 모델로 확장할 때 활용.
