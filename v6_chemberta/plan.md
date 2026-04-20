# v6_chemberta 설계 계획서

## 개요

v5_ann 파이프라인 위에 **ChemBERTa pretrained embedding**을 concat해 R² 부스트를 검증합니다. v6_stacking과 동일 조건에서 실행해 비교하는 것이 목표.

## 차별점 (v5 대비 유일한 차이)

v5 파이프라인을 그대로 유지하면서 피처 소스만 확장:
- `seyonec/ChemBERTa-zinc-base-v1` (PubChem 77M 분자 pretrained, 768-dim)
- SMILES → tokenizer → BERT → `[CLS]` pooling → 768-dim embedding
- 기존 피처와 concat: Morgan(2048) + MACCS(167) + RDKit(~200) + **ChemBERTa(768)** = 3199
- VT + MI top-300 파이프라인 동일 (MI가 ChemBERTa 비트 중 유용한 것 선별)
- 모델, 학습, 앙상블은 v5와 100% 동일

## v5와 동일한 부분 (Fair Comparison)
- 데이터 (2117, RS=42), Train/Test split, Stratified/Scaffold 이중 CV
- VT(0.01), MI top-300 (fold-내부 fit)
- 모델: ResidualMLP(256×3), Mixup α=0.3 p=0.5, WD 5e-4
- 학습: EPOCHS 200, PATIENCE 25, BATCH 128
- 5-seed Snapshot [42, 53, 65, 79, 93]
- 평가 지표, 시각화, 시간 예산 ≤ 30분

## 예상 성과
- **Test R²: 0.80~0.88** (v5 ensemble 0.7536 대비 +0.05~0.13)
- **Test MAE: 22~28 K** (v5 32.0 대비 큰 개선)
- ChemBERTa가 MI top-300에 얼마나 살아남는지 확인

## 논문 근거
- Chithrananda et al. 2020 (arXiv:2010.09885) — ChemBERTa
- Fabian et al. 2020 (arXiv:2011.13230) — MolBERT
- Ross et al. 2022 (Nature Machine Intelligence) — MolFormer
- Jaeger et al. 2018 (JCIM) — Mol2Vec (고전 baseline)

## 예상 실행 시간
ChemBERTa 추출 (CPU 5~10분) + v5 1.1분 = **~7~12분**

## 주의
- 첫 실행 시 HuggingFace에서 약 400MB 모델 다운로드
- `transformers` 패키지 추가 설치 필요
