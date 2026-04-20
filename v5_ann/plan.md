# v5_ann 설계 계획서 (3차 개선)

## 왜 v5인가?

v4_ann 실측 결과:
- Test R² **0.6958** (목표 0.70~0.75 하단 근접)
- Test MAE **32.53 K** (v3 대비 -14 K)
- CV-Test 갭 **-0.026 (역전)**
- 실행 시간 **0.5분** (예산 30분의 1.7%)

하지만 다음 한계:
1. **Option B 미적용** — MI가 분할 전 전체 train에서 fit → CV 누수 남음
2. **Fold 3이 epoch 25에서 조기 종료** — patience=12가 너무 엄격, 과소적합 의심
3. **CV 편차 ±0.124** 여전 — Fold 2 R²=0.48 (v3·v4 동일 문제)
4. **v3 비교가 CV만으로는 불공정** (누수 때문)

5인 전문가 2차 토의(만장일치 수렴) 결과:
- 라운드 2 패치 미반영 → 우선 적용
- Epochs/patience 상향 (3/5 동의, 과소적합 증거)
- Scaffold split 병행 출력 (만장일치)
- 5-seed 확장 (저비용)
- Fold 2 진단 (feature-expert 제안)

---

## 3차 개선 내역

### A. 라운드 2 패치 전면 적용
| 항목 | v4 (실행됨) | v5 |
|---|---|---|
| MI fit 위치 | 전체 train | **fold-내부 tr_idx only (Option B)** |
| proj layer | Linear 단독 | **Linear + LayerNorm** (pre-norm 효과) |
| Test 출력 | ensemble만 | **single-seed + ensemble 둘 다** (공정 비교) |
| Fold overlap | 없음 | **Jaccard overlap 출력** (selection 안정성) |

### B. 학습량 상향 (과소적합 해소)
| 항목 | v4 | v5 |
|---|---|---|
| EPOCHS | 80 | **200** |
| PATIENCE | 12 | **25** |
| LR scheduler patience | 6 | **10** |
| MAX_FINAL_EPOCHS | 120 | **250** |

근거: v4 Fold 3이 epoch 25에서 조기 종료. Mixup(p=0.5)이 손실을 요동치게 하므로 patience 여유 필요.

### C. Scaffold Split 병행 출력
- RDKit `MurckoScaffold.MurckoScaffoldSmiles`로 각 분자의 Bemis-Murcko scaffold 추출
- Scaffold 단위 `GroupKFold(n_splits=5)` 추가 CV
- 두 split 결과 동시 보고 → v3 비교 호환성(StratifiedKFold) + 정직한 일반화(ScaffoldSplit) 동시 확보

### D. 5-seed Snapshot Ensemble
- 최종 모델: 3-seed → 5-seed
- 비용: +~10초. snapshot 한계 효용 K=5에서 95% 수렴

### E. Fold 2 진단 셀
- Morgan fingerprint 기반 k-means (k=5~8) 클러스터링
- 각 StratifiedKFold fold의 cluster 분포 출력
- Fold 2 val에 특정 cluster 집중 여부 확인 → 구조 편향 가설 검증

---

## 파이프라인

```
원본 2117샘플 / 2431 피처
  → VarianceThreshold(0.01) → 504
  → StandardScaler + 클리핑 [-10, 10]
  → Stratified CV (v4 비교용):
       · 각 fold 내부에서 MI top-300 (Option B)
       · ResidualMLP + Mixup 학습
  → Scaffold CV (정직한 일반화):
       · Murcko scaffold GroupKFold
       · 동일 ResidualMLP 학습
  → Fold 2 진단:
       · Morgan 기반 k-means
       · fold별 cluster 분포 출력
  → 최종 모델:
       · MI top-300 (전체 train fit — Test용)
       · 5-seed Snapshot
       · single-seed vs 5-seed ensemble 둘 다 출력
```

---

## 하이퍼파라미터 (v4 vs v5)

| 항목 | v4 | v5 |
|---|---|---|
| hidden | 256 | 256 |
| n_blocks | 3 | 3 |
| activation | Swish | Swish |
| norm | LayerNorm | LayerNorm (+ **proj LN 추가**) |
| dropout | 0.3 | 0.3 |
| lr | 1e-3 | 1e-3 |
| weight_decay | 5e-4 | 5e-4 |
| **EPOCHS** | 80 | **200** |
| **PATIENCE** | 12 | **25** |
| BATCH | 128 | 128 |
| mixup_α | 0.3 | 0.3 |
| mixup_prob | 0.5 | 0.5 |
| **ensemble seeds** | 3 | **5** |
| MI fit | 전체 train | **fold-내부 (CV), 전체 train (final)** |

---

## 예상 성과

| 단계 | 예상 CV R² | 예상 Test R² | 비고 |
|---|---|---|---|
| v3_ann | 0.7065 | 0.5821 | 과적합 |
| v4_ann (round 1) | 0.6702 | 0.6958 | 갭 역전 |
| v5_ann (Stratified CV) | 0.68~0.72 | **0.72~0.76** | Epochs↑ + 5-seed |
| v5_ann (Scaffold CV) | 0.55~0.65 | **0.72~0.76** | 정직한 generalization |

Test R² 목표 0.75 진입 가능. Scaffold CV는 stratified 대비 낮게 나오는 것이 정상 (어려운 split).

---

## 후속 작업 (v5 1차 빌드 후)

합의안 중 다음은 v5 2차/3차에서 진행:
1. **Stacking with v2 XGBoost/LightGBM** — ensemble-expert 강력 추천
2. **Optuna ASHA 40-trial joint search** — hyperparam-expert 제안
3. **Tautomer enumeration** — augmentation-expert 제안
4. **Scaffold-aware Mixup** — architecture-expert 제안 (구조적 보간)

---

## 실행 시간 예상

| 단계 | 예상 소요 |
|---|---|
| 피처 추출 + scaffold 추출 | 10초 |
| VT + split + scaling | 2초 |
| Stratified CV (fold-내부 MI + 200 epoch) | 3~5분 |
| Scaffold CV (fold-내부 MI + 200 epoch) | 3~5분 |
| Fold 2 진단 | 30초 |
| 5-seed Final | 1~2분 |
| 시각화 + 요약 | 10초 |
| **총** | **8~13분** |

30분 예산 내 여유.
