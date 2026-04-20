"""Mixup ablation study for v4_ann.

Runs the full v4_ann pipeline with 3 different mixup configs and compares:
  - no_mixup  : mixup_prob=0.0
  - weak      : alpha=0.1, prob=0.5
  - baseline  : alpha=0.3, prob=0.5  (v4_ann default)
  - strong    : alpha=0.5, prob=0.5
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======== Feature extraction (once, shared across ablations) ========
def build_features():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/sunghee-sketch/middle-test/main/data/Melting_point_2.csv"
    )
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df["MP"] >= 0].reset_index(drop=True)
    df["MP_log"] = np.log1p(df["MP"])

    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    desc_list = [(n, f) for n, f in Descriptors.descList if not n.startswith("Ipc")]

    records, valid_idx = [], []
    for i, smi in enumerate(df["SMILES"].tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        morgan = morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        descs = []
        for _, func in desc_list:
            try:
                v = func(mol)
                descs.append(float(v) if (v is not None and np.isfinite(v)) else 0.0)
            except Exception:
                descs.append(0.0)
        records.append(np.concatenate([morgan, maccs, np.array(descs, np.float32)]))
        valid_idx.append(i)
    X = np.nan_to_num(np.array(records, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = df["MP"].values[valid_idx]
    y_log = df["MP_log"].values[valid_idx]
    return X, y, y_log


def preprocess(X_all, y_all, y_all_log):
    vt = VarianceThreshold(threshold=0.01)
    X_all_vt = vt.fit_transform(X_all).astype(np.float32)

    mp_bins = pd.qcut(y_all, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, y_tr_log, y_te_log = train_test_split(
        X_all_vt, y_all, y_all_log,
        test_size=0.2, random_state=RANDOM_STATE, stratify=mp_bins,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)
    X_train_sc = np.clip(np.nan_to_num(X_train_sc, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0).astype(np.float32)
    X_test_sc = np.clip(np.nan_to_num(X_test_sc, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0).astype(np.float32)

    y_tr_log = y_tr_log.astype(np.float32)
    y_te_log = y_te_log.astype(np.float32)

    high_thr = np.percentile(y_train, 90)
    sw_train = np.where(y_train >= high_thr, 3.0, 1.0).astype(np.float32)

    # MI top-300
    mi = mutual_info_regression(X_train_sc, y_tr_log, random_state=RANDOM_STATE, n_neighbors=3)
    top_idx = np.argsort(mi)[::-1][:300]
    X_train_sel = X_train_sc[:, top_idx]
    X_test_sel = X_test_sc[:, top_idx]

    return X_train_sel, X_test_sel, y_train, y_test, y_tr_log, y_te_log, sw_train


# ======== Model ========
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.act(self.ln1(self.fc1(x))))
        h = self.ln2(self.fc2(h))
        return x + h


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, n_blocks=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(1)


def mixup_batch(X, y, w, alpha):
    if alpha <= 0:
        return X, y, w
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], lam * y + (1 - lam) * y[idx], lam * w + (1 - lam) * w[idx]


def train_epoch(model, loader, optimizer, criterion, mixup_alpha, mixup_prob):
    model.train()
    total_loss = 0
    for X_b, y_b, w_b in loader:
        X_b, y_b, w_b = X_b.to(DEVICE), y_b.to(DEVICE), w_b.to(DEVICE)
        if mixup_prob > 0 and np.random.rand() < mixup_prob:
            X_b, y_b, w_b = mixup_batch(X_b, y_b, w_b, alpha=mixup_alpha)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = (criterion(pred, y_b) * w_b).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict(model, X_sc):
    model.eval()
    X_t = torch.tensor(X_sc).to(DEVICE)
    return model(X_t).cpu().numpy()


# ======== Full pipeline for one mixup config ========
def run_pipeline(X_train_sel, X_test_sel, y_train, y_test, y_tr_log, y_te_log, sw_train,
                 mixup_alpha, mixup_prob, config_name):
    INPUT_DIM = X_train_sel.shape[1]
    EPOCHS, BATCH, LR, PATIENCE, WD = 80, 128, 1e-3, 12, 5e-4

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mp_bins_train = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
    cv_r2, cv_mse, cv_mae = [], [], []

    t_cv = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_sel, mp_bins_train), 1):
        X_tr, X_val = X_train_sel[tr_idx], X_train_sel[val_idx]
        y_tr, y_val = y_tr_log[tr_idx], y_tr_log[val_idx]
        sw_tr = sw_train[tr_idx]
        y_val_orig = y_train[val_idx]

        ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(sw_tr))
        ldr = DataLoader(ds, batch_size=BATCH, shuffle=True)

        torch.manual_seed(RANDOM_STATE + fold)
        np.random.seed(RANDOM_STATE + fold)
        model = ResidualMLP(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
        criterion = nn.MSELoss(reduction="none")

        best_val_loss, patience_cnt, best_state = np.inf, 0, None
        for epoch in range(EPOCHS):
            train_epoch(model, ldr, optimizer, criterion, mixup_alpha, mixup_prob)
            val_loss = mean_squared_error(y_val, predict(model, X_val))
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

        model.load_state_dict(best_state)
        pred = np.expm1(predict(model, X_val))
        cv_r2.append(r2_score(y_val_orig, pred))
        cv_mse.append(mean_squared_error(y_val_orig, pred))
        cv_mae.append(mean_absolute_error(y_val_orig, pred))
    cv_time = time.time() - t_cv

    # Final: 3-seed snapshot ensemble
    X_tr_f, X_val_f, y_tr_f, y_val_f, sw_f, _ = train_test_split(
        X_train_sel, y_tr_log, sw_train,
        test_size=0.1, random_state=RANDOM_STATE,
    )
    SEEDS = [RANDOM_STATE, RANDOM_STATE + 11, RANDOM_STATE + 23]
    test_preds_log = []

    t_final = time.time()
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ResidualMLP(INPUT_DIM).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
        criterion = nn.MSELoss(reduction="none")
        ds_f = TensorDataset(torch.tensor(X_tr_f), torch.tensor(y_tr_f), torch.tensor(sw_f))
        ldr_f = DataLoader(ds_f, batch_size=BATCH, shuffle=True)

        best_val_loss, patience_cnt, best_state = np.inf, 0, None
        for epoch in range(120):
            train_epoch(model, ldr_f, optimizer, criterion, mixup_alpha, mixup_prob)
            vl = mean_squared_error(y_val_f, predict(model, X_val_f))
            scheduler.step(vl)
            if vl < best_val_loss:
                best_val_loss, patience_cnt = vl, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break
        model.load_state_dict(best_state)
        test_preds_log.append(predict(model, X_test_sel))

    y_pred = np.expm1(np.mean(test_preds_log, axis=0))
    test_r2 = r2_score(y_test, y_pred)
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    final_time = time.time() - t_final

    return {
        "config": config_name,
        "mixup_alpha": mixup_alpha,
        "mixup_prob": mixup_prob,
        "cv_r2_mean": np.mean(cv_r2),
        "cv_r2_std": np.std(cv_r2),
        "cv_r2_per_fold": cv_r2,
        "cv_mae": np.mean(cv_mae),
        "test_r2": test_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "gap": np.mean(cv_r2) - test_r2,
        "cv_time": cv_time,
        "final_time": final_time,
    }


def main():
    print("=" * 70)
    print("v4_ann Mixup Ablation Study")
    print("=" * 70)

    T0 = time.time()
    print("\n[1/2] 피처 추출 중...")
    X_all, y_all, y_all_log = build_features()
    print(f"    완료: {X_all.shape}, {time.time()-T0:.1f}s")

    print("\n[2/2] 전처리 (VT + Scale + MI top-300)...")
    t = time.time()
    X_train_sel, X_test_sel, y_train, y_test, y_tr_log, y_te_log, sw_train = preprocess(
        X_all, y_all, y_all_log
    )
    print(f"    완료: Train {X_train_sel.shape}, Test {X_test_sel.shape}, {time.time()-t:.1f}s")

    configs = [
        ("no_mixup",  0.0, 0.0),
        ("weak",      0.1, 0.5),
        ("baseline",  0.3, 0.5),
        ("strong",    0.5, 0.5),
    ]

    results = []
    for name, alpha, prob in configs:
        print(f"\n{'='*70}")
        print(f"Running: {name}  (alpha={alpha}, prob={prob})")
        print('=' * 70)
        t = time.time()
        r = run_pipeline(X_train_sel, X_test_sel, y_train, y_test, y_tr_log, y_te_log, sw_train,
                         alpha, prob, name)
        print(f"  CV R² = {r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}  |  Test R² = {r['test_r2']:.4f}  |  "
              f"Gap = {r['gap']:+.4f}  |  Test MAE = {r['test_mae']:.2f}")
        print(f"  Folds: {['%.3f' % x for x in r['cv_r2_per_fold']]}")
        print(f"  Time: {time.time()-t:.1f}s")
        results.append(r)

    print(f"\n{'='*70}")
    print("최종 비교")
    print('=' * 70)
    df = pd.DataFrame([
        {
            "Config":    r["config"],
            "α":         r["mixup_alpha"],
            "p":         r["mixup_prob"],
            "CV R²":     f"{r['cv_r2_mean']:.4f}",
            "CV std":    f"±{r['cv_r2_std']:.4f}",
            "Test R²":   f"{r['test_r2']:.4f}",
            "Test MAE":  f"{r['test_mae']:.2f}",
            "Gap":       f"{r['gap']:+.4f}",
        }
        for r in results
    ])
    print(df.to_string(index=False))
    print(f"\n총 실행 시간: {(time.time()-T0)/60:.1f}분")

    # Save to CSV
    out_csv = os.path.join(os.path.dirname(__file__), "ablation_mixup_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n저장: {out_csv}")


if __name__ == "__main__":
    main()
