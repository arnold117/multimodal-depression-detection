#!/usr/bin/env python3
"""
Phase 16h — Deep Learning Baseline
====================================
Analysis 42: Compare passive sensing prediction using:
  (a) Ridge on PCA composites (existing baseline)
  (b) GradientBoosting + Optuna on raw aggregate features
  (c) 1D-CNN on daily time series (MPS-accelerated)
  (d) MOMENT foundation model embeddings → Ridge
  (e) Personality Ridge (reference)

Tests whether modern ML/DL can rescue sensing's poor predictive validity.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = Path("results/robustness")
OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]
RS = 42
COHORTS = ["INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4"]
RAW_DIR = Path("data/raw/globem")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Device: {DEVICE}", flush=True)


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════
print("Loading datasets...", flush=True)
s2 = pd.read_parquet("data/processed/nethealth/nethealth_analysis_dataset.parquet")
s3 = pd.read_parquet("data/processed/globem/globem_analysis_dataset.parquet")
S2_BEH_PCA = [c for c in s2.columns if c.startswith("nh_") and c.endswith("_pc1")]
S3_BEH_PCA = [c for c in s3.columns if c.endswith("_pc1") and not c.startswith("nh_")]

S3_OUTCOMES = {
    "bdi2_total": "BDI-II", "stai_state": "STAI",
    "pss_10": "PSS-10", "cesd_total": "CESD", "ucla_loneliness": "UCLA",
}
S2_OUTCOMES = {
    "cesd_total": "CES-D", "stai_trait_total": "STAI", "bai_total": "BAI",
}


def find_col(columns, substring):
    matches = [c for c in columns if substring in c]
    return min(matches, key=len) if matches else None


# ═══════════════════════════════════════════════════════════════════════
# Load S3 daily time series → per-person matrix (N × T × C)
# ═══════════════════════════════════════════════════════════════════════
def load_s3_daily(max_days=90):
    """Load S3 daily sensing data, return dict {pid: array(T, C)}."""
    FEATURE_MAP = [
        ("steps.csv", "avgsumsteps:14dhist", "steps"),
        ("sleep.csv", "avgdurationasleepmain:14dhist", "sleep_dur"),
        ("sleep.csv", "avgefficiencymain:14dhist", "sleep_eff"),
        ("screen.csv", "rapids_sumdurationunlock:14dhist", "screen_dur"),
        ("call.csv", "incoming_count:14dhist", "calls_in"),
        ("location.csv", "barnett_hometime:14dhist", "hometime"),
    ]

    daily_by_feat = {}
    for modality_file, col_substr, short_name in FEATURE_MAP:
        all_daily = []
        for cohort in COHORTS:
            fpath = RAW_DIR / cohort / "FeatureData" / modality_file
            if not fpath.exists():
                continue
            df_raw = pd.read_csv(fpath, low_memory=False)
            col = find_col(df_raw.columns.tolist(), col_substr)
            if col is None:
                continue
            daily = df_raw[["pid", "date", col]].copy()
            daily.columns = ["pid", "date", "value"]
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            daily["date"] = pd.to_datetime(daily["date"])
            all_daily.append(daily)
        if all_daily:
            daily_by_feat[short_name] = pd.concat(all_daily, ignore_index=True)

    feat_names = sorted(daily_by_feat.keys())
    print(f"  Loaded {len(feat_names)} daily features: {feat_names}", flush=True)

    all_pids = set()
    for df in daily_by_feat.values():
        all_pids.update(df["pid"].unique())

    person_data = {}
    for pid in all_pids:
        frames = []
        for feat in feat_names:
            df = daily_by_feat[feat]
            sub = df[df["pid"] == pid].sort_values("date")
            if len(sub) == 0:
                frames.append(None)
                continue
            frames.append(sub[["date", "value"]].rename(columns={"value": feat}))

        if all(f is None for f in frames):
            continue

        merged = frames[0]
        for f in frames[1:]:
            if f is None:
                continue
            merged = merged.merge(f, on="date", how="outer")

        merged = merged.sort_values("date").reset_index(drop=True)

        if len(merged) > max_days:
            merged = merged.tail(max_days).reset_index(drop=True)

        mat = merged[feat_names].values
        df_temp = pd.DataFrame(mat, columns=feat_names)
        df_temp = df_temp.ffill().fillna(0)
        mat = df_temp.values

        if mat.shape[0] >= 7:
            person_data[pid] = mat

    print(f"  S3 daily: {len(person_data)} persons with ≥7 days", flush=True)
    return person_data, feat_names


# ═══════════════════════════════════════════════════════════════════════
# Load S2 daily time series
# ═══════════════════════════════════════════════════════════════════════
def load_s2_daily(max_days=180):
    """Load S2 daily Fitbit data, return dict {egoid: array(T, C)}."""
    act = pd.read_csv("data/raw/nethealth/FitbitActivity(1-30-20).csv", low_memory=False)
    slp = pd.read_csv("data/raw/nethealth/FitbitSleep(1-30-20).csv", low_memory=False)

    act["datadate"] = pd.to_datetime(act["datadate"])
    # Sleep file uses "dataDate" (capital D)
    slp = slp.rename(columns={"dataDate": "datadate"})
    slp["datadate"] = pd.to_datetime(slp["datadate"])

    act_feats = ["steps", "sedentaryminutes", "lightlyactiveminutes",
                 "fairlyactiveminutes", "veryactiveminutes"]
    act_sub = act[["egoid", "datadate"] + act_feats].copy()
    for c in act_feats:
        act_sub[c] = pd.to_numeric(act_sub[c], errors="coerce")

    slp_feats = ["minsasleep", "minsawake", "Efficiency"]
    slp_feats = [c for c in slp_feats if c in slp.columns]

    slp_sub = slp[["egoid", "datadate"] + slp_feats].copy()
    for c in slp_feats:
        slp_sub[c] = pd.to_numeric(slp_sub[c], errors="coerce")

    merged = act_sub.merge(slp_sub, on=["egoid", "datadate"], how="outer")
    all_feats = act_feats + slp_feats

    person_data = {}
    for egoid, grp in merged.groupby("egoid"):
        grp = grp.sort_values("datadate").reset_index(drop=True)
        if len(grp) > max_days:
            grp = grp.tail(max_days).reset_index(drop=True)
        mat = grp[all_feats].values
        df_temp = pd.DataFrame(mat, columns=all_feats)
        df_temp = df_temp.ffill().fillna(0)
        mat = df_temp.values
        if mat.shape[0] >= 7:
            person_data[egoid] = mat

    print(f"  S2 daily: {len(person_data)} persons with ≥7 days, {len(all_feats)} features", flush=True)
    return person_data, all_feats


# ═══════════════════════════════════════════════════════════════════════
# Pad/truncate to fixed length
# ═══════════════════════════════════════════════════════════════════════
def pad_to_length(person_data, target_len):
    """Pad (zero) or truncate each person's matrix to (target_len, C)."""
    result = {}
    for pid, mat in person_data.items():
        T, C = mat.shape
        if T >= target_len:
            result[pid] = mat[-target_len:]
        else:
            padded = np.zeros((target_len, C))
            padded[-T:] = mat
            result[pid] = padded
    return result


# ═══════════════════════════════════════════════════════════════════════
# 1D-CNN Model
# ═══════════════════════════════════════════════════════════════════════
class SensingCNN(nn.Module):
    def __init__(self, n_channels, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def train_cnn(X_train, y_train, X_test, n_channels, seq_len,
              epochs=100, lr=1e-3, batch_size=32):
    """Train 1D-CNN on MPS, return predictions for X_test."""
    model = SensingCNN(n_channels, seq_len).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train.transpose(0, 2, 1)).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test.transpose(0, 2, 1)).to(DEVICE)

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience_counter = 0
    patience = 15
    best_state = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te).cpu().numpy()
    return preds


# ═══════════════════════════════════════════════════════════════════════
# MOMENT Embedding → Ridge
# ═══════════════════════════════════════════════════════════════════════
def get_moment_embeddings(person_data_padded, seq_len):
    """Extract MOMENT embeddings for each person's time series."""
    try:
        from momentfm import MOMENTPipeline
    except ImportError:
        print("  momentfm not installed, skipping MOMENT", flush=True)
        return None

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "embedding"},
    )
    model.init()

    pids = sorted(person_data_padded.keys())
    all_embeddings = []

    for i, pid in enumerate(pids):
        mat = person_data_padded[pid]  # (T, C)
        T, C = mat.shape

        if T > 512:
            mat = mat[-512:]
            T = 512

        mat_norm = mat.copy()
        for c in range(C):
            m, s = mat_norm[:, c].mean(), mat_norm[:, c].std()
            if s > 0:
                mat_norm[:, c] = (mat_norm[:, c] - m) / s

        x = torch.FloatTensor(mat_norm.T).unsqueeze(0)  # (1, C, T)

        with torch.no_grad():
            output = model(x_enc=x)
            emb = output.embeddings.squeeze(0).numpy()  # (1024,)
            all_embeddings.append(emb)

        if (i + 1) % 100 == 0:
            print(f"    MOMENT: {i+1}/{len(pids)} persons", flush=True)

    embeddings = np.array(all_embeddings)
    print(f"  MOMENT embeddings: {embeddings.shape}", flush=True)
    return dict(zip(pids, embeddings))


# ═══════════════════════════════════════════════════════════════════════
# GradientBoosting + Optuna
# ═══════════════════════════════════════════════════════════════════════
def optuna_gb_cv(X, y, n_splits=5):
    """Tune GradientBoosting with Optuna, return best R2."""
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 30:
        return np.nan, len(y_c), []

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        }
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=3, random_state=RS)
        r2s = []
        for tr, te in cv.split(X_c):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_c[tr])
            Xte = sc.transform(X_c[te])
            gb = GradientBoostingRegressor(random_state=RS, **params)
            gb.fit(Xtr, y_c[tr])
            r2s.append(r2_score(y_c[te], gb.predict(Xte)))
        return np.mean(r2s)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=5, random_state=RS)
    best_params = study.best_params
    r2s = []
    for tr, te in cv.split(X_c):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_c[tr])
        Xte = sc.transform(X_c[te])
        gb = GradientBoostingRegressor(random_state=RS, **best_params)
        gb.fit(Xtr, y_c[tr])
        r2s.append(r2_score(y_c[te], gb.predict(Xte)))

    return float(np.mean(r2s)), len(y_c), r2s


# ═══════════════════════════════════════════════════════════════════════
# Quick Ridge CV (matches existing pattern)
# ═══════════════════════════════════════════════════════════════════════
def quick_cv_r2(X, y, n_splits=5, n_repeats=5, alpha=1.0):
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X_c, y_c = X[mask], y[mask]
    if len(y_c) < 30:
        return np.nan, len(y_c), []
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    r2s = []
    for tr, te in cv.split(X_c):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_c[tr])
        Xte = sc.transform(X_c[te])
        m = Ridge(alpha=alpha)
        m.fit(Xtr, y_c[tr])
        r2s.append(r2_score(y_c[te], m.predict(Xte)))
    return float(np.mean(r2s)), len(y_c), r2s


# ═══════════════════════════════════════════════════════════════════════
# CNN CV
# ═══════════════════════════════════════════════════════════════════════
def cnn_cv_r2(X_3d, y, n_splits=5, n_repeats=3):
    """Cross-validate 1D-CNN. X_3d shape: (N, T, C)."""
    mask = ~np.isnan(y)
    X_c, y_c = X_3d[mask], y[mask]
    if len(y_c) < 30:
        return np.nan, len(y_c), []

    N, T, C = X_c.shape
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    r2s = []

    for fold_i, (tr, te) in enumerate(cv.split(X_c)):
        X_tr, X_te = X_c[tr].copy(), X_c[te].copy()
        y_tr, y_te = y_c[tr], y_c[te]

        for c in range(C):
            m, s = X_tr[:, :, c].mean(), X_tr[:, :, c].std()
            if s > 0:
                X_tr[:, :, c] = (X_tr[:, :, c] - m) / s
                X_te[:, :, c] = (X_te[:, :, c] - m) / s

        preds = train_cnn(X_tr, y_tr, X_te, n_channels=C, seq_len=T)
        r2s.append(r2_score(y_te, preds))

        if (fold_i + 1) % 5 == 0:
            print(f"      CNN fold {fold_i+1}/{n_splits*n_repeats}", flush=True)

    return float(np.mean(r2s)), len(y_c), r2s


# ═══════════════════════════════════════════════════════════════════════
# MOMENT CV
# ═══════════════════════════════════════════════════════════════════════
def moment_cv_r2(embeddings_dict, outcome_df, outcome_col, n_splits=5, n_repeats=5):
    """Cross-validate MOMENT embeddings → Ridge."""
    if embeddings_dict is None:
        return np.nan, 0, []

    pids = []
    X_list = []
    y_list = []
    for pid, emb in embeddings_dict.items():
        row = outcome_df[outcome_df["pid"] == pid] if "pid" in outcome_df.columns else None
        if row is None or len(row) == 0:
            continue
        y_val = row[outcome_col].values[0]
        if np.isnan(y_val):
            continue
        pids.append(pid)
        X_list.append(emb)
        y_list.append(y_val)

    if len(y_list) < 30:
        return np.nan, len(y_list), []

    X = np.array(X_list)
    y = np.array(y_list)

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    r2s = []
    for tr, te in cv.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        m = Ridge(alpha=1.0)
        m.fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], m.predict(Xte)))

    return float(np.mean(r2s)), len(y), r2s


# ═══════════════════════════════════════════════════════════════════════
# Main: Run All Comparisons
# ═══════════════════════════════════════════════════════════════════════
def run_deep_learning():
    print("\n" + "=" * 70, flush=True)
    print("ANALYSIS 42: Deep Learning Baseline", flush=True)
    print("=" * 70, flush=True)

    rows = []

    # ── S3 (GLOBEM) ──────────────────────────────────────────────────
    print("\n--- S3 (GLOBEM) ---", flush=True)
    s3_daily, s3_feat_names = load_s3_daily(max_days=90)
    s3_daily_padded = pad_to_length(s3_daily, target_len=90)

    s3_pids = sorted(s3_daily_padded.keys())
    s3_pids_in_df = [p for p in s3_pids if p in s3["pid"].values]
    X_3d_s3 = np.array([s3_daily_padded[p] for p in s3_pids_in_df])
    s3_idx = s3.set_index("pid")

    print("  Computing MOMENT embeddings (S3)...", flush=True)
    s3_moment_embs = get_moment_embeddings(
        {p: s3_daily_padded[p] for p in s3_pids_in_df}, seq_len=90
    )

    for outcome_col, outcome_label in S3_OUTCOMES.items():
        if outcome_col not in s3.columns:
            continue

        print(f"\n  S3 — {outcome_label}:", flush=True)

        y_dl = np.array([s3_idx.loc[p, outcome_col] if p in s3_idx.index else np.nan
                         for p in s3_pids_in_df])

        # (a) Personality Ridge
        X_pers = s3[TRAITS].values
        y_all = s3[outcome_col].values
        r2_pers, n_pers, r2s_pers = quick_cv_r2(X_pers, y_all)
        print(f"    Personality Ridge:    R²={r2_pers:.4f} (N={n_pers})", flush=True)
        rows.append({"Study": "S3", "Outcome": outcome_label,
                     "Model": "Personality (Ridge)", "N": n_pers,
                     "R2_mean": r2_pers, "R2_std": np.std(r2s_pers) if r2s_pers else np.nan,
                     "R2_CI_lo": np.percentile(r2s_pers, 2.5) if r2s_pers else np.nan,
                     "R2_CI_hi": np.percentile(r2s_pers, 97.5) if r2s_pers else np.nan})

        # (b) Sensing PCA Ridge
        X_beh = s3[S3_BEH_PCA].values
        r2_pca, n_pca, r2s_pca = quick_cv_r2(X_beh, y_all)
        print(f"    Sensing PCA Ridge:   R²={r2_pca:.4f} (N={n_pca})", flush=True)
        rows.append({"Study": "S3", "Outcome": outcome_label,
                     "Model": "Sensing PCA (Ridge)", "N": n_pca,
                     "R2_mean": r2_pca, "R2_std": np.std(r2s_pca) if r2s_pca else np.nan,
                     "R2_CI_lo": np.percentile(r2s_pca, 2.5) if r2s_pca else np.nan,
                     "R2_CI_hi": np.percentile(r2s_pca, 97.5) if r2s_pca else np.nan})

        # (c) GradientBoosting + Optuna
        X_raw = s3[S3_BEH_PCA].values
        r2_gb, n_gb, r2s_gb = optuna_gb_cv(X_raw, y_all)
        print(f"    GradientBoosting+Optuna: R²={r2_gb:.4f} (N={n_gb})", flush=True)
        rows.append({"Study": "S3", "Outcome": outcome_label,
                     "Model": "GradientBoosting+Optuna", "N": n_gb,
                     "R2_mean": r2_gb, "R2_std": np.std(r2s_gb) if r2s_gb else np.nan,
                     "R2_CI_lo": np.percentile(r2s_gb, 2.5) if r2s_gb else np.nan,
                     "R2_CI_hi": np.percentile(r2s_gb, 97.5) if r2s_gb else np.nan})

        # (d) 1D-CNN
        r2_cnn, n_cnn, r2s_cnn = cnn_cv_r2(X_3d_s3.copy(), y_dl)
        print(f"    1D-CNN (MPS):        R²={r2_cnn:.4f} (N={n_cnn})", flush=True)
        rows.append({"Study": "S3", "Outcome": outcome_label,
                     "Model": "1D-CNN", "N": n_cnn,
                     "R2_mean": r2_cnn, "R2_std": np.std(r2s_cnn) if r2s_cnn else np.nan,
                     "R2_CI_lo": np.percentile(r2s_cnn, 2.5) if r2s_cnn else np.nan,
                     "R2_CI_hi": np.percentile(r2s_cnn, 97.5) if r2s_cnn else np.nan})

        # (e) MOMENT → Ridge
        r2_mom, n_mom, r2s_mom = moment_cv_r2(s3_moment_embs, s3, outcome_col)
        print(f"    MOMENT → Ridge:      R²={r2_mom:.4f} (N={n_mom})", flush=True)
        rows.append({"Study": "S3", "Outcome": outcome_label,
                     "Model": "MOMENT → Ridge", "N": n_mom,
                     "R2_mean": r2_mom, "R2_std": np.std(r2s_mom) if r2s_mom else np.nan,
                     "R2_CI_lo": np.percentile(r2s_mom, 2.5) if r2s_mom else np.nan,
                     "R2_CI_hi": np.percentile(r2s_mom, 97.5) if r2s_mom else np.nan})

    # ── S2 (NetHealth) ───────────────────────────────────────────────
    print("\n--- S2 (NetHealth) ---", flush=True)
    s2_daily, s2_feat_names = load_s2_daily(max_days=180)
    s2_daily_padded = pad_to_length(s2_daily, target_len=180)

    s2_eids = sorted(s2_daily_padded.keys())
    s2_eids_in_df = [e for e in s2_eids if e in s2["egoid"].values]

    if len(s2_eids_in_df) >= 30:
        X_3d_s2 = np.array([s2_daily_padded[e] for e in s2_eids_in_df])
        s2_idx = s2.set_index("egoid")

        print("  Computing MOMENT embeddings (S2)...", flush=True)
        s2_moment_embs = get_moment_embeddings(
            {e: s2_daily_padded[e] for e in s2_eids_in_df}, seq_len=180
        )

        for outcome_col, outcome_label in S2_OUTCOMES.items():
            if outcome_col not in s2.columns:
                continue

            print(f"\n  S2 — {outcome_label}:", flush=True)
            y_dl = np.array([s2_idx.loc[e, outcome_col] if e in s2_idx.index else np.nan
                             for e in s2_eids_in_df])

            # (a) Personality Ridge
            X_pers = s2[TRAITS].values
            y_all = s2[outcome_col].values
            r2_pers, n_pers, r2s_pers = quick_cv_r2(X_pers, y_all)
            print(f"    Personality Ridge:    R²={r2_pers:.4f} (N={n_pers})", flush=True)
            rows.append({"Study": "S2", "Outcome": outcome_label,
                         "Model": "Personality (Ridge)", "N": n_pers,
                         "R2_mean": r2_pers, "R2_std": np.std(r2s_pers) if r2s_pers else np.nan,
                         "R2_CI_lo": np.percentile(r2s_pers, 2.5) if r2s_pers else np.nan,
                         "R2_CI_hi": np.percentile(r2s_pers, 97.5) if r2s_pers else np.nan})

            # (b) Sensing PCA Ridge
            X_beh = s2[S2_BEH_PCA].values
            r2_pca, n_pca, r2s_pca = quick_cv_r2(X_beh, y_all)
            print(f"    Sensing PCA Ridge:   R²={r2_pca:.4f} (N={n_pca})", flush=True)
            rows.append({"Study": "S2", "Outcome": outcome_label,
                         "Model": "Sensing PCA (Ridge)", "N": n_pca,
                         "R2_mean": r2_pca, "R2_std": np.std(r2s_pca) if r2s_pca else np.nan,
                         "R2_CI_lo": np.percentile(r2s_pca, 2.5) if r2s_pca else np.nan,
                         "R2_CI_hi": np.percentile(r2s_pca, 97.5) if r2s_pca else np.nan})

            # (c) GradientBoosting + Optuna
            r2_gb, n_gb, r2s_gb = optuna_gb_cv(X_beh, y_all)
            print(f"    GradientBoosting+Optuna: R²={r2_gb:.4f} (N={n_gb})", flush=True)
            rows.append({"Study": "S2", "Outcome": outcome_label,
                         "Model": "GradientBoosting+Optuna", "N": n_gb,
                         "R2_mean": r2_gb, "R2_std": np.std(r2s_gb) if r2s_gb else np.nan,
                         "R2_CI_lo": np.percentile(r2s_gb, 2.5) if r2s_gb else np.nan,
                         "R2_CI_hi": np.percentile(r2s_gb, 97.5) if r2s_gb else np.nan})

            # (d) 1D-CNN
            r2_cnn, n_cnn, r2s_cnn = cnn_cv_r2(X_3d_s2.copy(), y_dl)
            print(f"    1D-CNN (MPS):        R²={r2_cnn:.4f} (N={n_cnn})", flush=True)
            rows.append({"Study": "S2", "Outcome": outcome_label,
                         "Model": "1D-CNN", "N": n_cnn,
                         "R2_mean": r2_cnn, "R2_std": np.std(r2s_cnn) if r2s_cnn else np.nan,
                         "R2_CI_lo": np.percentile(r2s_cnn, 2.5) if r2s_cnn else np.nan,
                         "R2_CI_hi": np.percentile(r2s_cnn, 97.5) if r2s_cnn else np.nan})

            # (e) MOMENT → Ridge
            if s2_moment_embs is not None:
                s2_mom_df = s2.copy()
                s2_mom_df = s2_mom_df.rename(columns={"egoid": "pid"})
                s2_moment_embs_pid = {str(k): v for k, v in s2_moment_embs.items()}
                s2_mom_df["pid"] = s2_mom_df["pid"].astype(str)
                r2_mom, n_mom, r2s_mom = moment_cv_r2(s2_moment_embs_pid, s2_mom_df, outcome_col)
            else:
                r2_mom, n_mom, r2s_mom = np.nan, 0, []
            print(f"    MOMENT → Ridge:      R²={r2_mom:.4f} (N={n_mom})", flush=True)
            rows.append({"Study": "S2", "Outcome": outcome_label,
                         "Model": "MOMENT → Ridge", "N": n_mom,
                         "R2_mean": r2_mom, "R2_std": np.std(r2s_mom) if r2s_mom else np.nan,
                         "R2_CI_lo": np.percentile(r2s_mom, 2.5) if r2s_mom else np.nan,
                         "R2_CI_hi": np.percentile(r2s_mom, 97.5) if r2s_mom else np.nan})

    # ── Save results ─────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "deep_learning_comparison.csv", index=False)
    print(f"\n  Saved: {OUT / 'deep_learning_comparison.csv'}", flush=True)

    # ── Figure ───────────────────────────────────────────────────────
    if len(df_out) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        model_order = ["Personality (Ridge)", "Sensing PCA (Ridge)",
                       "GradientBoosting+Optuna", "1D-CNN", "MOMENT → Ridge"]
        colors = {"Personality (Ridge)": "#e74c3c",
                  "Sensing PCA (Ridge)": "#95a5a6",
                  "GradientBoosting+Optuna": "#f39c12",
                  "1D-CNN": "#3498db",
                  "MOMENT → Ridge": "#9b59b6"}

        for ax, study in zip(axes, ["S3", "S2"]):
            sub = df_out[df_out["Study"] == study]
            if len(sub) == 0:
                continue

            outcomes = sub["Outcome"].unique()
            x = np.arange(len(outcomes))
            width = 0.15

            for i, model_name in enumerate(model_order):
                vals = []
                for outcome in outcomes:
                    row = sub[(sub["Outcome"] == outcome) & (sub["Model"] == model_name)]
                    vals.append(row["R2_mean"].values[0] if len(row) > 0 else 0)
                ax.bar(x + i * width, vals, width, label=model_name,
                       color=colors.get(model_name, "#333"), edgecolor="white")

            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(outcomes, fontsize=9)
            ax.set_ylabel("R² (Cross-Validated)")
            ax.set_title(f"Study: {study}", fontweight="bold")
            ax.axhline(0, color="grey", linestyle="--", alpha=0.5)

        axes[0].legend(fontsize=8, loc="upper right")
        plt.suptitle("Deep Learning Cannot Rescue Passive Sensing",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        fig.savefig(OUT / "figure_deep_learning.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUT / 'figure_deep_learning.png'}", flush=True)

    return df_out


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run_deep_learning()
    print("\n✓ Analysis 42 complete.", flush=True)
    print(result.to_string(index=False), flush=True)
