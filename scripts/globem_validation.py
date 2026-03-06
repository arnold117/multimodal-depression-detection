#!/usr/bin/env python3
"""
Phase 13 Step 4: Core Validation Analyses on GLOBEM Dataset

Study 3 validation — focused on mental health (no GPA in GLOBEM):
  1. Personality → Mental Health (5 outcomes × 4 ML models, 10-fold CV)
  2. SHAP feature importance — Is Neuroticism #1 for mental health?
  3. Behavior → Mental Health (3 feature sets × 4 models × 5 outcomes)
  4. LPA behavioral profiles → mental health differences
  5. Cohort sensitivity: exclude INS-W_3 (COVID)

Input:  data/processed/globem/globem_analysis_dataset.parquet
Output: results/globem/tables/  (validation result tables)
        results/globem/figures/ (SHAP plots)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from scipy import stats as sp_stats
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from multi_outcome_prediction import _tune_params, _make_model
from ml_interpretability import run_shap_analysis, run_cross_model_importance, fit_model

GB_DATA = project_root / 'data' / 'processed' / 'globem' / 'globem_analysis_dataset.parquet'
GB_TABLE_DIR = project_root / 'results' / 'globem' / 'tables'
GB_FIGURE_DIR = project_root / 'results' / 'globem' / 'figures'

for d in [GB_TABLE_DIR, GB_FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 10})

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
GB_BEHAVIOR_PC = ['activity_pc1', 'sleep_pc1', 'communication_pc1',
                  'digital_pc1', 'mobility_pc1']
MODELS = ['elastic_net', 'ridge', 'random_forest', 'svr']
MODEL_LABELS = {'elastic_net': 'Elastic Net', 'ridge': 'Ridge',
                'random_forest': 'Random Forest', 'svr': 'SVR'}

MH_OUTCOMES = [
    ('bdi2_total', 'BDI-II'),
    ('stai_state', 'STAI'),
    ('pss_10', 'PSS-10'),
    ('cesd_total', 'CESD'),
    ('ucla_loneliness', 'UCLA'),
]

RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────
# K-Fold CV utilities (same as NetHealth)
# ──────────────────────────────────────────────────────────────────────

def kfold_predict(X, y, model_name='elastic_net', n_splits=10, n_repeats=10):
    """Repeated k-fold CV. Returns mean R² across all repeats."""
    best_params = _tune_params(model_name, X, y)

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                         random_state=RANDOM_STATE)
    r2_scores = []

    for train_idx, test_idx in rkf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = _make_model(model_name, len(train_idx), best_params)
        model.fit(X_train, y[train_idx])
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y[test_idx], y_pred))

    return {
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores),
        'R2_ci_lo': np.percentile(r2_scores, 2.5),
        'R2_ci_hi': np.percentile(r2_scores, 97.5),
    }


def _single_perm_kfold(args):
    X, y_perm, model_name, best_params, kf_splits = args
    fold_r2 = []
    for train_idx, test_idx in kf_splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = _make_model(model_name, len(train_idx), best_params)
        model.fit(X_train, y_perm[train_idx])
        y_pred = model.predict(X_test)
        fold_r2.append(r2_score(y_perm[test_idx], y_pred))
    return np.mean(fold_r2)


def kfold_permutation_test(X, y, model_name, observed_r2, n_perm=200):
    """Permutation test using 10-fold CV with joblib parallelization."""
    from joblib import Parallel, delayed

    best_params = _tune_params(model_name, X, y)
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    kf_splits = list(kf.split(X))

    rng = np.random.RandomState(RANDOM_STATE)
    perms = [rng.permutation(y) for _ in range(n_perm)]

    perm_r2 = Parallel(n_jobs=-1)(
        delayed(_single_perm_kfold)((X, y_perm, model_name, best_params, kf_splits))
        for y_perm in perms
    )

    p_val = (np.sum(np.array(perm_r2) >= observed_r2) + 1) / (n_perm + 1)
    return p_val


# ──────────────────────────────────────────────────────────────────────
# 1. Personality → Mental Health (all 5 outcomes)
# ──────────────────────────────────────────────────────────────────────

def run_personality_mental_health(df):
    """4 models × Personality → each mental health outcome."""
    print("\n[1/5] Personality → Mental Health")
    print("─" * 60)

    all_rows = []
    for outcome_col, outcome_label in MH_OUTCOMES:
        subset = df[PERSONALITY + [outcome_col]].dropna()
        X = subset[PERSONALITY].values
        y = subset[outcome_col].values
        n = len(subset)
        print(f"\n  {outcome_label} (N={n}):")

        for model_name in MODELS:
            label = MODEL_LABELS[model_name]
            kf_result = kfold_predict(X, y, model_name)
            p_val = kfold_permutation_test(X, y, model_name, kf_result['R2_mean'])
            print(f"    {label:15s}: R²={kf_result['R2_mean']:.3f} "
                  f"[{kf_result['R2_ci_lo']:.3f}, {kf_result['R2_ci_hi']:.3f}] p={p_val:.4f}")

            all_rows.append({
                'Outcome': outcome_label,
                'Feature_Set': 'Personality',
                'Model': label,
                'N': n,
                'R2_mean': kf_result['R2_mean'],
                'R2_std': kf_result['R2_std'],
                'R2_ci_lo': kf_result['R2_ci_lo'],
                'R2_ci_hi': kf_result['R2_ci_hi'],
                'p_perm': p_val,
            })

    results = pd.DataFrame(all_rows)
    results.to_csv(GB_TABLE_DIR / 'personality_mental_health.csv', index=False)
    print(f"\n  Saved: personality_mental_health.csv")
    return results


# ──────────────────────────────────────────────────────────────────────
# 2. SHAP for Mental Health (is Neuroticism #1?)
# ──────────────────────────────────────────────────────────────────────

def run_shap_mental_health(df, outcome_col, outcome_label):
    """SHAP analysis for Personality → one mental health outcome."""
    print(f"\n  SHAP: Personality → {outcome_label}")
    print("  " + "─" * 56)

    subset = df[PERSONALITY + [outcome_col]].dropna()
    X = subset[PERSONALITY].values
    y = subset[outcome_col].values
    feature_names = PERSONALITY
    n = len(subset)

    try:
        import shap

        all_shap = {}
        for model_name in MODELS:
            label = MODEL_LABELS[model_name]
            print(f"    {label}...")

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = fit_model(model_name, X_s, y)

            if model_name == 'random_forest':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_s)
            else:
                explainer = shap.KernelExplainer(model.predict, X_s[:50])
                shap_values = explainer.shap_values(X_s)

            importance = np.abs(shap_values).mean(axis=0)
            all_shap[label] = dict(zip(feature_names, importance))

            top_feat = feature_names[np.argmax(importance)]
            print(f"      Top feature: {top_feat}")

        safe_name = outcome_label.lower().replace('-', '').replace(' ', '_')

        shap_df = pd.DataFrame(all_shap).T
        shap_df.index.name = 'Model'
        shap_df.to_csv(GB_TABLE_DIR / f'shap_personality_{safe_name}.csv')

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_s, feature_names=feature_names,
                         show=False, plot_size=None)
        plt.title(f'SHAP: Personality → {outcome_label} (GLOBEM, N={n})')
        plt.tight_layout()
        fig.savefig(GB_FIGURE_DIR / f'shap_personality_{safe_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"    Saved: shap_personality_{safe_name}.csv/.png")

    except Exception as e:
        print(f"    SHAP failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# 3. Behavior → Mental Health (3 feature sets × 4 models × 5 outcomes)
# ──────────────────────────────────────────────────────────────────────

def run_behavior_outcome(df, outcome_col='bdi2_total', outcome_label='BDI-II'):
    """Test behavior features → mental health outcome."""
    print(f"\n  Behavior → {outcome_label}")
    print("  " + "─" * 56)

    feature_sets = {
        'Personality': PERSONALITY,
        'Behavior': GB_BEHAVIOR_PC,
        'Pers+Beh': PERSONALITY + GB_BEHAVIOR_PC,
    }

    rows = []
    for fs_name, fs_cols in feature_sets.items():
        available = [c for c in fs_cols if c in df.columns]
        subset = df[available + [outcome_col]].dropna()
        X = subset[available].values
        y = subset[outcome_col].values
        n = len(subset)

        if n < 30:
            print(f"    {fs_name}: skipped (N={n} < 30)")
            continue

        for model_name in MODELS:
            label = MODEL_LABELS[model_name]
            kf_result = kfold_predict(X, y, model_name)
            p_val = kfold_permutation_test(X, y, model_name, kf_result['R2_mean'])
            sig = '*' if p_val < 0.05 else ''
            print(f"    {fs_name:10s} × {label:15s}: R²={kf_result['R2_mean']:.3f} p={p_val:.4f}{sig}")

            rows.append({
                'Outcome': outcome_label,
                'Feature_Set': fs_name,
                'Model': label,
                'N': n,
                'R2_mean': kf_result['R2_mean'],
                'R2_std': kf_result['R2_std'],
                'R2_ci_lo': kf_result['R2_ci_lo'],
                'R2_ci_hi': kf_result['R2_ci_hi'],
                'p_perm': p_val,
            })

    results = pd.DataFrame(rows)
    safe_name = outcome_label.lower().replace('-', '').replace(' ', '_')
    results.to_csv(GB_TABLE_DIR / f'behavior_{safe_name}.csv', index=False)
    return results


# ──────────────────────────────────────────────────────────────────────
# 4. LPA: Behavioral Profiles → Mental Health
# ──────────────────────────────────────────────────────────────────────

def run_lpa(df):
    """GMM-based latent profile analysis on behavior features."""
    print("\n[4/5] LPA: Behavioral Profiles → Mental Health")
    print("─" * 60)

    # Cluster on behavior features only
    available = [c for c in GB_BEHAVIOR_PC if c in df.columns]
    beh_subset = df[['pid'] + available].dropna()
    X_beh = beh_subset[available].values
    n = len(beh_subset)
    print(f"  Behavior features available: {available}")
    print(f"  N with complete behavior data: {n}")

    if n < 50:
        print("  Too few participants, skipping LPA")
        return

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_beh)

    # BIC model selection
    bic_scores = {}
    for k in range(2, 7):
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                              n_init=10, covariance_type='full')
        gmm.fit(X_s)
        bic_scores[k] = gmm.bic(X_s)
        print(f"  k={k}: BIC={bic_scores[k]:.1f}")

    best_k = min(bic_scores, key=bic_scores.get)
    print(f"  → Best k = {best_k}")

    # Fit best model and assign profiles
    gmm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE,
                          n_init=10, covariance_type='full')
    gmm.fit(X_s)
    beh_subset = beh_subset.copy()
    beh_subset['profile'] = gmm.predict(X_s) + 1

    # Merge profiles back to original df
    df_with_profiles = df.merge(beh_subset[['pid', 'profile']], on='pid', how='left')

    # Test each mental health outcome independently
    rows = []
    for outcome_col, outcome_label in MH_OUTCOMES:
        test_df = df_with_profiles[['profile', outcome_col]].dropna()
        if len(test_df) < 30:
            continue

        groups = [g[outcome_col].values for _, g in test_df.groupby('profile')]
        if len(groups) < 2:
            continue

        F_stat, p_val = sp_stats.f_oneway(*groups)
        eta2 = F_stat * (best_k - 1) / (F_stat * (best_k - 1) + len(test_df) - best_k)
        sig = '*' if p_val < 0.05 else ''
        print(f"  {outcome_label:10s}: F={F_stat:.2f}, p={p_val:.4f}{sig}, η²={eta2:.3f}, N={len(test_df)}")

        rows.append({
            'Outcome': outcome_label,
            'N': len(test_df),
            'k': best_k,
            'F': F_stat,
            'p': p_val,
            'eta_sq': eta2,
        })

    if rows:
        lpa_df = pd.DataFrame(rows)
        lpa_df.to_csv(GB_TABLE_DIR / 'lpa_outcomes.csv', index=False)
        print(f"  Saved: lpa_outcomes.csv")


# ──────────────────────────────────────────────────────────────────────
# 5. Cohort Sensitivity: Exclude COVID (INS-W_3)
# ──────────────────────────────────────────────────────────────────────

def run_covid_sensitivity(df, full_results):
    """Re-run personality → MH excluding Cohort 3 (COVID). Compare R²."""
    print("\n[5/5] COVID Sensitivity: Exclude INS-W_3")
    print("─" * 60)

    df_no_covid = df[df['cohort'] != 3].copy()
    n_full = len(df)
    n_no_covid = len(df_no_covid)
    print(f"  Full sample: {n_full}, excluding COVID: {n_no_covid}")

    sensitivity_rows = []
    for outcome_col, outcome_label in MH_OUTCOMES:
        subset = df_no_covid[PERSONALITY + [outcome_col]].dropna()
        X = subset[PERSONALITY].values
        y = subset[outcome_col].values
        n = len(subset)

        if n < 30:
            continue

        # Use just one model (Ridge) for efficiency
        kf_result = kfold_predict(X, y, 'ridge')
        print(f"  {outcome_label:10s}: R²={kf_result['R2_mean']:.3f} (N={n})")

        # Compare with full-sample Ridge result
        full_r2 = full_results[
            (full_results['Outcome'] == outcome_label) &
            (full_results['Model'] == 'Ridge')
        ]['R2_mean'].values
        if len(full_r2) > 0:
            delta = kf_result['R2_mean'] - full_r2[0]
            print(f"    Delta from full sample: {delta:+.3f}")

        sensitivity_rows.append({
            'Outcome': outcome_label,
            'Full_R2': full_r2[0] if len(full_r2) > 0 else np.nan,
            'NoCovid_R2': kf_result['R2_mean'],
            'Delta': delta if len(full_r2) > 0 else np.nan,
            'N_full': len(df[PERSONALITY + [outcome_col]].dropna()),
            'N_no_covid': n,
        })

    if sensitivity_rows:
        sens_df = pd.DataFrame(sensitivity_rows)
        sens_df.to_csv(GB_TABLE_DIR / 'covid_sensitivity.csv', index=False)
        print(f"\n  Saved: covid_sensitivity.csv")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 13 Step 4: GLOBEM Validation (Study 3)")
    print("=" * 60)

    df = pd.read_parquet(GB_DATA)
    print(f"Dataset: {df.shape[0]} participants × {df.shape[1]} columns")

    # Step 1: Personality → Mental Health
    pers_mh_results = run_personality_mental_health(df)

    # Step 2: SHAP for each mental health outcome
    print("\n[2/5] SHAP: Personality → Mental Health")
    print("─" * 60)
    for outcome_col, outcome_label in MH_OUTCOMES:
        run_shap_mental_health(df, outcome_col, outcome_label)

    # Step 3: Behavior → Mental Health (all outcomes)
    print("\n[3/5] Behavior → Mental Health (all outcomes)")
    print("─" * 60)
    all_beh_results = []
    for outcome_col, outcome_label in MH_OUTCOMES:
        result = run_behavior_outcome(df, outcome_col, outcome_label)
        all_beh_results.append(result)
    combined_beh = pd.concat(all_beh_results, ignore_index=True)
    combined_beh.to_csv(GB_TABLE_DIR / 'behavior_mental_health_all.csv', index=False)
    print(f"\n  Saved: behavior_mental_health_all.csv ({len(combined_beh)} rows)")

    # Step 4: LPA
    run_lpa(df)

    # Step 5: COVID sensitivity
    run_covid_sensitivity(df, pers_mh_results)

    print("\n" + "=" * 60)
    print("GLOBEM Validation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
