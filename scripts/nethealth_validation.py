#!/usr/bin/env python3
"""
Phase 12 Step 4: Core Validation Analyses on NetHealth Dataset

Replicates key findings from Study 1 (StudentLife) on Study 2 (NetHealth):
  1. Personality → GPA (4 ML models, 10-fold CV + LOO-CV, permutation test)
  2. SHAP feature importance (Conscientiousness = #1?)
  3. Cross-model consistency (Kendall's τ)
  4. Behavior → Depression (CES-D)
  5. LPA behavioral profiles
  6. Moderation: Personality × Behavior → GPA

Input:  data/processed/nethealth/nethealth_analysis_dataset.parquet
Output: results/nethealth/tables/  (validation result tables)
        results/nethealth/figures/ (SHAP, heatmaps)
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
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score
from scipy import stats as sp_stats
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

# Import reusable functions from existing scripts
from multi_outcome_prediction import (
    _tune_params, _make_model
)
from ml_interpretability import (
    run_shap_analysis, run_cross_model_importance, fit_model
)

NH_DATA = project_root / 'data' / 'processed' / 'nethealth' / 'nethealth_analysis_dataset.parquet'
NH_TABLE_DIR = project_root / 'results' / 'nethealth' / 'tables'
NH_FIGURE_DIR = project_root / 'results' / 'nethealth' / 'figures'

for d in [NH_TABLE_DIR, NH_FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.size': 10})

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
NH_BEHAVIOR_PC = ['nh_activity_pc1', 'nh_sleep_pc1', 'nh_communication_pc1']
MODELS = ['elastic_net', 'ridge', 'random_forest', 'svr']
MODEL_LABELS = {'elastic_net': 'Elastic Net', 'ridge': 'Ridge',
                'random_forest': 'Random Forest', 'svr': 'SVR'}

RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────
# K-Fold CV (for N=220+, more efficient than LOO)
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
    """Single permutation iteration for joblib parallelization."""
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
    from sklearn.model_selection import KFold

    best_params = _tune_params(model_name, X, y)
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    kf_splits = list(kf.split(X))  # materialize once

    rng = np.random.RandomState(RANDOM_STATE)
    perms = [rng.permutation(y) for _ in range(n_perm)]

    perm_r2 = Parallel(n_jobs=-1)(
        delayed(_single_perm_kfold)((X, y_perm, model_name, best_params, kf_splits))
        for y_perm in perms
    )

    p_val = (np.sum(np.array(perm_r2) >= observed_r2) + 1) / (n_perm + 1)
    return p_val


# ──────────────────────────────────────────────────────────────────────
# 1. Personality → GPA (core validation)
# ──────────────────────────────────────────────────────────────────────

def run_personality_gpa(df):
    """Core validation: 4 models × Personality → GPA."""
    print("\n[1/6] Personality → GPA (Core Validation)")
    print("─" * 60)

    subset = df[PERSONALITY + ['gpa_overall']].dropna()
    X = subset[PERSONALITY].values
    y = subset['gpa_overall'].values
    n = len(subset)
    print(f"  N = {n}")

    rows = []
    for model_name in MODELS:
        label = MODEL_LABELS[model_name]
        print(f"\n  {label}...")

        # 10×10-fold CV (primary)
        kf_result = kfold_predict(X, y, model_name)
        print(f"    10×10-fold: R² = {kf_result['R2_mean']:.3f} "
              f"(95% CI: [{kf_result['R2_ci_lo']:.3f}, {kf_result['R2_ci_hi']:.3f}])")

        # Permutation test (10-fold, joblib parallel)
        p_val = kfold_permutation_test(X, y, model_name, kf_result['R2_mean'])
        print(f"    Permutation: p = {p_val:.4f}")

        rows.append({
            'Model': label,
            'N': n,
            'R2_kfold': kf_result['R2_mean'],
            'R2_kfold_std': kf_result['R2_std'],
            'R2_kfold_ci_lo': kf_result['R2_ci_lo'],
            'R2_kfold_ci_hi': kf_result['R2_ci_hi'],
            'p_perm': p_val,
        })

    results = pd.DataFrame(rows)
    results.to_csv(NH_TABLE_DIR / 'personality_gpa_validation.csv', index=False)
    print(f"\n  Saved: personality_gpa_validation.csv")
    return results


# ──────────────────────────────────────────────────────────────────────
# 2. SHAP Feature Importance
# ──────────────────────────────────────────────────────────────────────

def run_shap(df):
    """SHAP analysis for Personality → GPA. Is Conscientiousness still #1?"""
    print("\n[2/6] SHAP Feature Importance")
    print("─" * 60)

    subset = df[PERSONALITY + ['gpa_overall']].dropna()
    X = subset[PERSONALITY].values
    y = subset['gpa_overall'].values
    feature_names = PERSONALITY
    n = len(subset)

    try:
        import shap

        all_shap = {}
        for model_name in MODELS:
            label = MODEL_LABELS[model_name]
            print(f"  {label}...")

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

            rank = np.argsort(-importance) + 1
            top_feat = feature_names[np.argmax(importance)]
            print(f"    Top feature: {top_feat} (rank: {rank})")

        # Save SHAP importance table
        shap_df = pd.DataFrame(all_shap).T
        shap_df.index.name = 'Model'
        shap_df.to_csv(NH_TABLE_DIR / 'shap_personality_gpa.csv')

        # Plot SHAP summary (using last model's values for beeswarm)
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_s, feature_names=feature_names,
                         show=False, plot_size=None)
        plt.title(f'SHAP: Personality → GPA (NetHealth, N={n})')
        plt.tight_layout()
        fig.savefig(NH_FIGURE_DIR / 'shap_personality_gpa.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: shap_personality_gpa.png, shap_personality_gpa.csv")

    except Exception as e:
        print(f"  SHAP failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# 3. Cross-Model Consistency
# ──────────────────────────────────────────────────────────────────────

def run_cross_model(df):
    """Cross-model feature importance consistency (Kendall's τ)."""
    print("\n[3/6] Cross-Model Consistency")
    print("─" * 60)

    subset = df[PERSONALITY + ['gpa_overall']].dropna()
    X = subset[PERSONALITY].values
    y = subset['gpa_overall'].values

    try:
        importance_dict = run_cross_model_importance(
            X, y, PERSONALITY,
            save_dir=NH_TABLE_DIR, fig_dir=NH_FIGURE_DIR,
            title_suffix='(NetHealth)'
        )
        print("  Saved: cross_model_importance.csv")
    except Exception as e:
        print(f"  Cross-model failed: {e}")
        # Fallback: manual computation
        from sklearn.inspection import permutation_importance as perm_imp
        from scipy.stats import rankdata, kendalltau

        rankings = {}
        for model_name in MODELS:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = fit_model(model_name, X_s, y)

            if model_name in ['elastic_net', 'ridge']:
                imp = np.abs(model.coef_)
            elif model_name == 'random_forest':
                imp = model.feature_importances_
            else:
                result = perm_imp(model, X_s, y, n_repeats=30, random_state=42)
                imp = result.importances_mean

            rankings[MODEL_LABELS[model_name]] = rankdata(-imp)

        # Compute pairwise Kendall's τ
        from itertools import combinations
        taus = []
        for m1, m2 in combinations(rankings.keys(), 2):
            tau, p = kendalltau(rankings[m1], rankings[m2])
            taus.append({'Model_1': m1, 'Model_2': m2, 'tau': tau, 'p': p})
            print(f"  τ({m1}, {m2}) = {tau:.3f} (p={p:.3f})")

        mean_tau = np.mean([t['tau'] for t in taus])
        print(f"  Mean τ = {mean_tau:.3f}")

        pd.DataFrame(taus).to_csv(NH_TABLE_DIR / 'cross_model_kendall.csv', index=False)


# ──────────────────────────────────────────────────────────────────────
# 4. Behavior → Depression (CES-D)
# ──────────────────────────────────────────────────────────────────────

def run_behavior_depression(df):
    """Test Fitbit behavior → CES-D depression."""
    print("\n[4/6] Behavior → Depression (CES-D)")
    print("─" * 60)

    feature_sets = {
        'Personality': PERSONALITY,
        'Behavior': NH_BEHAVIOR_PC,
        'Pers + Beh': PERSONALITY + NH_BEHAVIOR_PC,
    }

    rows = []
    for fset_name, fset_cols in feature_sets.items():
        available = [c for c in fset_cols if c in df.columns]
        subset = df[available + ['cesd_total']].dropna()
        if len(subset) < 20:
            print(f"  {fset_name}: insufficient data (N={len(subset)})")
            continue

        X = subset[available].values
        y = subset['cesd_total'].values

        for model_name in MODELS:
            result = kfold_predict(X, y, model_name)

            rows.append({
                'Features': fset_name,
                'Model': MODEL_LABELS[model_name],
                'N': len(subset),
                'R2_kfold': result['R2_mean'],
                'R2_kfold_std': result['R2_std'],
                'R2_kfold_ci_lo': result['R2_ci_lo'],
                'R2_kfold_ci_hi': result['R2_ci_hi'],
            })

        print(f"  {fset_name} (N={len(subset)}): done")

    results = pd.DataFrame(rows)
    results.to_csv(NH_TABLE_DIR / 'behavior_depression.csv', index=False)

    # Print best result
    if len(results) > 0:
        best = results.loc[results['R2_kfold'].idxmax()]
        print(f"\n  Best: {best['Features']} × {best['Model']} → "
              f"R²={best['R2_kfold']:.3f} (kfold)")
    print(f"  Saved: behavior_depression.csv")
    return results


# ──────────────────────────────────────────────────────────────────────
# 5. LPA (Latent Profile Analysis)
# ──────────────────────────────────────────────────────────────────────

def run_lpa(df):
    """LPA on behavioral composites."""
    print("\n[5/6] Latent Profile Analysis")
    print("─" * 60)

    available = [c for c in NH_BEHAVIOR_PC if c in df.columns]
    subset = df[available + ['gpa_overall', 'cesd_total', 'loneliness_total']].dropna()

    if len(subset) < 30:
        print(f"  Insufficient data (N={len(subset)})")
        return

    X_beh = subset[available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_beh)

    from sklearn.mixture import GaussianMixture

    # Test k=2..6
    bic_scores = {}
    for k in range(2, 7):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               n_init=10, random_state=RANDOM_STATE)
        gmm.fit(X_scaled)
        bic_scores[k] = gmm.bic(X_scaled)
        print(f"  k={k}: BIC={bic_scores[k]:.1f}")

    best_k = min(bic_scores, key=bic_scores.get)
    print(f"  Best k={best_k}")

    # Fit best model
    gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                           n_init=10, random_state=RANDOM_STATE)
    labels = gmm.fit_predict(X_scaled)
    subset = subset.copy()
    subset['profile'] = labels

    # Compare outcomes across profiles
    print(f"\n  Profile outcomes (k={best_k}):")
    lpa_rows = []
    for outcome in ['gpa_overall', 'cesd_total', 'loneliness_total']:
        if outcome not in subset.columns:
            continue
        groups = [g[outcome].dropna().values for _, g in subset.groupby('profile')]
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            f_stat, p_val = sp_stats.f_oneway(*groups)
            print(f"    {outcome:20s}  F={f_stat:.2f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
            lpa_rows.append({'Outcome': outcome, 'F': f_stat, 'p': p_val, 'k': best_k, 'N': len(subset)})

    if lpa_rows:
        pd.DataFrame(lpa_rows).to_csv(NH_TABLE_DIR / 'lpa_outcomes.csv', index=False)
        print(f"  Saved: lpa_outcomes.csv")


# ──────────────────────────────────────────────────────────────────────
# 6. Moderation: Personality × Behavior → GPA
# ──────────────────────────────────────────────────────────────────────

def run_moderation(df):
    """Test if personality moderates behavior → GPA."""
    print("\n[6/6] Moderation: Personality × Behavior → GPA")
    print("─" * 60)

    behavior_cols = [c for c in NH_BEHAVIOR_PC if c in df.columns]
    subset = df[PERSONALITY + behavior_cols + ['gpa_overall']].dropna()
    n = len(subset)
    print(f"  N = {n}")

    if n < 30:
        print("  Insufficient data for moderation analysis")
        return

    from sklearn.linear_model import LinearRegression

    rows = []
    for trait in PERSONALITY:
        for beh in behavior_cols:
            y = subset['gpa_overall'].values

            # Step 1: Main effects
            X1 = subset[[trait, beh]].values
            scaler1 = StandardScaler()
            X1_s = scaler1.fit_transform(X1)
            m1 = LinearRegression().fit(X1_s, y)
            r2_1 = m1.score(X1_s, y)

            # Step 2: Add interaction
            interaction = (subset[trait] * subset[beh]).values.reshape(-1, 1)
            X2 = np.column_stack([X1, interaction])
            scaler2 = StandardScaler()
            X2_s = scaler2.fit_transform(X2)
            m2 = LinearRegression().fit(X2_s, y)
            r2_2 = m2.score(X2_s, y)

            delta_r2 = r2_2 - r2_1

            # F-test for ΔR²
            df1 = 1  # one interaction term added
            df2 = n - X2.shape[1] - 1
            if df2 > 0 and (1 - r2_2) > 0:
                f_stat = (delta_r2 / df1) / ((1 - r2_2) / df2)
                p_val = 1 - sp_stats.f.cdf(f_stat, df1, df2)
            else:
                f_stat, p_val = np.nan, np.nan

            rows.append({
                'Personality': trait,
                'Behavior': beh,
                'R2_main': r2_1,
                'R2_interaction': r2_2,
                'Delta_R2': delta_r2,
                'F': f_stat,
                'p': p_val,
                'N': n,
                'interaction_beta': m2.coef_[-1],
            })

    results = pd.DataFrame(rows)
    results.to_csv(NH_TABLE_DIR / 'moderation_results.csv', index=False)

    # Print significant results
    sig = results[results['p'] < 0.05]
    if len(sig) > 0:
        print(f"\n  Significant interactions ({len(sig)}):")
        for _, row in sig.iterrows():
            print(f"    {row['Personality']} × {row['Behavior']} → GPA: "
                  f"ΔR²={row['Delta_R2']:.3f}, F={row['F']:.2f}, p={row['p']:.4f}")
    else:
        print("  No significant interactions (p < .05)")

    print(f"  Saved: moderation_results.csv")
    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 12 STEP 4: NETHEALTH VALIDATION")
    print("=" * 60)

    df = pd.read_parquet(NH_DATA)
    print(f"  Loaded: {df.shape[0]} participants × {df.shape[1]} columns")

    # 1. Core: Personality → GPA
    gpa_results = run_personality_gpa(df)

    # 2. SHAP
    run_shap(df)

    # 3. Cross-model consistency
    run_cross_model(df)

    # 4. Behavior → Depression
    dep_results = run_behavior_depression(df)

    # 5. LPA
    run_lpa(df)

    # 6. Moderation
    run_moderation(df)

    print(f"\n{'=' * 60}")
    print("PHASE 12 STEP 4 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
