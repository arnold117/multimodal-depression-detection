#!/usr/bin/env python3
"""
Phase 11: ML Interpretability & Cross-Model Feature Importance

Research Questions:
  1. Which features drive predictions across different ML models? (SHAP)
  2. Do models agree on feature importance? (Cross-model consistency)
  3. Does SVR capture non-linear behavior→GPA relationships missed by linear models?

Method:
  - SHAP analysis for 4 models on key prediction scenarios
  - Cross-model feature importance comparison (standardized rankings)
  - Kendall's τ for inter-model agreement
  - Non-linearity visualization (SHAP dependence plots)

Input:  data/processed/analysis_dataset.parquet
Output: results/tables/shap_importance.csv
        results/tables/cross_model_importance.csv
        results/figures/shap_summary.png
        results/figures/cross_model_importance.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from scipy import stats as sp_stats
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

DATA_PATH = project_root / 'data' / 'processed' / 'analysis_dataset.parquet'
TABLE_DIR = project_root / 'results' / 'tables'
FIGURE_DIR = project_root / 'results' / 'figures'

for d in [TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
})

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
BEHAVIOR_PC = ['mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
               'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']
RANDOM_STATE = 42

# Readable feature names for plots
FEATURE_LABELS = {
    'extraversion': 'Extraversion',
    'agreeableness': 'Agreeableness',
    'conscientiousness': 'Conscientiousness',
    'neuroticism': 'Neuroticism',
    'openness': 'Openness',
    'mobility_pc1': 'Mobility',
    'digital_pc1': 'Digital Usage',
    'social_pc1': 'Social Activity',
    'activity_pc1': 'Physical Activity',
    'screen_pc1': 'Screen Time',
    'proximity_pc1': 'Proximity',
    'face2face_pc1': 'Face-to-Face',
    'audio_pc1': 'Audio Environment',
}


def get_feature_label(f):
    return FEATURE_LABELS.get(f, f)


# ── Model fitting helpers ─────────────────────────────────────

def _tune_and_fit(model_name, X_train, y_train):
    """Tune hyperparameters via Optuna and fit on training data."""
    cv_folds = min(5, len(y_train))

    if model_name == 'elastic_net':
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            n_alphas=30, cv=cv_folds,
            random_state=RANDOM_STATE, max_iter=5000)
        model.fit(X_train, y_train)
        return model

    elif model_name == 'ridge':
        model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=cv_folds)
        model.fit(X_train, y_train)
        return model

    elif model_name == 'random_forest':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200]),
                'max_depth': trial.suggest_int('max_depth', 2, 6),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 5),
            }
            m = RandomForestRegressor(random_state=RANDOM_STATE, **params)
            return cross_val_score(m, X_train, y_train, cv=cv_folds, scoring='r2').mean()

        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        model = RandomForestRegressor(random_state=RANDOM_STATE, **study.best_params)
        model.fit(X_train, y_train)
        return model

    elif model_name == 'svr':
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'epsilon': trial.suggest_float('epsilon', 0.01, 0.3),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            }
            m = SVR(kernel='rbf', **params)
            return cross_val_score(m, X_train, y_train, cv=cv_folds, scoring='r2').mean()

        study = optuna.create_study(direction='maximize',
                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        model = SVR(kernel='rbf', **study.best_params)
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def fit_model(model_name, X_train, y_train):
    """Fit a model with Optuna-tuned hyperparameters."""
    return _tune_and_fit(model_name, X_train, y_train)


def get_fitted_estimator(model):
    """Get the actual estimator (identity since we no longer use GridSearchCV)."""
    return model


# ── 1. SHAP Analysis ─────────────────────────────────────────

def run_shap_analysis(df):
    """Run SHAP analysis for key prediction scenarios."""
    print("\n[1/3] SHAP Analysis")
    print("─" * 60)

    scenarios = [
        ('Personality → GPA', PERSONALITY, 'gpa_overall'),
        ('Behavior → PHQ-9', BEHAVIOR_PC, 'phq9_total'),
        ('Pers + Beh → GPA', PERSONALITY + BEHAVIOR_PC, 'gpa_overall'),
    ]
    model_names = ['elastic_net', 'ridge', 'random_forest', 'svr']
    model_labels = {
        'elastic_net': 'Elastic Net',
        'ridge': 'Ridge',
        'random_forest': 'Random Forest',
        'svr': 'SVR',
    }

    all_shap_results = {}

    for scenario_name, feature_cols, outcome_col in scenarios:
        print(f"\n  Scenario: {scenario_name}")
        available = [c for c in feature_cols if c in df.columns]
        subset = df[available + [outcome_col]].dropna()
        X = subset[available].values
        y = subset[outcome_col].values
        feature_names = [get_feature_label(f) for f in available]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name in model_names:
            key = f"{scenario_name}|{model_labels[model_name]}"
            print(f"    {model_labels[model_name]}...", end=' ')

            model = fit_model(model_name, X_scaled, y)
            estimator = get_fitted_estimator(model)

            # Choose appropriate SHAP explainer
            if model_name == 'random_forest':
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_scaled)
            else:
                # KernelExplainer for linear models and SVR
                # Use kmeans summarization for background data
                background = shap.kmeans(X_scaled, min(10, len(X_scaled)))
                explainer = shap.KernelExplainer(estimator.predict, background)
                shap_values = explainer.shap_values(X_scaled, nsamples=100)

            all_shap_results[key] = {
                'shap_values': shap_values,
                'X': X_scaled,
                'feature_names': feature_names,
                'raw_feature_names': available,
                'model': estimator,
                'scenario': scenario_name,
                'model_label': model_labels[model_name],
            }

            # Mean absolute SHAP
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:3]
            top_str = ', '.join(
                f"{feature_names[i]}={mean_abs[i]:.3f}" for i in top_idx)
            print(f"top: {top_str}")

    # Save SHAP importance table
    rows = []
    for key, result in all_shap_results.items():
        scenario, model_label = key.split('|')
        mean_abs = np.abs(result['shap_values']).mean(axis=0)
        for i, fname in enumerate(result['feature_names']):
            rows.append({
                'Scenario': scenario,
                'Model': model_label,
                'Feature': fname,
                'Mean_Abs_SHAP': mean_abs[i],
            })
    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(TABLE_DIR / 'shap_importance.csv', index=False)
    print(f"\n  Saved: shap_importance.csv ({len(shap_df)} rows)")

    return all_shap_results


# ── 2. Cross-model feature importance ────────────────────────

def run_cross_model_importance(df):
    """Compare feature importance across 4 models for GPA prediction."""
    print("\n[2/3] Cross-Model Feature Importance")
    print("─" * 60)

    feature_cols = PERSONALITY + BEHAVIOR_PC
    available = [c for c in feature_cols if c in df.columns]
    subset = df[available + ['gpa_overall']].dropna()
    X = subset[available].values
    y = subset['gpa_overall'].values
    feature_names = [get_feature_label(f) for f in available]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_names = ['elastic_net', 'ridge', 'random_forest', 'svr']
    model_labels = {
        'elastic_net': 'Elastic Net',
        'ridge': 'Ridge',
        'random_forest': 'Random Forest',
        'svr': 'SVR',
    }

    importance_dict = {}

    for model_name in model_names:
        label = model_labels[model_name]
        print(f"  {label}...", end=' ')

        model = fit_model(model_name, X_scaled, y)
        estimator = get_fitted_estimator(model)

        if model_name in ('elastic_net', 'ridge'):
            # Use absolute coefficients as importance
            imp = np.abs(estimator.coef_)
        elif model_name == 'random_forest':
            # Use built-in feature importances
            imp = estimator.feature_importances_
        elif model_name == 'svr':
            # Use permutation importance (SVR has no coef_ with RBF kernel)
            perm_result = permutation_importance(
                estimator, X_scaled, y,
                n_repeats=30, random_state=RANDOM_STATE, scoring='r2')
            imp = perm_result.importances_mean
            # Clip negative values to 0
            imp = np.maximum(imp, 0)

        # Normalize to sum to 1
        if imp.sum() > 0:
            imp_norm = imp / imp.sum()
        else:
            imp_norm = imp

        importance_dict[label] = imp_norm

        # Rank features (1 = most important)
        rank = sp_stats.rankdata(-imp, method='average')
        top_idx = np.argsort(imp)[::-1][:3]
        top_str = ', '.join(f"{feature_names[i]}" for i in top_idx)
        print(f"top 3: {top_str}")

    # Build cross-model importance DataFrame
    imp_df = pd.DataFrame(importance_dict, index=feature_names)
    imp_df.index.name = 'Feature'
    imp_df.to_csv(TABLE_DIR / 'cross_model_importance.csv')
    print(f"  Saved: cross_model_importance.csv")

    # Compute Kendall's τ between model rankings
    print("\n  Cross-model ranking agreement (Kendall's τ):")
    rank_dict = {}
    for label, imp in importance_dict.items():
        rank_dict[label] = sp_stats.rankdata(-imp, method='average')

    tau_rows = []
    labels = list(rank_dict.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            tau, p = sp_stats.kendalltau(rank_dict[labels[i]], rank_dict[labels[j]])
            tau_rows.append({
                'Model_A': labels[i],
                'Model_B': labels[j],
                'Kendall_tau': tau,
                'p_value': p,
            })
            sig = '*' if p < 0.05 else ''
            print(f"    {labels[i]:15s} vs {labels[j]:15s}: τ={tau:.3f} (p={p:.3f}) {sig}")

    tau_df = pd.DataFrame(tau_rows)

    return imp_df, tau_df


# ── 3. Visualization ────────────────────────────────────────

def plot_shap_summary(shap_results):
    """SHAP beeswarm plots for key scenarios."""
    print("\n[3/3] Generating SHAP Figures")
    print("─" * 60)

    # Select 2 key scenarios × best model
    panels = [
        ('Personality → GPA|Elastic Net', 'A. Personality → GPA (Elastic Net)'),
        ('Behavior → PHQ-9|SVR', 'B. Behavior → PHQ-9 (SVR)'),
    ]

    # Fallback: use whatever models are available
    available_panels = []
    for key, title in panels:
        if key in shap_results:
            available_panels.append((key, title))
        else:
            # Try any model for this scenario
            scenario = key.split('|')[0]
            for k in shap_results:
                if k.startswith(scenario):
                    available_panels.append((k, title.replace(
                        key.split('|')[1], k.split('|')[1])))
                    break

    if len(available_panels) == 0:
        print("  No SHAP results available for plotting")
        return

    fig, axes = plt.subplots(1, len(available_panels),
                              figsize=(7 * len(available_panels), 6))
    if len(available_panels) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, available_panels):
        result = shap_results[key]
        shap_vals = result['shap_values']
        X = result['X']
        names = result['feature_names']

        # Sort features by mean |SHAP|
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)  # ascending for horizontal beeswarm

        # Plot beeswarm manually (horizontal strip plot)
        for i, idx in enumerate(order):
            vals = shap_vals[:, idx]
            feat_vals = X[:, idx]

            # Color by feature value
            norm = plt.Normalize(vmin=feat_vals.min(), vmax=feat_vals.max())
            colors = plt.cm.RdBu_r(norm(feat_vals))

            # Jitter y
            y_jitter = np.random.RandomState(42).normal(i, 0.15, len(vals))
            ax.scatter(vals, y_jitter, c=colors, s=15, alpha=0.7,
                      edgecolors='none')

        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=9)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('SHAP value')
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                    norm=plt.Normalize(vmin=-2, vmax=2))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label('Feature value\n(standardized)', fontsize=8)

    plt.suptitle('SHAP Feature Attribution Analysis',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: shap_summary.png")


def plot_cross_model_importance(imp_df):
    """Heatmap: feature importance across 4 models."""
    # Sort by average importance
    imp_df['Mean'] = imp_df.mean(axis=1)
    imp_sorted = imp_df.sort_values('Mean', ascending=True).drop(columns='Mean')

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(imp_sorted, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Normalized Importance'})
    ax.set_title('Cross-Model Feature Importance for GPA Prediction\n'
                 '(Pers + Beh → GPA, Standardized)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'cross_model_importance.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: cross_model_importance.png")


def plot_shap_dependence(shap_results):
    """SHAP dependence plots for key features showing non-linearity."""
    # Focus on Pers+Beh → GPA scenario
    target_key = None
    for key in shap_results:
        if 'Pers + Beh → GPA' in key and 'SVR' in key:
            target_key = key
            break
    if target_key is None:
        for key in shap_results:
            if 'Pers + Beh → GPA' in key:
                target_key = key
                break
    if target_key is None:
        print("  Skipping dependence plots (no suitable scenario)")
        return

    result = shap_results[target_key]
    shap_vals = result['shap_values']
    X = result['X']
    names = result['feature_names']

    # Top 4 features by mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:4]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, idx in zip(axes, top_idx):
        ax.scatter(X[:, idx], shap_vals[:, idx], c='#1565c0',
                  s=30, alpha=0.7, edgecolors='white', linewidth=0.3)
        # Fit LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(shap_vals[:, idx], X[:, idx], frac=0.6)
            ax.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2, alpha=0.8)
        except ImportError:
            pass
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel(f'{names[idx]} (standardized)')
        ax.set_ylabel('SHAP value')
        ax.set_title(names[idx], fontsize=10, fontweight='bold')

    plt.suptitle(f'SHAP Dependence Plots ({result["model_label"]}: Pers+Beh → GPA)',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: shap_dependence.png")


# ── main ──────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 11: ML INTERPRETABILITY")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    # 1. SHAP analysis
    shap_results = run_shap_analysis(df)

    # 2. Cross-model feature importance
    imp_df, tau_df = run_cross_model_importance(df)

    # 3. Visualization
    plot_shap_summary(shap_results)
    plot_cross_model_importance(imp_df)
    plot_shap_dependence(shap_results)

    # Final summary
    print("\n" + "─" * 60)
    print("KEY FINDINGS:")

    # Top features consensus
    print("\n  Cross-model top-3 features for GPA (by mean importance):")
    mean_imp = imp_df.mean(axis=1).sort_values(ascending=False)
    for i, (feat, val) in enumerate(mean_imp.head(3).items()):
        print(f"    {i+1}. {feat}: {val:.3f}")

    # Kendall's τ summary
    print(f"\n  Mean Kendall's τ: {tau_df['Kendall_tau'].mean():.3f}")
    print(f"  Range: [{tau_df['Kendall_tau'].min():.3f}, {tau_df['Kendall_tau'].max():.3f}]")

    print("\n" + "=" * 60)
    print("PHASE 11 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
