#!/usr/bin/env python3
"""
Elastic Net Regularized Regression for GPA Prediction

Research Question: Which data source (personality, behavior, wellbeing)
best predicts GPA? Does combining them improve prediction?

Models:
  - Model 1: Personality only (5 traits)
  - Model 2: Behavior only (8 PCA composites)
  - Model 3: Wellbeing only (~6 measures)
  - Model 4: All combined (~19 features)
  - Model 5: All raw features (~87 behavioral + 5 personality + 6 wellbeing)

Evaluation:
  - LOO-CV: R², RMSE, MAE
  - Bootstrap coefficient CIs (10,000 resamples)
  - Permutation test for model significance (1,000 permutations)
  - Incremental R² analysis

Input:  data/processed/analysis_dataset.parquet
Output: results/tables/elastic_net_comparison.csv
        results/tables/elastic_net_coefficients.csv
        results/figures/elastic_net_model_comparison.png
        results/figures/elastic_net_coefficients.png
        results/figures/elastic_net_predicted_vs_actual.png
        results/figures/elastic_net_feature_importance.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

DATA_PATH = project_root / 'data' / 'processed' / 'analysis_dataset.parquet'
TABLE_DIR = project_root / 'results' / 'tables'
FIGURE_DIR = project_root / 'results' / 'figures'

for d in [TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
BEHAVIOR_PC = ['mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
               'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']
WELLBEING = ['phq9_total', 'pss_total', 'loneliness_total',
             'flourishing_total', 'panas_positive', 'panas_negative']
OUTCOME = 'gpa_overall'

L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
N_ALPHAS = 50
N_BOOT = 10000
N_PERM = 1000
RANDOM_STATE = 42


def loo_cv_elastic_net(X, y, l1_ratios=L1_RATIOS, n_alphas=N_ALPHAS):
    """LOO-CV with Elastic Net. Returns predictions, best model, and scores."""
    loo = LeaveOneOut()
    n = len(y)
    y_pred = np.zeros(n)

    # First pass: find best hyperparameters with nested CV
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    enet_cv = ElasticNetCV(
        l1_ratio=l1_ratios, n_alphas=n_alphas,
        cv=min(5, n - 1), random_state=RANDOM_STATE, max_iter=10000)
    enet_cv.fit(X_scaled, y)
    best_alpha = enet_cv.alpha_
    best_l1 = enet_cv.l1_ratio_

    # LOO predictions using fixed hyperparameters
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler_loo = StandardScaler()
        X_train_s = scaler_loo.fit_transform(X_train)
        X_test_s = scaler_loo.transform(X_test)

        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1,
                           max_iter=10000, random_state=RANDOM_STATE)
        model.fit(X_train_s, y_train)
        y_pred[test_idx] = model.predict(X_test_s)

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    # Full model for coefficients
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    model_full = ElasticNet(alpha=best_alpha, l1_ratio=best_l1,
                            max_iter=10000, random_state=RANDOM_STATE)
    model_full.fit(X_full, y)

    return {
        'y_pred': y_pred, 'r2': r2, 'rmse': rmse, 'mae': mae,
        'alpha': best_alpha, 'l1_ratio': best_l1,
        'coef': model_full.coef_, 'intercept': model_full.intercept_,
        'n_nonzero': np.sum(model_full.coef_ != 0),
    }


def bootstrap_coefficients(X, y, alpha, l1_ratio, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Bootstrap CIs for Elastic Net coefficients."""
    rng = np.random.RandomState(seed)
    n, p = X.shape
    boot_coefs = np.zeros((n_boot, p))

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        X_b, y_b = X[idx], y[idx]
        scaler = StandardScaler()
        X_b_s = scaler.fit_transform(X_b)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           max_iter=10000, random_state=seed)
        model.fit(X_b_s, y_b)
        boot_coefs[i] = model.coef_

    return boot_coefs


def permutation_test(X, y, observed_r2, alpha, l1_ratio, n_perm=N_PERM, seed=RANDOM_STATE):
    """Permutation test for model significance."""
    rng = np.random.RandomState(seed)
    n = len(y)
    perm_r2 = np.zeros(n_perm)

    for i in range(n_perm):
        y_perm = rng.permutation(y)
        loo = LeaveOneOut()
        y_pred = np.zeros(n)

        for train_idx, test_idx in loo.split(X):
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X[train_idx])
            X_test_s = scaler.transform(X[test_idx])
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               max_iter=10000, random_state=seed)
            model.fit(X_train_s, y_perm[train_idx])
            y_pred[test_idx] = model.predict(X_test_s)

        perm_r2[i] = r2_score(y_perm, y_pred)

    p_value = np.mean(perm_r2 >= observed_r2)
    return p_value, perm_r2


def run_all_models(df):
    """Run all 5 model configurations."""
    print("\n[1/4] Running model comparisons (LOO-CV)...")
    print("─" * 60)

    # Define model configurations
    models_config = {
        'M1: Personality': PERSONALITY,
        'M2: Behavior': BEHAVIOR_PC,
        'M3: Wellbeing': WELLBEING,
        'M4: Combined (composites)': PERSONALITY + BEHAVIOR_PC + WELLBEING,
    }

    # Raw behavioral features (non-PC)
    raw_beh = [c for c in df.columns if c not in
               ['uid', OUTCOME] + PERSONALITY + BEHAVIOR_PC + WELLBEING +
               ['gpa_13s', 'cs65_grade'] and not c.endswith('_pc1')]
    if len(raw_beh) > 0:
        models_config['M5: All raw features'] = PERSONALITY + raw_beh + WELLBEING

    results = {}
    comparison_rows = []

    for name, features in models_config.items():
        available = [f for f in features if f in df.columns]
        subset = df[available + [OUTCOME]].dropna()
        X = subset[available].values
        y = subset[OUTCOME].values

        if len(subset) < 10 or len(available) == 0:
            print(f"  {name}: skipped (n={len(subset)}, p={len(available)})")
            continue

        result = loo_cv_elastic_net(X, y)
        results[name] = {**result, 'features': available, 'n': len(subset)}

        print(f"  {name:35s}: R²={result['r2']:.3f}, RMSE={result['rmse']:.3f}, "
              f"MAE={result['mae']:.3f} (n={len(subset)}, p={len(available)}, "
              f"selected={result['n_nonzero']}, α={result['alpha']:.4f}, "
              f"l1={result['l1_ratio']:.2f})")

        comparison_rows.append({
            'Model': name, 'N': len(subset), 'p_features': len(available),
            'p_selected': result['n_nonzero'],
            'LOO_R²': result['r2'], 'RMSE': result['rmse'], 'MAE': result['mae'],
            'alpha': result['alpha'], 'l1_ratio': result['l1_ratio'],
        })

    return results, pd.DataFrame(comparison_rows)


def analyze_best_model(df, results):
    """Detailed analysis of the best-performing model."""
    # Find best model by R²
    best_name = max(results, key=lambda k: results[k]['r2'])
    best = results[best_name]
    print(f"\n[2/4] Detailed analysis of best model: {best_name}")
    print("─" * 60)

    features = best['features']
    subset = df[features + [OUTCOME]].dropna()
    X = subset[features].values
    y = subset[OUTCOME].values

    # Bootstrap coefficients
    print(f"  Bootstrap coefficients ({N_BOOT} resamples)...")
    boot_coefs = bootstrap_coefficients(X, y, best['alpha'], best['l1_ratio'])

    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': best['coef'],
        'Boot_Mean': boot_coefs.mean(axis=0),
        'Boot_SD': boot_coefs.std(axis=0),
        'CI_lo': np.percentile(boot_coefs, 2.5, axis=0),
        'CI_hi': np.percentile(boot_coefs, 97.5, axis=0),
        'Selection_Freq': np.mean(boot_coefs != 0, axis=0),
    })
    coef_df['Significant'] = (coef_df['CI_lo'] > 0) | (coef_df['CI_hi'] < 0)
    coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)

    print("\n  Top features:")
    for _, row in coef_df.head(10).iterrows():
        sig = '*' if row['Significant'] else ''
        print(f"    {row['Feature']:30s}: β={row['Coefficient']:.3f} "
              f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] "
              f"sel={row['Selection_Freq']:.0%} {sig}")

    # Permutation test
    print(f"\n  Permutation test ({N_PERM} permutations)...")
    p_perm, perm_r2 = permutation_test(X, y, best['r2'], best['alpha'], best['l1_ratio'])
    print(f"  Observed R²: {best['r2']:.3f}, p_perm = {p_perm:.4f}")

    return coef_df, p_perm, perm_r2, best_name


def incremental_r2(df, results):
    """Incremental R² analysis: personality → +behavior → +wellbeing."""
    print("\n[3/4] Incremental R² Analysis")
    print("─" * 60)

    models = ['M1: Personality', 'M4: Combined (composites)']
    steps = []
    prev_r2 = 0

    # Step 1: Personality alone
    if 'M1: Personality' in results:
        r2 = results['M1: Personality']['r2']
        steps.append({'Step': 'Personality', 'R²': r2, 'ΔR²': r2})
        prev_r2 = r2

    # Step 2: + Behavior
    if 'M2: Behavior' in results:
        # Run personality + behavior
        feats = PERSONALITY + BEHAVIOR_PC
        available = [f for f in feats if f in df.columns]
        subset = df[available + [OUTCOME]].dropna()
        X = subset[available].values
        y = subset[OUTCOME].values
        res = loo_cv_elastic_net(X, y)
        steps.append({'Step': '+ Behavior', 'R²': res['r2'], 'ΔR²': res['r2'] - prev_r2})
        prev_r2 = res['r2']

    # Step 3: + Wellbeing (full combined)
    if 'M4: Combined (composites)' in results:
        r2 = results['M4: Combined (composites)']['r2']
        steps.append({'Step': '+ Wellbeing', 'R²': r2, 'ΔR²': r2 - prev_r2})

    steps_df = pd.DataFrame(steps)
    print(steps_df.to_string(index=False))
    return steps_df


def plot_model_comparison(comparison_df, output_path):
    """Bar chart comparing LOO-CV R² across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric, label in zip(axes,
                                  ['LOO_R²', 'RMSE', 'MAE'],
                                  ['LOO-CV R²', 'RMSE', 'MAE']):
        colors = ['#bbdefb', '#ffcdd2', '#c8e6c9', '#fff9c4', '#e1bee7']
        bars = ax.barh(comparison_df['Model'], comparison_df[metric],
                      color=colors[:len(comparison_df)])
        ax.set_xlabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                   f'{val:.3f}', va='center', fontsize=9)

    plt.suptitle('Elastic Net Model Comparison (LOO-CV)', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_coefficients(coef_df, model_name, output_path):
    """Coefficient plot with bootstrap CIs."""
    df = coef_df[coef_df['Coefficient'] != 0].sort_values('Coefficient')
    if len(df) == 0:
        df = coef_df.nlargest(10, 'Coefficient', key=abs).sort_values('Coefficient')

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    colors = ['#d32f2f' if sig else '#1565c0' for sig in df['Significant']]

    ax.barh(df['Feature'], df['Coefficient'], color=colors, alpha=0.7)
    ax.errorbar(df['Coefficient'], df['Feature'],
               xerr=[df['Coefficient'] - df['CI_lo'], df['CI_hi'] - df['Coefficient']],
               fmt='none', color='black', capsize=3, linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Standardized Coefficient')
    ax.set_title(f'{model_name}\n(red = significant, blue = not significant)')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_predicted_vs_actual(results, best_name, df, output_path):
    """Scatter plot of predicted vs actual GPA."""
    best = results[best_name]
    features = best['features']
    subset = df[features + [OUTCOME]].dropna()
    y_actual = subset[OUTCOME].values
    y_pred = best['y_pred']

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_actual, y_pred, c='#1565c0', s=60, alpha=0.7, edgecolors='white')

    # Identity line
    lims = [min(y_actual.min(), y_pred.min()) - 0.1,
            max(y_actual.max(), y_pred.max()) + 0.1]
    ax.plot(lims, lims, '--', color='gray', alpha=0.5)

    # Regression line
    z = np.polyfit(y_actual, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, p(x_line), '-', color='#d32f2f', alpha=0.7)

    ax.set_xlabel('Actual GPA')
    ax.set_ylabel('Predicted GPA (LOO-CV)')
    ax.set_title(f'{best_name}\nLOO-CV R²={best["r2"]:.3f}, RMSE={best["rmse"]:.3f}')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_feature_importance(coef_df, output_path):
    """Feature selection frequency from bootstrap."""
    df = coef_df.sort_values('Selection_Freq', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    colors = ['#2e7d32' if f > 0.8 else '#ff8f00' if f > 0.5 else '#757575'
              for f in df['Selection_Freq']]
    ax.barh(df['Feature'], df['Selection_Freq'], color=colors)
    ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Bootstrap Selection Frequency')
    ax.set_title('Feature Selection Stability (10,000 Bootstrap Resamples)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("ELASTIC NET REGULARIZED REGRESSION")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    # Run models
    results, comparison_df = run_all_models(df)
    comparison_df.to_csv(TABLE_DIR / 'elastic_net_comparison.csv', index=False)

    # Best model analysis
    coef_df, p_perm, perm_r2, best_name = analyze_best_model(df, results)
    coef_df.to_csv(TABLE_DIR / 'elastic_net_coefficients.csv', index=False)

    # Incremental R²
    incremental_r2(df, results)

    # Figures
    print("\n[4/4] Generating figures...")
    plot_model_comparison(comparison_df, FIGURE_DIR / 'elastic_net_model_comparison.png')
    plot_coefficients(coef_df, best_name, FIGURE_DIR / 'elastic_net_coefficients.png')
    plot_predicted_vs_actual(results, best_name, df, FIGURE_DIR / 'elastic_net_predicted_vs_actual.png')
    plot_feature_importance(coef_df, FIGURE_DIR / 'elastic_net_feature_importance.png')

    print("\n" + "=" * 60)
    print("ELASTIC NET ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
