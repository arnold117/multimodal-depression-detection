#!/usr/bin/env python3
"""
Latent Profile Analysis (LPA)

Research Question: Can we identify distinct personality-behavior student
profiles? Do these profiles differ in academic and wellbeing outcomes?

Method:
  - Gaussian Mixture Models (GMM) with k = 1..4 profiles
  - Input: Big Five traits + 8 behavioral composites (standardized)
  - Model selection: BIC
  - Profile comparison on GPA, PHQ-9, PSS, Loneliness, Flourishing

Input:  data/processed/analysis_dataset.parquet
Output: results/tables/lpa_fit_indices.csv
        results/tables/lpa_profiles.csv
        results/tables/lpa_outcome_comparison.csv
        results/figures/lpa_bic.png
        results/figures/lpa_radar.png
        results/figures/lpa_outcomes.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats as sp_stats
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

PROFILE_VARS = ['extraversion', 'agreeableness', 'conscientiousness',
                'neuroticism', 'openness',
                'mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
                'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']

OUTCOME_VARS = ['gpa_overall', 'phq9_total', 'pss_total',
                'loneliness_total', 'flourishing_total']

MAX_K = 4
N_INIT = 100
RANDOM_STATE = 42
N_BOOT_STABILITY = 500


def fit_gmm_models(X, max_k=MAX_K):
    """Fit GMM for k = 1..max_k, return fit indices."""
    results = []
    models = {}

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            n_init=N_INIT,
            max_iter=500,
            random_state=RANDOM_STATE,
        )
        gmm.fit(X)

        n = X.shape[0]
        p = X.shape[1]
        n_params = k * p + k * p * (p + 1) / 2 + k - 1  # means + covariances + weights
        log_lik = gmm.score(X) * n

        # AIC, BIC, sample-adjusted BIC
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n)
        sabic = -2 * log_lik + n_params * np.log((n + 2) / 24)

        # Entropy
        probs = gmm.predict_proba(X)
        entropy = -np.sum(probs * np.log(probs + 1e-10)) / n
        entropy_r2 = 1 - entropy / np.log(k) if k > 1 else 1.0

        labels = gmm.predict(X)
        min_class = min(np.bincount(labels))

        results.append({
            'k': k, 'Log-Likelihood': log_lik,
            'AIC': aic, 'BIC': bic, 'SABIC': sabic,
            'Entropy': entropy_r2,
            'Min_Class_Size': min_class,
            'Converged': gmm.converged_,
        })
        models[k] = gmm

    return pd.DataFrame(results), models


def bootstrap_stability(X, best_k, n_boot=N_BOOT_STABILITY):
    """Bootstrap Adjusted Rand Index for profile stability."""
    from sklearn.metrics import adjusted_rand_score

    # Reference labels
    gmm_ref = GaussianMixture(
        n_components=best_k, covariance_type='full',
        n_init=N_INIT, random_state=RANDOM_STATE)
    gmm_ref.fit(X)
    ref_labels = gmm_ref.predict(X)

    rng = np.random.RandomState(RANDOM_STATE)
    ari_scores = []

    for _ in range(n_boot):
        idx = rng.randint(0, len(X), size=len(X))
        X_boot = X[idx]
        gmm_boot = GaussianMixture(
            n_components=best_k, covariance_type='full',
            n_init=20, random_state=rng.randint(0, 10000))
        gmm_boot.fit(X_boot)
        boot_labels = gmm_boot.predict(X)  # predict on FULL data
        ari_scores.append(adjusted_rand_score(ref_labels, boot_labels))

    return np.array(ari_scores)


def cliffs_delta(x, y):
    """Cliff's delta effect size (non-parametric)."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0
    count = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                count += 1
            elif xi < yi:
                count -= 1
    return count / (nx * ny)


def compare_outcomes(df, labels, outcome_vars):
    """Compare outcome variables across profiles."""
    df = df.copy()
    df['Profile'] = labels
    unique_profiles = sorted(df['Profile'].unique())
    k = len(unique_profiles)

    rows = []
    for var in outcome_vars:
        groups = [df[df['Profile'] == p][var].dropna().values for p in unique_profiles]

        # Descriptive per group
        means = [g.mean() if len(g) > 0 else np.nan for g in groups]
        sds = [g.std() if len(g) > 0 else np.nan for g in groups]
        ns = [len(g) for g in groups]

        # Kruskal-Wallis (non-parametric ANOVA)
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) >= 2:
            stat, p_kw = sp_stats.kruskal(*valid_groups)
        else:
            stat, p_kw = np.nan, np.nan

        row = {'Variable': var, 'H_stat': stat, 'p_kruskal': p_kw}
        for i, p in enumerate(unique_profiles):
            row[f'Profile_{p}_n'] = ns[i]
            row[f'Profile_{p}_mean'] = means[i]
            row[f'Profile_{p}_sd'] = sds[i]

        # Pairwise Cliff's delta (if k == 2)
        if k == 2 and all(len(g) >= 2 for g in groups):
            row['cliffs_delta'] = cliffs_delta(groups[0], groups[1])

        rows.append(row)

    return pd.DataFrame(rows)


def plot_bic(fit_df, output_path):
    """BIC elbow plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fit_df['k'], fit_df['BIC'], 'o-', color='#1565c0', linewidth=2, markersize=8)
    ax.plot(fit_df['k'], fit_df['AIC'], 's--', color='#d32f2f', linewidth=1.5, markersize=6, alpha=0.7)
    best_k = fit_df.loc[fit_df['BIC'].idxmin(), 'k']
    ax.axvline(best_k, color='green', linestyle=':', alpha=0.7, label=f'Best k={best_k}')
    ax.set_xlabel('Number of Profiles (k)')
    ax.set_ylabel('Information Criterion')
    ax.set_title('LPA Model Selection')
    ax.legend(['BIC', 'AIC', f'Best k={best_k}'])
    ax.set_xticks(fit_df['k'].values)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_radar(means_df, profile_vars, output_path):
    """Radar plot of profile means on standardized variables."""
    profiles = sorted(means_df['Profile'].unique())
    n_vars = len(profile_vars)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    short_names = {
        'extraversion': 'E', 'agreeableness': 'A', 'conscientiousness': 'C',
        'neuroticism': 'N', 'openness': 'O',
        'mobility_pc1': 'Mobility', 'digital_pc1': 'Digital',
        'social_pc1': 'Social', 'activity_pc1': 'Activity',
        'screen_pc1': 'Screen', 'proximity_pc1': 'Proximity',
        'face2face_pc1': 'F2F', 'audio_pc1': 'Audio',
    }

    colors = ['#1565c0', '#d32f2f', '#2e7d32', '#ff8f00']
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, profile in enumerate(profiles):
        row = means_df[means_df['Profile'] == profile]
        values = [row[v].values[0] for v in profile_vars]
        values += values[:1]
        n_members = row['n'].values[0]
        ax.plot(angles, values, 'o-', color=colors[i % len(colors)],
                linewidth=2, label=f'Profile {profile} (n={n_members})')
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

    labels = [short_names.get(v, v) for v in profile_vars]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Latent Profile Characteristics\n(Standardized Means)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_outcomes(outcome_df, df_with_labels, output_path):
    """Box plots of outcome variables by profile."""
    outcome_vars = outcome_df['Variable'].tolist()
    n_vars = len(outcome_vars)
    profiles = sorted(df_with_labels['Profile'].unique())

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]

    colors = ['#bbdefb', '#ffcdd2', '#c8e6c9', '#fff9c4']
    for i, var in enumerate(outcome_vars):
        data_plot = df_with_labels[[var, 'Profile']].dropna()
        sns.boxplot(data=data_plot, x='Profile', y=var, ax=axes[i],
                    palette=colors[:len(profiles)])
        sns.stripplot(data=data_plot, x='Profile', y=var, ax=axes[i],
                      color='black', size=4, alpha=0.5)

        # Add stats
        row = outcome_df[outcome_df['Variable'] == var].iloc[0]
        p = row['p_kruskal']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        axes[i].set_title(f'{var}\n(H={row["H_stat"]:.1f}, p={p:.3f} {sig})', fontsize=9)
        axes[i].set_xlabel('Profile')

    plt.suptitle('Outcome Differences Across Profiles', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("LATENT PROFILE ANALYSIS")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    # Prepare input variables
    print("\n[1/5] Preparing profile variables...")
    available = [v for v in PROFILE_VARS if v in df.columns]
    subset = df[['uid'] + available + OUTCOME_VARS].dropna(subset=available)
    print(f"  Complete cases: {len(subset)} (using {len(available)} profile variables)")

    scaler = StandardScaler()
    X = scaler.fit_transform(subset[available])

    # Fit models
    print("\n[2/5] Fitting GMM models (k=1..4)...")
    fit_df, models = fit_gmm_models(X)
    print(fit_df.to_string(index=False))
    fit_df.to_csv(TABLE_DIR / 'lpa_fit_indices.csv', index=False)

    # Select best k
    best_k = int(fit_df.loc[fit_df['BIC'].idxmin(), 'k'])
    print(f"\n  Best k by BIC: {best_k}")

    # Stability check
    print(f"\n[3/5] Bootstrap stability (k={best_k}, {N_BOOT_STABILITY} resamples)...")
    ari_scores = bootstrap_stability(X, best_k)
    print(f"  ARI: mean={ari_scores.mean():.3f}, "
          f"95% CI=[{np.percentile(ari_scores, 2.5):.3f}, "
          f"{np.percentile(ari_scores, 97.5):.3f}]")

    # Profile characterization
    print(f"\n[4/5] Characterizing {best_k} profiles...")
    best_model = models[best_k]
    labels = best_model.predict(X)
    subset = subset.copy()
    subset['Profile'] = labels

    # Profile means (standardized)
    profile_means = []
    for p in sorted(np.unique(labels)):
        mask = labels == p
        row = {'Profile': p, 'n': int(mask.sum())}
        for j, var in enumerate(available):
            row[var] = X[mask, j].mean()
        profile_means.append(row)
    means_df = pd.DataFrame(profile_means)
    means_df.to_csv(TABLE_DIR / 'lpa_profiles.csv', index=False)

    print("\n  Profile sizes:")
    for _, row in means_df.iterrows():
        print(f"    Profile {int(row['Profile'])}: n={int(row['n'])}")

    # Outcome comparison
    print("\n[5/5] Comparing outcomes across profiles...")
    outcome_comp = compare_outcomes(subset, labels, OUTCOME_VARS)
    outcome_comp.to_csv(TABLE_DIR / 'lpa_outcome_comparison.csv', index=False)

    for _, row in outcome_comp.iterrows():
        p = row['p_kruskal']
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {row['Variable']:20s}: H={row['H_stat']:.2f}, p={p:.4f} {sig}")

    # Figures
    print("\n  Generating figures...")
    plot_bic(fit_df, FIGURE_DIR / 'lpa_bic.png')
    plot_radar(means_df, available, FIGURE_DIR / 'lpa_radar.png')
    plot_outcomes(outcome_comp, subset, FIGURE_DIR / 'lpa_outcomes.png')

    print("\n" + "=" * 60)
    print("LATENT PROFILE ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
