#!/usr/bin/env python3
"""
Bootstrap Mediation Analysis

Research Question: Do smartphone behavioral patterns mediate the effect
of Big Five personality traits on academic performance (GPA)?

Models:
  - Simple mediation: Trait → Behavior_PC → GPA (5 traits × 8 mediators = 40 tests)
  - Parallel mediation: Each trait → all 8 behavioral mediators simultaneously → GPA
  - Extended: Trait → Behavior + Wellbeing → GPA (serial mediation)

Method: Bootstrap 10,000 resamples for indirect effect CIs (bias-corrected).
Multiple comparison correction: Benjamini-Hochberg FDR.

Input:  data/processed/analysis_dataset.parquet
Output: results/tables/mediation_simple.csv
        results/tables/mediation_parallel.csv
        results/figures/mediation_forest_plot.png
        results/figures/mediation_path_diagram.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_PATH = project_root / 'data' / 'processed' / 'analysis_dataset.parquet'
TABLE_DIR = project_root / 'results' / 'tables'
FIGURE_DIR = project_root / 'results' / 'figures'

for d in [TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

TRAITS = ['extraversion', 'agreeableness', 'conscientiousness',
          'neuroticism', 'openness']
MEDIATORS = ['mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
             'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']
OUTCOME = 'gpa_overall'
N_BOOT = 10000
RANDOM_STATE = 42


def standardize(series):
    """Z-score standardization."""
    return (series - series.mean()) / series.std()


def bootstrap_mediation(x, m, y, n_boot=N_BOOT, seed=RANDOM_STATE):
    """
    Simple mediation: X → M → Y.
    Returns path coefficients and bootstrap CI for indirect effect.

    Paths:
      a: X → M (OLS)
      b: M → Y controlling for X (OLS)
      c: X → Y total effect
      c': X → Y direct (controlling for M)
      ab: indirect = a * b
    """
    from numpy.linalg import lstsq

    n = len(x)
    X_a = np.column_stack([np.ones(n), x])
    X_b = np.column_stack([np.ones(n), x, m])

    # Path a: X → M
    a_coefs, _, _, _ = lstsq(X_a, m, rcond=None)
    a = a_coefs[1]

    # Paths b, c': X + M → Y
    b_coefs, _, _, _ = lstsq(X_b, y, rcond=None)
    c_prime = b_coefs[1]
    b = b_coefs[2]

    # Total effect c: X → Y
    c_coefs, _, _, _ = lstsq(X_a, y, rcond=None)
    c = c_coefs[1]

    # Indirect effect
    ab = a * b

    # Bootstrap for indirect effect CI
    rng = np.random.RandomState(seed)
    boot_ab = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        x_b, m_b, y_b = x[idx], m[idx], y[idx]
        X_a_b = np.column_stack([np.ones(n), x_b])
        X_b_b = np.column_stack([np.ones(n), x_b, m_b])
        a_b, _, _, _ = lstsq(X_a_b, m_b, rcond=None)
        b_b, _, _, _ = lstsq(X_b_b, y_b, rcond=None)
        boot_ab[i] = a_b[1] * b_b[2]

    # Bias-corrected CI
    z0 = sp_stats.norm.ppf(np.mean(boot_ab < ab))
    alpha_levels = [0.025, 0.975]
    bc_ci = np.percentile(boot_ab,
                          [sp_stats.norm.cdf(2 * z0 + sp_stats.norm.ppf(a)) * 100
                           for a in alpha_levels])

    # p-value for indirect effect (proportion of bootstrap samples crossing zero)
    if ab >= 0:
        p_indirect = 2 * np.mean(boot_ab <= 0)
    else:
        p_indirect = 2 * np.mean(boot_ab >= 0)
    p_indirect = min(p_indirect, 1.0)

    # R² for outcome model
    y_pred = X_b @ b_coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'a': a, 'b': b, 'c': c, 'c_prime': c_prime,
        'ab': ab, 'ab_ci_lo': bc_ci[0], 'ab_ci_hi': bc_ci[1],
        'p_indirect': p_indirect, 'r_squared': r_sq,
        'boot_ab': boot_ab,
    }


def parallel_mediation(x, mediators_dict, y, n_boot=N_BOOT, seed=RANDOM_STATE):
    """
    Parallel mediation: X → M1, M2, ..., Mk → Y simultaneously.
    All mediators entered together in the outcome equation.
    """
    from numpy.linalg import lstsq

    n = len(x)
    m_names = list(mediators_dict.keys())
    m_arrays = [mediators_dict[name] for name in m_names]
    k = len(m_names)

    # Path a for each mediator: X → Mi
    X_a = np.column_stack([np.ones(n), x])
    a_paths = {}
    for name, m in zip(m_names, m_arrays):
        coefs, _, _, _ = lstsq(X_a, m, rcond=None)
        a_paths[name] = coefs[1]

    # Path b: X + all M → Y
    X_b = np.column_stack([np.ones(n), x] + m_arrays)
    b_coefs, _, _, _ = lstsq(X_b, y, rcond=None)
    c_prime = b_coefs[1]
    b_paths = {name: b_coefs[2 + i] for i, name in enumerate(m_names)}

    # Total effect
    c_coefs, _, _, _ = lstsq(X_a, y, rcond=None)
    c_total = c_coefs[1]

    # Specific indirect effects
    specific_ab = {name: a_paths[name] * b_paths[name] for name in m_names}
    total_indirect = sum(specific_ab.values())

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot_specific = {name: np.zeros(n_boot) for name in m_names}

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        x_b = x[idx]
        y_b = y[idx]
        m_b = [m[idx] for m in m_arrays]

        X_a_b = np.column_stack([np.ones(n), x_b])
        X_b_b = np.column_stack([np.ones(n), x_b] + m_b)

        for j, name in enumerate(m_names):
            a_coef, _, _, _ = lstsq(X_a_b, m_b[j], rcond=None)
            b_coef, _, _, _ = lstsq(X_b_b, y_b, rcond=None)
            boot_specific[name][i] = a_coef[1] * b_coef[2 + j]

    results = {}
    for name in m_names:
        ab = specific_ab[name]
        ci = np.percentile(boot_specific[name], [2.5, 97.5])
        if ab >= 0:
            p = 2 * np.mean(boot_specific[name] <= 0)
        else:
            p = 2 * np.mean(boot_specific[name] >= 0)
        results[name] = {
            'a': a_paths[name], 'b': b_paths[name],
            'ab': ab, 'ab_ci_lo': ci[0], 'ab_ci_hi': ci[1],
            'p_indirect': min(p, 1.0),
        }

    return results, c_total, c_prime, total_indirect


def run_simple_mediation(df):
    """Run all simple mediation models (5 traits × 8 mediators)."""
    print("\n[1/4] Simple Mediation Analysis (Bootstrap)")
    print("─" * 60)

    rows = []
    for trait in TRAITS:
        for mediator in MEDIATORS:
            subset = df[[trait, mediator, OUTCOME]].dropna()
            if len(subset) < 10:
                continue

            x = standardize(subset[trait]).values
            m = standardize(subset[mediator]).values
            y = standardize(subset[OUTCOME]).values

            result = bootstrap_mediation(x, m, y)
            sig = '*' if (result['ab_ci_lo'] > 0 or result['ab_ci_hi'] < 0) else ''

            rows.append({
                'Trait': trait, 'Mediator': mediator,
                'N': len(subset),
                'a (X→M)': result['a'],
                'b (M→Y|X)': result['b'],
                'c (total)': result['c'],
                "c' (direct)": result['c_prime'],
                'ab (indirect)': result['ab'],
                'ab_CI_lo': result['ab_ci_lo'],
                'ab_CI_hi': result['ab_ci_hi'],
                'p_indirect': result['p_indirect'],
                'R²': result['r_squared'],
                'sig': sig,
            })

    results_df = pd.DataFrame(rows)

    # FDR correction
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_indirect'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df['sig_fdr'] = ['*' if p < 0.05 else '' for p in p_fdr]

    # Report
    n_sig = results_df['sig'].str.contains(r'\*').sum()
    n_sig_fdr = results_df['sig_fdr'].str.contains(r'\*').sum() if 'sig_fdr' in results_df.columns else 0
    print(f"  Total tests: {len(results_df)}")
    print(f"  Significant indirect effects (95% CI): {n_sig}")
    print(f"  Significant after FDR correction: {n_sig_fdr}")

    # Show top effects by |ab|
    top = results_df.assign(_abs=results_df['ab (indirect)'].abs()).nlargest(10, '_abs').drop(columns='_abs')
    for _, row in top.iterrows():
        print(f"  {row['Trait']:20s} → {row['Mediator']:15s} → GPA  "
              f"ab={row['ab (indirect)']:.3f} [{row['ab_CI_lo']:.3f}, {row['ab_CI_hi']:.3f}] "
              f"{row['sig']}")

    return results_df


def run_parallel_mediation(df):
    """Run parallel mediation (each trait with all 8 mediators simultaneously)."""
    print("\n[2/4] Parallel Mediation Analysis")
    print("─" * 60)

    rows = []
    for trait in TRAITS:
        cols = [trait] + MEDIATORS + [OUTCOME]
        subset = df[cols].dropna()
        if len(subset) < 10:
            print(f"  {trait}: insufficient data (n={len(subset)})")
            continue

        x = standardize(subset[trait]).values
        y = standardize(subset[OUTCOME]).values
        mediators_dict = {m: standardize(subset[m]).values for m in MEDIATORS}

        results, c_total, c_prime, total_indirect = parallel_mediation(
            x, mediators_dict, y)

        print(f"\n  {trait.upper()} (n={len(subset)})")
        print(f"    Total effect (c): {c_total:.3f}")
        print(f"    Direct effect (c'): {c_prime:.3f}")
        print(f"    Total indirect: {total_indirect:.3f}")

        for med_name, res in results.items():
            sig = '*' if (res['ab_ci_lo'] > 0 or res['ab_ci_hi'] < 0) else ''
            print(f"    via {med_name:15s}: ab={res['ab']:.3f} "
                  f"[{res['ab_ci_lo']:.3f}, {res['ab_ci_hi']:.3f}] {sig}")
            rows.append({
                'Trait': trait, 'Mediator': med_name,
                'a': res['a'], 'b': res['b'],
                'ab': res['ab'],
                'ab_CI_lo': res['ab_ci_lo'], 'ab_CI_hi': res['ab_ci_hi'],
                'p_indirect': res['p_indirect'],
                'c_total': c_total, 'c_prime': c_prime,
                'sig': sig,
            })

    return pd.DataFrame(rows)


def sensitivity_analysis(df, simple_results):
    """Leave-one-out influence analysis for significant effects."""
    print("\n[3/4] Sensitivity Analysis (Leave-One-Out)")
    print("─" * 60)

    sig_rows = simple_results[simple_results['sig'] == '*']
    if len(sig_rows) == 0:
        # If no significant results, run LOO on top 5 effects
        sig_rows = simple_results.assign(_abs=simple_results['ab (indirect)'].abs()).nlargest(5, '_abs').drop(columns='_abs')
        print("  No significant effects; running LOO on top 5 by |ab|")

    for _, row in sig_rows.iterrows():
        trait, mediator = row['Trait'], row['Mediator']
        cols = [trait, mediator, OUTCOME]
        subset = df[cols].dropna().reset_index(drop=True)
        n = len(subset)

        loo_abs = []
        for i in range(n):
            loo = subset.drop(i)
            x = standardize(loo[trait]).values
            m = standardize(loo[mediator]).values
            y = standardize(loo[OUTCOME]).values
            res = bootstrap_mediation(x, m, y, n_boot=1000, seed=42)
            loo_abs.append(res['ab'])

        loo_abs = np.array(loo_abs)
        influence = np.abs(loo_abs - row['ab (indirect)'])
        most_influential = np.argmax(influence)
        uid = subset.index[most_influential]

        print(f"  {trait} → {mediator}: original ab={row['ab (indirect)']:.3f}, "
              f"LOO range=[{loo_abs.min():.3f}, {loo_abs.max():.3f}], "
              f"most influential obs={most_influential}")


def plot_forest(simple_results, output_path):
    """Forest plot of indirect effects with 95% CIs."""
    df = simple_results.copy()
    df['label'] = df['Trait'].str[:4] + ' → ' + df['Mediator'].str.replace('_pc1', '')
    df = df.sort_values('ab (indirect)')

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
    y_pos = range(len(df))

    colors = ['#d32f2f' if sig == '*' else '#666666' for sig in df['sig']]
    ax.hlines(y_pos, df['ab_CI_lo'], df['ab_CI_hi'], colors=colors, linewidth=1.5)
    ax.scatter(df['ab (indirect)'], y_pos, c=colors, s=40, zorder=5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df['label'], fontsize=7)
    ax.set_xlabel('Indirect Effect (ab) with 95% Bootstrap CI')
    ax.set_title('Simple Mediation: Trait → Behavior → GPA\n(red = significant)')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_path_diagram(simple_results, output_path):
    """Heatmap-style path diagram showing a, b, and ab paths."""
    pivot_a = simple_results.pivot(
        index='Trait', columns='Mediator', values='a (X→M)').reindex(
        index=TRAITS, columns=MEDIATORS)
    pivot_b = simple_results.pivot(
        index='Trait', columns='Mediator', values='b (M→Y|X)').reindex(
        index=TRAITS, columns=MEDIATORS)
    pivot_ab = simple_results.pivot(
        index='Trait', columns='Mediator', values='ab (indirect)').reindex(
        index=TRAITS, columns=MEDIATORS)

    short_med = [m.replace('_pc1', '') for m in MEDIATORS]
    short_trait = [t[:4].title() for t in TRAITS]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, data, title in zip(axes,
                                [pivot_a, pivot_b, pivot_ab],
                                ['Path a (Trait → Behavior)',
                                 'Path b (Behavior → GPA | Trait)',
                                 'Indirect Effect (a × b)']):
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-0.6, vmax=0.6, xticklabels=short_med,
                    yticklabels=short_trait, ax=ax, linewidths=0.5,
                    annot_kws={'size': 8})
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.suptitle('Bootstrap Mediation: Big Five → Behavior → GPA', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("BOOTSTRAP MEDIATION ANALYSIS")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    # Simple mediation
    simple = run_simple_mediation(df)
    simple.to_csv(TABLE_DIR / 'mediation_simple.csv', index=False)

    # Parallel mediation
    parallel = run_parallel_mediation(df)
    parallel.to_csv(TABLE_DIR / 'mediation_parallel.csv', index=False)

    # Sensitivity
    sensitivity_analysis(df, simple)

    # Plots
    print("\n[4/4] Generating figures...")
    plot_forest(simple, FIGURE_DIR / 'mediation_forest_plot.png')
    plot_path_diagram(simple, FIGURE_DIR / 'mediation_path_diagram.png')

    print("\n" + "=" * 60)
    print("MEDIATION ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
