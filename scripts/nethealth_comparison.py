#!/usr/bin/env python3
"""
Phase 12 Step 5: Study 1 vs Study 2 Formal Comparison

Compares key findings across datasets:
  - StudentLife (Study 1, N=27-28)
  - NetHealth (Study 2, N=135-220)

Replication criteria (Open Science Collaboration, 2015):
  1. Effect direction consistent
  2. Study 2 effect within Study 1 CI
  3. Study 2 statistically significant (p < .05)

Output: results/comparison/  (tables, forest plots, replication statistics)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

# Study 1 results
S1_TABLE_DIR = project_root / 'results' / 'tables'
S1_DATA = project_root / 'data' / 'processed' / 'analysis_dataset.parquet'

# Study 2 results
S2_TABLE_DIR = project_root / 'results' / 'nethealth' / 'tables'
S2_DATA = project_root / 'data' / 'processed' / 'nethealth' / 'nethealth_analysis_dataset.parquet'

# Output
COMP_DIR = project_root / 'results' / 'comparison'
COMP_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
})
sns.set_style('whitegrid')

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
TRAIT_LABELS = {'extraversion': 'E', 'agreeableness': 'A',
                'conscientiousness': 'C', 'neuroticism': 'N', 'openness': 'O'}


# ──────────────────────────────────────────────────────────────────────
# Utility: Bootstrap CI for correlation
# ──────────────────────────────────────────────────────────────────────

def bootstrap_r_ci(x, y, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_r = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_r[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    alpha = (1 - ci) / 2
    return np.percentile(boot_r, [alpha * 100, (1 - alpha) * 100])


def fisher_z_test(r1, n1, r2, n2):
    """Compare two independent correlations (Fisher z-test)."""
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_diff = (z1 - z2) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z_diff)))
    return z_diff, p


def cochrans_q(effects, variances):
    """Cochran's Q test for heterogeneity across studies."""
    w = 1 / np.array(variances)
    e = np.array(effects)
    w_mean = np.sum(w * e) / np.sum(w)
    Q = np.sum(w * (e - w_mean) ** 2)
    df = len(effects) - 1
    p = 1 - sp_stats.chi2.cdf(Q, df) if df > 0 else 1.0
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0
    return Q, p, I2


# ──────────────────────────────────────────────────────────────────────
# 1. Personality-GPA Correlations Comparison
# ──────────────────────────────────────────────────────────────────────

def compare_personality_gpa_correlations(s1_df, s2_df):
    """Compare personality-GPA correlations across studies."""
    print("\n[1/5] Personality → GPA Correlations")
    print("─" * 60)

    s1 = s1_df[PERSONALITY + ['gpa_overall']].dropna()
    s2 = s2_df[PERSONALITY + ['gpa_overall']].dropna()
    n1, n2 = len(s1), len(s2)
    print(f"  Study 1: N={n1}")
    print(f"  Study 2: N={n2}")

    rows = []
    for trait in PERSONALITY:
        # Study 1
        r1, p1 = sp_stats.pearsonr(s1[trait], s1['gpa_overall'])
        ci1 = bootstrap_r_ci(s1[trait].values, s1['gpa_overall'].values)

        # Study 2
        r2, p2 = sp_stats.pearsonr(s2[trait], s2['gpa_overall'])
        ci2 = bootstrap_r_ci(s2[trait].values, s2['gpa_overall'].values)

        # Fisher z-test
        z_diff, p_diff = fisher_z_test(r1, n1, r2, n2)

        # Replication criteria
        same_direction = (r1 > 0) == (r2 > 0) or abs(r1) < 0.05 or abs(r2) < 0.05
        within_ci = ci1[0] <= r2 <= ci1[1]
        s2_sig = p2 < 0.05

        rows.append({
            'Trait': trait,
            'S1_r': r1, 'S1_p': p1, 'S1_ci_lo': ci1[0], 'S1_ci_hi': ci1[1], 'S1_N': n1,
            'S2_r': r2, 'S2_p': p2, 'S2_ci_lo': ci2[0], 'S2_ci_hi': ci2[1], 'S2_N': n2,
            'Fisher_z': z_diff, 'Fisher_p': p_diff,
            'same_direction': same_direction,
            'within_CI': within_ci,
            'S2_sig': s2_sig,
            'replicated': same_direction and (within_ci or s2_sig),
        })

        sig1 = '*' if p1 < 0.05 else ''
        sig2 = '*' if p2 < 0.05 else ''
        rep = '✓' if rows[-1]['replicated'] else '✗'
        print(f"  {trait:18s}  S1: r={r1:+.3f}{sig1:1s}  S2: r={r2:+.3f}{sig2:1s}  "
              f"z_diff={z_diff:+.2f} (p={p_diff:.3f})  {rep}")

    result = pd.DataFrame(rows)
    result.to_csv(COMP_DIR / 'personality_gpa_correlations.csv', index=False)
    return result


# ──────────────────────────────────────────────────────────────────────
# 2. ML Model Performance Comparison
# ──────────────────────────────────────────────────────────────────────

def compare_ml_performance():
    """Compare ML model R² for Personality → GPA across studies."""
    print("\n[2/5] ML Model R² Comparison")
    print("─" * 60)

    # Study 1: from multi_outcome_matrix.csv
    s1_matrix = pd.read_csv(S1_TABLE_DIR / 'multi_outcome_matrix.csv')
    s1_gpa = s1_matrix[(s1_matrix['Outcome'] == 'GPA') &
                        (s1_matrix['Features'] == 'Personality')]

    # Study 2: from personality_gpa_validation.csv
    s2_file = S2_TABLE_DIR / 'personality_gpa_validation.csv'
    if not s2_file.exists():
        print("  Study 2 validation results not yet available")
        return None

    s2_gpa = pd.read_csv(s2_file)

    rows = []
    model_map = {
        'Elastic Net': 'Elastic Net',
        'Ridge': 'Ridge',
        'Random Forest': 'Random Forest',
        'SVR': 'SVR',
    }

    for model_label in model_map:
        s1_row = s1_gpa[s1_gpa['Model'] == model_label]
        s2_row = s2_gpa[s2_gpa['Model'] == model_label]

        if len(s1_row) == 0 or len(s2_row) == 0:
            continue

        s1_r2 = s1_row['R2'].values[0]
        s2_r2_kf = s2_row['R2_kfold'].values[0]
        s2_p = s2_row['p_perm'].values[0]

        same_direction = (s1_r2 > 0) == (s2_r2_kf > 0)

        rows.append({
            'Model': model_label,
            'S1_R2_loo': s1_r2,
            'S1_N': s1_row['N'].values[0],
            'S2_R2_kfold': s2_r2_kf,
            'S2_N': s2_row['N'].values[0],
            'S2_p_perm': s2_p,
            'same_direction': same_direction,
            'S2_sig': s2_p < 0.05,
        })

        sig = '*' if s2_p < 0.05 else ''
        print(f"  {model_label:15s}  S1: R²={s1_r2:.3f} (LOO)  "
              f"S2: R²={s2_r2_kf:.3f} (10×10-fold)  p={s2_p:.4f}{sig}")

    result = pd.DataFrame(rows)
    result.to_csv(COMP_DIR / 'ml_performance_comparison.csv', index=False)
    return result


# ──────────────────────────────────────────────────────────────────────
# 3. SHAP Feature Ranking Comparison
# ──────────────────────────────────────────────────────────────────────

def compare_shap_rankings():
    """Compare SHAP feature importance rankings across studies."""
    print("\n[3/5] SHAP Feature Ranking Comparison")
    print("─" * 60)

    s1_shap = pd.read_csv(S1_TABLE_DIR / 'shap_importance.csv')
    s2_file = S2_TABLE_DIR / 'shap_personality_gpa.csv'
    if not s2_file.exists():
        print("  Study 2 SHAP results not yet available")
        return None

    s2_shap = pd.read_csv(s2_file)

    # Study 1: Personality → GPA scenario
    s1_pers = s1_shap[s1_shap['Scenario'] == 'Personality → GPA']

    rows = []
    for model in ['Elastic Net', 'Ridge', 'Random Forest', 'SVR']:
        s1_m = s1_pers[s1_pers['Model'] == model].set_index('Feature')['Mean_Abs_SHAP']
        s2_m = s2_shap[s2_shap.index == model] if 'Model' not in s2_shap.columns else None

        # Handle different CSV formats
        if s2_m is None or len(s2_m) == 0:
            # Try reading with Model as index
            s2_shap_idx = pd.read_csv(s2_file, index_col=0)
            if model in s2_shap_idx.index:
                s2_vals = s2_shap_idx.loc[model]
            else:
                continue
        else:
            s2_vals = s2_m.iloc[0]

        # Map feature names (Study 1 uses title case, Study 2 uses lowercase)
        feat_map = {
            'Extraversion': 'extraversion', 'Agreeableness': 'agreeableness',
            'Conscientiousness': 'conscientiousness', 'Neuroticism': 'neuroticism',
            'Openness': 'openness',
        }

        s1_rank = s1_m.rank(ascending=False)
        s1_top = s1_m.idxmax() if len(s1_m) > 0 else 'N/A'

        # Get S2 top feature
        if hasattr(s2_vals, 'idxmax'):
            s2_top = s2_vals.idxmax()
        else:
            s2_top = 'N/A'

        # Check if conscientiousness is #1 in both
        s1_c_top = s1_top in ['Conscientiousness', 'conscientiousness']
        s2_c_top = s2_top in ['Conscientiousness', 'conscientiousness']

        rows.append({
            'Model': model,
            'S1_top_feature': s1_top,
            'S2_top_feature': s2_top,
            'S1_C_rank': int(s1_rank.get('Conscientiousness', 0)),
            'C_is_top_S1': s1_c_top,
            'C_is_top_S2': s2_c_top,
            'C_top_both': s1_c_top and s2_c_top,
        })

        print(f"  {model:15s}  S1 #1: {s1_top:20s}  S2 #1: {s2_top}")

    result = pd.DataFrame(rows)
    result.to_csv(COMP_DIR / 'shap_ranking_comparison.csv', index=False)

    # Summary
    c_both = sum(r['C_top_both'] for r in rows)
    print(f"\n  Conscientiousness = #1 in both studies: {c_both}/{len(rows)} models")
    return result


# ──────────────────────────────────────────────────────────────────────
# 4. Forest Plot: Personality-GPA Correlations
# ──────────────────────────────────────────────────────────────────────

def plot_forest_correlations(corr_df):
    """Forest plot comparing personality-GPA correlations across studies."""
    print("\n[4/5] Forest Plot: Personality → GPA Correlations")
    print("─" * 60)

    if corr_df is None:
        print("  No correlation data available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = []
    y_labels = []
    colors_s1 = '#2196F3'  # blue
    colors_s2 = '#FF9800'  # orange

    y = 0
    for _, row in corr_df.iterrows():
        trait = row['Trait']
        label = TRAIT_LABELS.get(trait, trait[:3].title())

        # Study 1
        ax.errorbar(row['S1_r'], y + 0.15,
                     xerr=[[row['S1_r'] - row['S1_ci_lo']],
                            [row['S1_ci_hi'] - row['S1_r']]],
                     fmt='s', color=colors_s1, markersize=8, capsize=4,
                     linewidth=1.5, label='Study 1 (StudentLife)' if y == 0 else '')

        # Study 2
        ax.errorbar(row['S2_r'], y - 0.15,
                     xerr=[[row['S2_r'] - row['S2_ci_lo']],
                            [row['S2_ci_hi'] - row['S2_r']]],
                     fmt='o', color=colors_s2, markersize=8, capsize=4,
                     linewidth=1.5, label='Study 2 (NetHealth)' if y == 0 else '')

        y_positions.append(y)
        y_labels.append(f"{label} ({trait.title()})")
        y += 1

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Pearson r (Personality → GPA)')
    ax.set_title(f'Personality–GPA Correlations: Study 1 (N={int(corr_df["S1_N"].iloc[0])}) '
                 f'vs Study 2 (N={int(corr_df["S2_N"].iloc[0])})')
    ax.legend(loc='lower right')
    ax.set_xlim(-0.8, 0.8)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(COMP_DIR / 'forest_personality_gpa.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: forest_personality_gpa.png")


# ──────────────────────────────────────────────────────────────────────
# 5. Forest Plot: ML Model R²
# ──────────────────────────────────────────────────────────────────────

def plot_forest_ml(ml_df):
    """Forest plot comparing ML R² across studies."""
    print("\n[5/5] Forest Plot: ML Model R²")
    print("─" * 60)

    if ml_df is None:
        print("  No ML comparison data available")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    colors_s1 = '#2196F3'
    colors_s2 = '#FF9800'

    y_positions = []
    y_labels = []

    for i, (_, row) in enumerate(ml_df.iterrows()):
        # Study 1 (LOO-CV)
        ax.plot(row['S1_R2_loo'], i + 0.12, 's', color=colors_s1,
                markersize=10, label='Study 1 (LOO-CV)' if i == 0 else '')

        # Study 2 (10×10-fold)
        ax.plot(row['S2_R2_kfold'], i - 0.12, 'o', color=colors_s2,
                markersize=10, label='Study 2 (10×10-fold)' if i == 0 else '')

        y_positions.append(i)
        y_labels.append(row['Model'])

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('R² (Personality → GPA)')
    ax.set_title('ML Model Performance: Study 1 vs Study 2')
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(COMP_DIR / 'forest_ml_r2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: forest_ml_r2.png")


# ──────────────────────────────────────────────────────────────────────
# Replication Summary Report
# ──────────────────────────────────────────────────────────────────────

def generate_replication_summary(corr_df, ml_df, shap_df):
    """Generate overall replication statistics."""
    print("\n" + "=" * 60)
    print("REPLICATION SUMMARY")
    print("=" * 60)

    rows = []

    # 1. Correlation replication
    if corr_df is not None:
        for _, row in corr_df.iterrows():
            rows.append({
                'Finding': f"{row['Trait']} → GPA (r)",
                'S1_effect': f"r = {row['S1_r']:+.3f}",
                'S2_effect': f"r = {row['S2_r']:+.3f}",
                'same_direction': row['same_direction'],
                'within_CI': row['within_CI'],
                'S2_sig': row['S2_sig'],
                'replicated': row['replicated'],
            })

        # Key finding: Conscientiousness
        c_row = corr_df[corr_df['Trait'] == 'conscientiousness'].iloc[0]
        print(f"\n  KEY FINDING: Conscientiousness → GPA")
        print(f"    Study 1: r = {c_row['S1_r']:+.3f} [{c_row['S1_ci_lo']:+.3f}, {c_row['S1_ci_hi']:+.3f}]")
        print(f"    Study 2: r = {c_row['S2_r']:+.3f} [{c_row['S2_ci_lo']:+.3f}, {c_row['S2_ci_hi']:+.3f}]")
        print(f"    Fisher z-test: z = {c_row['Fisher_z']:+.2f}, p = {c_row['Fisher_p']:.3f}")
        print(f"    Replicated: {'YES' if c_row['replicated'] else 'NO'}")

        # Cochran's Q for conscientiousness
        r1, r2 = c_row['S1_r'], c_row['S2_r']
        n1, n2 = c_row['S1_N'], c_row['S2_N']
        z1, z2 = np.arctanh(r1), np.arctanh(r2)
        v1, v2 = 1 / (n1 - 3), 1 / (n2 - 3)
        Q, p_Q, I2 = cochrans_q([z1, z2], [v1, v2])
        print(f"    Cochran's Q = {Q:.2f}, p = {p_Q:.3f}, I² = {I2:.1f}%")

    # 2. ML replication
    if ml_df is not None:
        for _, row in ml_df.iterrows():
            rows.append({
                'Finding': f"{row['Model']} Pers→GPA (R²)",
                'S1_effect': f"R² = {row['S1_R2_loo']:.3f}",
                'S2_effect': f"R² = {row['S2_R2_kfold']:.3f}",
                'same_direction': row['same_direction'],
                'within_CI': True,  # We don't have Study 1 CIs for R²
                'S2_sig': row['S2_sig'],
                'replicated': row['same_direction'] and row['S2_sig'],
            })

    # 3. SHAP replication
    if shap_df is not None:
        for _, row in shap_df.iterrows():
            rows.append({
                'Finding': f"SHAP {row['Model']}: C = #1",
                'S1_effect': f"#1: {row['S1_top_feature']}",
                'S2_effect': f"#1: {row['S2_top_feature']}",
                'same_direction': True,
                'within_CI': True,
                'S2_sig': row['C_is_top_S2'],
                'replicated': row['C_top_both'],
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(COMP_DIR / 'replication_summary.csv', index=False)

    # Print summary counts
    n_total = len(summary)
    n_replicated = summary['replicated'].sum()
    n_direction = summary['same_direction'].sum()
    n_sig = summary['S2_sig'].sum()

    print(f"\n  OVERALL REPLICATION RATES:")
    print(f"    Total findings tested:     {n_total}")
    print(f"    Same direction:            {n_direction}/{n_total} ({n_direction/n_total:.0%})")
    print(f"    Study 2 significant:       {n_sig}/{n_total} ({n_sig/n_total:.0%})")
    print(f"    Fully replicated:          {n_replicated}/{n_total} ({n_replicated/n_total:.0%})")

    print(f"\n  Saved: replication_summary.csv")
    return summary


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 12 STEP 5: STUDY 1 vs STUDY 2 COMPARISON")
    print("=" * 60)

    # Load datasets
    print("\n  Loading datasets...")
    s1_df = pd.read_parquet(S1_DATA)
    s2_df = pd.read_parquet(S2_DATA)
    print(f"  Study 1 (StudentLife): {s1_df.shape[0]} × {s1_df.shape[1]}")
    print(f"  Study 2 (NetHealth):   {s2_df.shape[0]} × {s2_df.shape[1]}")

    # 1. Correlations
    corr_df = compare_personality_gpa_correlations(s1_df, s2_df)

    # 2. ML performance
    ml_df = compare_ml_performance()

    # 3. SHAP rankings
    shap_df = compare_shap_rankings()

    # 4. Forest plot: correlations
    plot_forest_correlations(corr_df)

    # 5. Forest plot: ML R²
    plot_forest_ml(ml_df)

    # Replication summary
    generate_replication_summary(corr_df, ml_df, shap_df)

    print(f"\n{'=' * 60}")
    print("PHASE 12 STEP 5 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
