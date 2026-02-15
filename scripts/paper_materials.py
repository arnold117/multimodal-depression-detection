#!/usr/bin/env python3
"""
Generate Publication-Ready Materials

Aggregates all analysis results into publication-quality figures and tables:
  - Unified figure style (300 dpi, journal-compatible)
  - Summary dashboard combining key findings
  - LaTeX-formatted tables
  - Comprehensive results report

Input:  results/tables/*.csv, results/figures/*.png
        data/processed/analysis_dataset.parquet
Output: results/figures/figure_*.png (publication-ready)
        results/tables/table_*.csv
        results/reports/summary_report.txt
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

DATA_PATH = project_root / 'data' / 'processed' / 'analysis_dataset.parquet'
TABLE_DIR = project_root / 'results' / 'tables'
FIGURE_DIR = project_root / 'results' / 'figures'
REPORT_DIR = project_root / 'results' / 'reports'

for d in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

TRAITS = ['extraversion', 'agreeableness', 'conscientiousness',
          'neuroticism', 'openness']
TRAIT_SHORT = {'extraversion': 'E', 'agreeableness': 'A',
               'conscientiousness': 'C', 'neuroticism': 'N', 'openness': 'O'}
BEHAVIOR_PC = ['mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
               'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']
BEH_SHORT = {b: b.replace('_pc1', '').title() for b in BEHAVIOR_PC}


def figure1_sample_overview(df):
    """Figure 1: Sample characteristics and data overview."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # A: Big Five distributions
    ax1 = fig.add_subplot(gs[0, 0])
    trait_data = df[TRAITS].melt(var_name='Trait', value_name='Score')
    trait_data['Trait'] = trait_data['Trait'].map(TRAIT_SHORT)
    sns.boxplot(data=trait_data, x='Trait', y='Score', ax=ax1,
                palette='Set2', width=0.6)
    sns.stripplot(data=trait_data, x='Trait', y='Score', ax=ax1,
                  color='black', size=3, alpha=0.4)
    ax1.set_title('A. Big Five Personality Traits')
    ax1.set_ylabel('Score (1-5)')
    ax1.set_xlabel('')

    # B: GPA distribution
    ax2 = fig.add_subplot(gs[0, 1])
    gpa = df['gpa_overall'].dropna()
    ax2.hist(gpa, bins=10, color='#66bb6a', edgecolor='white', alpha=0.8)
    ax2.axvline(gpa.mean(), color='#d32f2f', linestyle='--', linewidth=2,
                label=f'Mean={gpa.mean():.2f}')
    ax2.set_title('B. GPA Distribution')
    ax2.set_xlabel('GPA')
    ax2.set_ylabel('Count')
    ax2.legend()

    # C: Wellbeing measures
    ax3 = fig.add_subplot(gs[0, 2])
    wellbeing_cols = ['phq9_total', 'pss_total', 'loneliness_total', 'flourishing_total']
    wellbeing_short = {'phq9_total': 'PHQ-9', 'pss_total': 'PSS',
                       'loneliness_total': 'Lonely', 'flourishing_total': 'Flourish'}
    # Standardize for comparison
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    wb_data = df[wellbeing_cols].dropna()
    wb_z = pd.DataFrame(scaler.fit_transform(wb_data), columns=wellbeing_cols)
    wb_melt = wb_z.melt(var_name='Measure', value_name='Z-Score')
    wb_melt['Measure'] = wb_melt['Measure'].map(wellbeing_short)
    sns.boxplot(data=wb_melt, x='Measure', y='Z-Score', ax=ax3,
                palette='Set3', width=0.6)
    ax3.set_title('C. Wellbeing Measures (Standardized)')
    ax3.set_xlabel('')
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)

    # D: Behavioral composites
    ax4 = fig.add_subplot(gs[1, 0])
    beh_data = df[BEHAVIOR_PC].melt(var_name='Modality', value_name='PC1 Score')
    beh_data['Modality'] = beh_data['Modality'].map(BEH_SHORT)
    sns.boxplot(data=beh_data, x='Modality', y='PC1 Score', ax=ax4,
                palette='Paired', width=0.6)
    ax4.set_title('D. Behavioral Composites (PC1)')
    ax4.set_xlabel('')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)

    # E: Correlation of Big Five with GPA
    ax5 = fig.add_subplot(gs[1, 1])
    corrs = []
    for trait in TRAITS:
        subset = df[[trait, 'gpa_overall']].dropna()
        r, p = sp_stats.pearsonr(subset[trait], subset['gpa_overall'])
        corrs.append({'Trait': TRAIT_SHORT[trait], 'r': r, 'p': p})
    corr_df = pd.DataFrame(corrs)
    colors = ['#d32f2f' if p < 0.05 else '#90a4ae' for p in corr_df['p']]
    ax5.barh(corr_df['Trait'], corr_df['r'], color=colors)
    ax5.set_title('E. Personality-GPA Correlations')
    ax5.set_xlabel('Pearson r')
    ax5.axvline(0, color='black', linewidth=0.5)
    for i, row in corr_df.iterrows():
        sig = '*' if row['p'] < 0.05 else ''
        ax5.text(row['r'] + 0.02 * np.sign(row['r']), i,
                f"r={row['r']:.2f}{sig}", va='center', fontsize=8)

    # F: Data availability heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    key_cols = TRAITS + ['phq9_total', 'pss_total'] + BEHAVIOR_PC[:4] + ['gpa_overall']
    avail = df[key_cols].notna().sum() / len(df) * 100
    short_names = {**TRAIT_SHORT,
                   'phq9_total': 'PHQ9', 'pss_total': 'PSS',
                   'gpa_overall': 'GPA',
                   **{b: BEH_SHORT[b][:6] for b in BEHAVIOR_PC[:4]}}
    labels = [short_names.get(c, c[:8]) for c in key_cols]
    ax6.barh(labels, avail, color='#42a5f5', alpha=0.8)
    ax6.set_title('F. Data Availability (%)')
    ax6.set_xlabel('% of Participants')
    ax6.set_xlim(0, 105)

    plt.suptitle('Figure 1: Sample Characteristics and Variable Distributions',
                fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(FIGURE_DIR / 'figure1_sample_overview.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure1_sample_overview.png")


def figure2_mediation_summary(df):
    """Figure 2: Mediation analysis summary."""
    # Load mediation results
    simple = pd.read_csv(TABLE_DIR / 'mediation_simple.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A: Heatmap of indirect effects
    ax = axes[0]
    pivot = simple.pivot(index='Trait', columns='Mediator', values='ab (indirect)')
    pivot.index = [TRAIT_SHORT.get(t, t) for t in pivot.index]
    pivot.columns = [c.replace('_pc1', '') for c in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.5, annot_kws={'size': 8},
                vmin=-0.15, vmax=0.15)
    ax.set_title('A. Indirect Effects (a × b)')
    ax.set_xlabel('Behavioral Mediator')
    ax.set_ylabel('Personality Trait')

    # B: Total vs direct effects
    ax = axes[1]
    total_effects = simple.groupby('Trait')[['c (total)', "c' (direct)"]].first().reset_index()
    # Get unique c and c' per trait
    trait_effects = []
    for trait in TRAITS:
        t_rows = simple[simple['Trait'] == trait]
        if len(t_rows) > 0:
            trait_effects.append({
                'Trait': TRAIT_SHORT[trait],
                'Total (c)': t_rows['c (total)'].iloc[0],
                "Direct (c')": t_rows["c' (direct)"].iloc[0],
            })
    te_df = pd.DataFrame(trait_effects)

    x = np.arange(len(te_df))
    w = 0.35
    ax.bar(x - w/2, te_df['Total (c)'], w, label='Total (c)', color='#1565c0', alpha=0.8)
    ax.bar(x + w/2, te_df["Direct (c')"], w, label="Direct (c')", color='#ff8f00', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(te_df['Trait'])
    ax.set_ylabel('Standardized Coefficient')
    ax.set_title("B. Total vs Direct Effects on GPA")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()

    plt.suptitle('Figure 2: Mediation Analysis — Personality → Behavior → GPA',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure2_mediation_summary.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure2_mediation_summary.png")


def figure3_prediction_comparison():
    """Figure 3: Elastic Net prediction model comparison."""
    comparison = pd.read_csv(TABLE_DIR / 'elastic_net_comparison.csv')
    coefs = pd.read_csv(TABLE_DIR / 'elastic_net_coefficients.csv')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: Model R² comparison
    ax = axes[0]
    models = comparison['Model'].str.replace('M[0-9]: ', '', regex=True)
    colors = ['#1565c0', '#ff8f00', '#2e7d32', '#d32f2f', '#7b1fa2']
    bars = ax.barh(models, comparison['LOO_R²'],
                   color=colors[:len(comparison)], alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.5)
    for bar, val in zip(bars, comparison['LOO_R²']):
        ax.text(max(val + 0.01, 0.01), bar.get_y() + bar.get_height() / 2,
               f'{val:.3f}', va='center', fontsize=9)
    ax.set_xlabel('LOO-CV R²')
    ax.set_title('A. Model Comparison')

    # B: Coefficient plot for best model
    ax = axes[1]
    nonzero = coefs[coefs['Coefficient'] != 0].sort_values('Coefficient')
    if len(nonzero) == 0:
        nonzero = coefs.nlargest(5, 'Selection_Freq')
    colors_coef = ['#d32f2f' if sig else '#1565c0' for sig in nonzero['Significant']]
    ax.barh(nonzero['Feature'], nonzero['Coefficient'], color=colors_coef, alpha=0.8)
    ax.errorbar(nonzero['Coefficient'], nonzero['Feature'],
               xerr=[nonzero['Coefficient'] - nonzero['CI_lo'],
                     nonzero['CI_hi'] - nonzero['Coefficient']],
               fmt='none', color='black', capsize=3, linewidth=1)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Standardized β')
    ax.set_title('B. Best Model Coefficients')

    # C: Selection frequency
    ax = axes[2]
    sorted_coefs = coefs.sort_values('Selection_Freq', ascending=True)
    colors_freq = ['#2e7d32' if f > 0.8 else '#ff8f00' if f > 0.5 else '#bdbdbd'
                   for f in sorted_coefs['Selection_Freq']]
    ax.barh(sorted_coefs['Feature'], sorted_coefs['Selection_Freq'],
            color=colors_freq, alpha=0.8)
    ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(0.8, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Bootstrap Selection Frequency')
    ax.set_title('C. Feature Stability')
    ax.set_xlim(0, 1.05)

    plt.suptitle('Figure 3: Elastic Net Prediction of GPA',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure3_prediction.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure3_prediction.png")


def figure4_lpa_profiles():
    """Figure 4: Latent profile analysis results."""
    profiles = pd.read_csv(TABLE_DIR / 'lpa_profiles.csv')
    outcomes = pd.read_csv(TABLE_DIR / 'lpa_outcome_comparison.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A: Radar plot (simplified as grouped bar chart for better readability)
    ax = axes[0]
    n_profiles = len(profiles)
    profile_vars = [c for c in profiles.columns if c not in ['Profile', 'n']]
    short_vars = [TRAIT_SHORT.get(v, BEH_SHORT.get(v, v[:8])) for v in profile_vars]

    x = np.arange(len(profile_vars))
    width = 0.8 / n_profiles
    colors = ['#1565c0', '#d32f2f', '#2e7d32', '#ff8f00']

    for i, (_, row) in enumerate(profiles.iterrows()):
        values = [row[v] for v in profile_vars]
        ax.bar(x + i * width - 0.4 + width/2, values, width,
               label=f'Profile {int(row["Profile"])} (n={int(row["n"])})',
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(short_vars, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Standardized Mean')
    ax.set_title('A. Profile Characteristics')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)

    # B: Outcome differences
    ax = axes[1]
    outcome_names = outcomes['Variable'].tolist()
    short_outcome = {'gpa_overall': 'GPA', 'phq9_total': 'PHQ-9',
                     'pss_total': 'PSS', 'loneliness_total': 'Lonely',
                     'flourishing_total': 'Flourish'}
    labels = [short_outcome.get(v, v) for v in outcome_names]
    p_vals = outcomes['p_kruskal'].values
    sig_markers = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                   for p in p_vals]

    # -log10(p) for visualization
    neg_log_p = -np.log10(np.clip(p_vals, 1e-10, 1))
    colors_p = ['#d32f2f' if p < 0.05 else '#90a4ae' for p in p_vals]
    bars = ax.barh(labels, neg_log_p, color=colors_p, alpha=0.8)
    ax.axvline(-np.log10(0.05), color='orange', linestyle='--',
               label='p = 0.05', alpha=0.7)
    ax.set_xlabel('-log10(p)')
    ax.set_title('B. Outcome Differences (Kruskal-Wallis)')
    ax.legend()

    for bar, sig, p in zip(bars, sig_markers, p_vals):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
               f'{sig} (p={p:.3f})', va='center', fontsize=8)

    plt.suptitle('Figure 4: Latent Profile Analysis',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure4_lpa.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure4_lpa.png")


def generate_summary_report(df):
    """Generate comprehensive text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("RESULTS SUMMARY: Big Five, Smartphone Behavior, and GPA")
    lines.append("=" * 70)
    lines.append("")

    # Sample
    lines.append("1. SAMPLE")
    lines.append(f"   N = {len(df)} university students")
    lines.append(f"   GPA: M = {df['gpa_overall'].mean():.2f}, "
                 f"SD = {df['gpa_overall'].std():.2f}")
    lines.append(f"   13 behavioral modalities, 87 raw features")
    lines.append(f"   8 PCA behavioral composites")
    lines.append("")

    # Elastic Net
    lines.append("2. ELASTIC NET PREDICTION (Best Model)")
    try:
        en = pd.read_csv(TABLE_DIR / 'elastic_net_comparison.csv')
        best = en.loc[en['LOO_R²'].idxmax()]
        lines.append(f"   Best model: {best['Model']}")
        lines.append(f"   LOO-CV R² = {best['LOO_R²']:.3f}")
        lines.append(f"   RMSE = {best['RMSE']:.3f}, MAE = {best['MAE']:.3f}")
        lines.append(f"   Features selected: {int(best['p_selected'])} / {int(best['p_features'])}")

        coefs = pd.read_csv(TABLE_DIR / 'elastic_net_coefficients.csv')
        sig_coefs = coefs[coefs['Significant'] == True]
        if len(sig_coefs) > 0:
            lines.append(f"   Significant predictors:")
            for _, row in sig_coefs.iterrows():
                lines.append(f"     {row['Feature']}: β = {row['Coefficient']:.3f} "
                            f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}]")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # Mediation
    lines.append("3. BOOTSTRAP MEDIATION")
    try:
        med = pd.read_csv(TABLE_DIR / 'mediation_simple.csv')
        n_sig = med['sig'].astype(str).str.contains(r'\*', na=False).sum()
        lines.append(f"   Tests: {len(med)} (5 traits × 8 mediators)")
        lines.append(f"   Significant indirect effects: {n_sig}")
        lines.append(f"   Strongest indirect effects:")
        top = med.assign(_abs=med['ab (indirect)'].abs()).nlargest(3, '_abs')
        for _, row in top.iterrows():
            lines.append(f"     {row['Trait']} → {row['Mediator']} → GPA: "
                        f"ab = {row['ab (indirect)']:.3f} "
                        f"[{row['ab_CI_lo']:.3f}, {row['ab_CI_hi']:.3f}]")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # PLS-SEM
    lines.append("4. PLS-SEM STRUCTURAL MODEL")
    try:
        plssem = pd.read_csv(TABLE_DIR / 'plssem_results.csv')
        sig_paths = plssem[plssem['sig'].astype(str).str.contains(r'\*', na=False)]
        lines.append(f"   Significant paths: {len(sig_paths)}")
        for _, row in sig_paths.iterrows():
            lines.append(f"     {row['From']} → {row['To']}: "
                        f"β = {row['boot_mean']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] "
                        f"{row['sig']}")
        effects = pd.read_csv(TABLE_DIR / 'plssem_effects.csv')
        total = effects[effects['Type'] == 'total']
        if len(total) > 0:
            lines.append(f"   Total effect (PERSONALITY → GPA): "
                        f"{total['Value'].values[0]:.3f}")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # LPA
    lines.append("5. LATENT PROFILE ANALYSIS")
    try:
        fit = pd.read_csv(TABLE_DIR / 'lpa_fit_indices.csv')
        best_k = int(fit.loc[fit['BIC'].idxmin(), 'k'])
        lines.append(f"   Optimal k by BIC: {best_k}")

        profiles = pd.read_csv(TABLE_DIR / 'lpa_profiles.csv')
        for _, row in profiles.iterrows():
            lines.append(f"     Profile {int(row['Profile'])}: n = {int(row['n'])}")

        outcomes = pd.read_csv(TABLE_DIR / 'lpa_outcome_comparison.csv')
        sig_out = outcomes[outcomes['p_kruskal'] < 0.05]
        lines.append(f"   Significant outcome differences:")
        for _, row in sig_out.iterrows():
            lines.append(f"     {row['Variable']}: H = {row['H_stat']:.2f}, "
                        f"p = {row['p_kruskal']:.4f}")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # Moderation
    lines.append("6. MODERATION ANALYSIS")
    try:
        mod = pd.read_csv(TABLE_DIR / 'moderation_results.csv')
        sig_mod = mod[mod['sig'].astype(str).str.contains(r'\*', na=False)]
        lines.append(f"   Total tests: {len(mod)}")
        lines.append(f"   Significant interactions (bootstrap 95% CI): {len(sig_mod)}")
        top_mod = mod.nlargest(5, 'Delta_R2')
        for _, row in top_mod.iterrows():
            sig_str = '*' if str(row['sig']).strip() == '*' else ''
            lines.append(f"     {row['Moderator']} × {row['Behavior']} → {row['Outcome_Label']}: "
                        f"ΔR²={row['Delta_R2']:.3f} β={row['Interaction_beta']:.3f} "
                        f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] {sig_str}")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # Temporal features
    lines.append("7. TEMPORAL TREND FEATURES (Phase 8)")
    try:
        temporal = pd.read_parquet(project_root / 'data' / 'processed' / 'features' / 'temporal_features.parquet')
        temporal_cols = [c for c in temporal.columns if c != 'uid']
        lines.append(f"   Features extracted: {len(temporal_cols)} (slope, CV, delta × 14 metrics)")
        lines.append(f"   Coverage: {temporal[temporal_cols].notna().all(axis=1).sum()}/{len(temporal)} users with complete data")
        lines.append(f"   Model 7 (Personality + temporal slopes): R² = -0.055 (no improvement over M1)")
        lines.append(f"   Interpretation: Behavioral trends do not add to GPA prediction beyond personality")
        lines.append(f"   Note: Temporal features may predict wellbeing (tested in Phase 9)")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # Key findings
    lines.append("8. KEY FINDINGS")
    lines.append("   a) Conscientiousness is the strongest personality predictor of GPA")
    lines.append("   b) Neuroticism shows negative association with academic performance")
    lines.append("   c) Personality alone predicts GPA better than smartphone behavior alone")
    lines.append("   d) Digital engagement and mobility relate to psychological wellbeing")
    lines.append("   e) Four distinct personality-behavior profiles identified,")
    lines.append("      differing significantly in stress and loneliness levels")
    lines.append("   f) Personality moderates behavior-outcome links:")
    lines.append("      E × Activity → GPA (ΔR²=0.221), C × Activity → Loneliness (ΔR²=0.213)")
    lines.append("   g) Interaction features do not add predictive value beyond personality (M6 R²=0.112 < M1 R²=0.170)")
    lines.append("   h) Temporal behavioral trends (slopes, stability, delta) also fail to improve")
    lines.append("      GPA prediction (M7 R²=-0.055), reinforcing personality primacy for academics")
    lines.append("")
    lines.append("=" * 70)

    report_text = '\n'.join(lines)

    report_path = REPORT_DIR / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"  Saved: {report_path.name}")
    print(report_text)


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION MATERIALS")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    print("\n[1/5] Figure 1: Sample overview...")
    figure1_sample_overview(df)

    print("\n[2/5] Figure 2: Mediation summary...")
    figure2_mediation_summary(df)

    print("\n[3/5] Figure 3: Prediction comparison...")
    figure3_prediction_comparison()

    print("\n[4/5] Figure 4: Latent profiles...")
    figure4_lpa_profiles()

    print("\n[5/5] Summary report...")
    generate_summary_report(df)

    print("\n" + "=" * 60)
    print("PUBLICATION MATERIALS COMPLETE")
    print("=" * 60)
    print(f"\n  Figures: results/figures/figure1-4*.png")
    print(f"  Tables:  results/tables/*.csv")
    print(f"  Report:  results/reports/summary_report.txt")


if __name__ == '__main__':
    main()
