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


def figure5_multi_outcome(df):
    """Figure 5: Multi-outcome prediction — what do personality vs behavior predict?"""
    try:
        matrix = pd.read_csv(TABLE_DIR / 'multi_outcome_matrix.csv')
    except FileNotFoundError:
        print("  Skipped (multi_outcome_matrix.csv not found)")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    outcome_order = ['GPA', 'PHQ-9', 'PSS', 'Loneliness', 'Flourishing', 'PANAS-NA']
    feat_order = ['Personality', 'Behavior', 'Pers + Beh']

    # A: Heatmap — average R² across models
    ax = fig.add_subplot(gs[0, 0])
    avg = matrix.groupby(['Features', 'Outcome'])['R2'].mean().reset_index()
    heatmap = avg.pivot(index='Features', columns='Outcome', values='R2')
    heatmap = heatmap.reindex(index=feat_order,
                               columns=[c for c in outcome_order if c in heatmap.columns])
    sns.heatmap(heatmap, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=0.5, vmin=-0.3, vmax=0.5,
                annot_kws={'size': 9, 'fontweight': 'bold'})
    ax.set_title('A. Mean LOO-CV R² (Averaged Across 4 Models)')
    ax.set_ylabel('')
    ax.set_xlabel('')

    # B: Behavior vs Personality advantage
    ax = fig.add_subplot(gs[0, 1])
    beh_avg = matrix[matrix['Features'] == 'Behavior'].groupby('Outcome')['R2'].mean()
    pers_avg = matrix[matrix['Features'] == 'Personality'].groupby('Outcome')['R2'].mean()
    diff = (beh_avg - pers_avg).reindex([c for c in outcome_order
                                          if c in beh_avg.index and c in pers_avg.index])
    colors = ['#2e7d32' if v > 0 else '#d32f2f' for v in diff.values]
    ax.barh(diff.index, diff.values, color=colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('ΔR² (Behavior − Personality)')
    ax.set_title('B. Behavior Advantage Over Personality')
    for i, (name, val) in enumerate(diff.items()):
        label = 'Behavior better' if val > 0 else 'Personality better'
        ax.text(val + 0.01 * np.sign(val), i, f'{val:+.3f}', va='center', fontsize=8)

    # C: Best model per outcome
    ax = fig.add_subplot(gs[1, 0])
    best_per_outcome = matrix.loc[matrix.groupby('Outcome')['R2'].idxmax()]
    best_per_outcome = best_per_outcome.set_index('Outcome').reindex(
        [c for c in outcome_order if c in best_per_outcome['Outcome'].values])
    model_colors = {'Elastic Net': '#1565c0', 'Ridge': '#ff8f00',
                    'Random Forest': '#2e7d32', 'SVR': '#d32f2f'}
    bar_colors = [model_colors.get(m, '#90a4ae') for m in best_per_outcome['Model']]
    bars = ax.barh(best_per_outcome.index, best_per_outcome['R2'],
                   color=bar_colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('LOO-CV R²')
    ax.set_title('C. Best Prediction Per Outcome')
    for bar, (_, row) in zip(bars, best_per_outcome.iterrows()):
        ax.text(max(bar.get_width() + 0.01, 0.01),
               bar.get_y() + bar.get_height() / 2,
               f'{row["Model"]}, {row["Features"]}', va='center', fontsize=7)

    # D: Cross-model consistency (GPA, Personality)
    ax = fig.add_subplot(gs[1, 1])
    try:
        comp = pd.read_csv(TABLE_DIR / 'multi_model_comparison.csv')
        gpa_all = comp[comp['Features'] == 'Personality']
        models = gpa_all['Model'].values
        r2s = gpa_all['R2'].values
        p_vals = gpa_all['p_perm'].values
        colors_bar = ['#2e7d32' if p < 0.05 else '#90a4ae' for p in p_vals]
        bars = ax.barh(models, r2s, color=colors_bar, alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.5)
        for bar, p in zip(bars, p_vals):
            sig = '*' if p < 0.05 else ''
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                   f'p={p:.3f}{sig}', va='center', fontsize=8)
        ax.set_xlabel('LOO-CV R²')
        ax.set_title('D. GPA Prediction: Cross-Model Consistency\n(Personality features, *p < .05)')
    except FileNotFoundError:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')

    plt.suptitle('Figure 5: Multi-Outcome × Multi-Model Prediction\n'
                 'Personality predicts GPA; Behavior predicts mental health',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(FIGURE_DIR / 'figure5_multi_outcome.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure5_multi_outcome.png")


def figure6_effect_size_forest():
    """Figure 6: Forest plot of all key effect sizes."""
    try:
        effects = pd.read_csv(TABLE_DIR / 'effect_size_summary.csv')
    except FileNotFoundError:
        print("  Skipped (effect_size_summary.csv not found)")
        return

    # Select key effects (with CIs or important R² values)
    key_effects = []

    # Correlations
    corr = effects[effects['Analysis'] == 'Correlation (GPA)'].copy()
    if len(corr) > 0:
        for _, row in corr.iterrows():
            key_effects.append({
                'Label': f"r({row['Effect'][:5]}, GPA)",
                'Value': row['Value'],
                'CI_lo': row.get('CI_lo', np.nan),
                'CI_hi': row.get('CI_hi', np.nan),
                'Category': 'Correlation',
            })

    # PLS-SEM paths
    plssem = effects[effects['Analysis'] == 'PLS-SEM'].copy()
    for _, row in plssem.iterrows():
        key_effects.append({
            'Label': f"β({row['Effect']})",
            'Value': row['Value'],
            'CI_lo': row.get('CI_lo', np.nan),
            'CI_hi': row.get('CI_hi', np.nan),
            'Category': 'PLS-SEM',
        })

    # Moderation ΔR²
    mod = effects[effects['Analysis'] == 'Moderation'].copy()
    for _, row in mod.iterrows():
        eff = row['Effect']
        if len(eff) > 30:
            eff = eff[:30] + '...'
        key_effects.append({
            'Label': f"ΔR²({eff})",
            'Value': row['Value'],
            'CI_lo': row.get('CI_lo', np.nan),
            'CI_hi': row.get('CI_hi', np.nan),
            'Category': 'Moderation',
        })

    # Multi-model best R²
    multi = effects[effects['Analysis'] == 'Multi-model best'].copy()
    for _, row in multi.iterrows():
        eff = row['Effect']
        if len(eff) > 35:
            eff = eff[:35] + '...'
        key_effects.append({
            'Label': f"R²({eff})",
            'Value': row['Value'],
            'CI_lo': np.nan,
            'CI_hi': np.nan,
            'Category': 'Prediction',
        })

    if len(key_effects) == 0:
        print("  No effects to plot")
        return

    edf = pd.DataFrame(key_effects).sort_values('Value')

    fig, ax = plt.subplots(figsize=(12, max(8, len(edf) * 0.4)))

    cat_colors = {
        'Correlation': '#1565c0',
        'PLS-SEM': '#d32f2f',
        'Moderation': '#2e7d32',
        'Prediction': '#7b1fa2',
    }

    y_pos = np.arange(len(edf))
    colors = [cat_colors.get(row['Category'], '#90a4ae') for _, row in edf.iterrows()]

    ax.scatter(edf['Value'], y_pos, c=colors, s=80, zorder=5, edgecolors='white')

    for i, (_, row) in enumerate(edf.iterrows()):
        if pd.notna(row['CI_lo']) and pd.notna(row['CI_hi']):
            ax.hlines(y=i, xmin=row['CI_lo'], xmax=row['CI_hi'],
                     color=colors[i], linewidth=2.5, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(edf['Label'], fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Effect Size')
    ax.set_title('Figure 6: Effect Size Summary Across All Analyses\n'
                 '(95% Bootstrap CIs where available)',
                 fontsize=13, fontweight='bold')

    for cat, color in cat_colors.items():
        ax.scatter([], [], c=color, label=cat, s=60)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'figure6_effect_sizes.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure6_effect_sizes.png")


def figure7_shap_summary():
    """Figure 7: SHAP-based ML interpretability analysis."""
    shap_path = TABLE_DIR / 'shap_importance.csv'
    cross_model_path = TABLE_DIR / 'cross_model_importance.csv'

    if not shap_path.exists() or not cross_model_path.exists():
        print("  Skipped (SHAP or cross-model importance data not found)")
        return

    shap_df = pd.read_csv(shap_path)
    imp_df = pd.read_csv(cross_model_path, index_col=0)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # A: Cross-model feature importance heatmap for GPA
    ax = fig.add_subplot(gs[0, :])
    imp_sorted = imp_df.copy()
    imp_sorted['Mean'] = imp_sorted.mean(axis=1)
    imp_sorted = imp_sorted.sort_values('Mean', ascending=True).drop(columns='Mean')
    sns.heatmap(imp_sorted, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Normalized Importance'})
    ax.set_title('A. Cross-Model Feature Importance for GPA (Pers + Beh)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('')

    # B: SHAP top features — Personality → GPA
    ax = fig.add_subplot(gs[1, 0])
    pers_gpa = shap_df[shap_df['Scenario'] == 'Personality → GPA']
    if len(pers_gpa) > 0:
        # Average SHAP across models
        avg_shap = pers_gpa.groupby('Feature')['Mean_Abs_SHAP'].mean().sort_values(ascending=True)
        colors_bar = ['#1565c0'] * len(avg_shap)
        avg_shap.plot.barh(ax=ax, color=colors_bar, alpha=0.8)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('B. Personality → GPA\n(Mean across 4 models)', fontsize=11, fontweight='bold')

    # C: SHAP top features — Behavior → PHQ-9
    ax = fig.add_subplot(gs[1, 1])
    beh_phq = shap_df[shap_df['Scenario'] == 'Behavior → PHQ-9']
    if len(beh_phq) > 0:
        avg_shap = beh_phq.groupby('Feature')['Mean_Abs_SHAP'].mean().sort_values(ascending=True)
        colors_bar = ['#d32f2f'] * len(avg_shap)
        avg_shap.plot.barh(ax=ax, color=colors_bar, alpha=0.8)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('C. Behavior → PHQ-9\n(Mean across 4 models)', fontsize=11, fontweight='bold')

    plt.suptitle('Figure 7: ML Interpretability — SHAP Feature Attribution',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(FIGURE_DIR / 'figure7_shap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure7_shap.png")


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

    # Multi-outcome prediction
    lines.append("8. MULTI-OUTCOME PREDICTION (Phase 9)")
    try:
        matrix = pd.read_csv(TABLE_DIR / 'multi_outcome_matrix.csv')
        comparison = pd.read_csv(TABLE_DIR / 'multi_model_comparison.csv')

        lines.append(f"   Combinations tested: {len(matrix)} (4 models × 6 outcomes × 3 feature sets)")
        lines.append("")
        lines.append("   Best R² per outcome (across all models & feature sets):")
        for outcome in matrix['Outcome'].unique():
            sub = matrix[matrix['Outcome'] == outcome]
            best = sub.loc[sub['R2'].idxmax()]
            lines.append(f"     {outcome:15s} R²={best['R2']:.3f} ({best['Model']}, {best['Features']})")

        lines.append("")
        lines.append("   GPA cross-model validation (Personality features):")
        gpa_pers = comparison[comparison['Features'] == 'Personality']
        for _, row in gpa_pers.iterrows():
            sig = '*' if row['p_perm'] < 0.05 else ''
            lines.append(f"     {row['Model']:15s} R²={row['R2']:.3f} p={row['p_perm']:.3f} {sig}")

        lines.append("")
        lines.append("   LPA profiles as predictors:")
        lines.append(f"     Profile alone → GPA: R² = -0.067")
        lines.append(f"     Personality + Profile → GPA: ΔR² = -0.039 (no improvement)")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # ML Interpretability
    lines.append("9. ML INTERPRETABILITY (Phase 11)")
    try:
        shap_imp = pd.read_csv(TABLE_DIR / 'shap_importance.csv')
        cross_imp = pd.read_csv(TABLE_DIR / 'cross_model_importance.csv', index_col=0)

        lines.append("   SHAP Analysis:")
        # Top features for Personality → GPA
        pers_gpa = shap_imp[shap_imp['Scenario'] == 'Personality → GPA']
        if len(pers_gpa) > 0:
            avg = pers_gpa.groupby('Feature')['Mean_Abs_SHAP'].mean().sort_values(ascending=False)
            lines.append(f"     Personality → GPA (top 3 by mean |SHAP|):")
            for feat, val in avg.head(3).items():
                lines.append(f"       {feat}: {val:.4f}")

        # Top features for Behavior → PHQ-9
        beh_phq = shap_imp[shap_imp['Scenario'] == 'Behavior → PHQ-9']
        if len(beh_phq) > 0:
            avg = beh_phq.groupby('Feature')['Mean_Abs_SHAP'].mean().sort_values(ascending=False)
            lines.append(f"     Behavior → PHQ-9 (top 3 by mean |SHAP|):")
            for feat, val in avg.head(3).items():
                lines.append(f"       {feat}: {val:.4f}")

        lines.append("")
        lines.append("   Cross-Model Feature Importance (Pers+Beh → GPA):")
        mean_imp = cross_imp.mean(axis=1).sort_values(ascending=False)
        for feat, val in mean_imp.head(5).items():
            lines.append(f"     {feat}: {val:.3f}")
    except FileNotFoundError:
        lines.append("   (results not available)")
    lines.append("")

    # Key findings
    lines.append("10. KEY FINDINGS")
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
    lines.append("   i) Cross-model validation: Personality → GPA significant in 2/4 models")
    lines.append("      (Elastic Net p=0.010, Random Forest p=0.008); 5/12 GPA combos significant after Optuna tuning")
    lines.append("   j) Behavior features best predict PHQ-9 (R²=0.468), not GPA —")
    lines.append("      confirming differential prediction targets for personality vs behavior")
    lines.append("   k) LPA behavioral profiles have no incremental predictive value for GPA")
    lines.append("   l) Optuna Bayesian optimization improved RF Personality→GPA from R²=0.101 to R²=0.212")
    lines.append("   m) SVR captures non-linear Behavior→GPA (R²=0.116, p=0.032), missed by linear models")
    lines.append("   n) SHAP confirms Conscientiousness as #1 GPA predictor across all 4 models")
    lines.append("   o) Cross-model feature ranking agreement: mean Kendall's τ=0.460 (4/6 pairs significant)")
    lines.append("")

    # Narrative
    lines.append("11. PAPER NARRATIVE")
    lines.append("")
    lines.append("   Title (working): Personality Predicts Grades, Behavior Predicts Wellbeing:")
    lines.append("     An Exploratory Multi-Method Study of Smartphone Sensing in College Students")
    lines.append("")
    lines.append("   Core thesis: Smartphone behavioral sensing captures psychological state,")
    lines.append("   not academic engagement. Personality traits (especially conscientiousness)")
    lines.append("   remain the primary predictor of academic performance, while passively")
    lines.append("   sensed behavioral patterns are informative for mental health screening.")
    lines.append("")
    lines.append("   Narrative structure:")
    lines.append("   1. PERSONALITY → GPA (direct path)")
    lines.append("      - Best: RF LOO-CV R²=0.212 (p=0.008), EN R²=0.126 (p=0.010)")
    lines.append("      - Conscientiousness is #1 predictor across all 4 models (SHAP-confirmed)")
    lines.append("      - Optuna tuning improved RF from R²=0.101 → 0.212 (+110%)")
    lines.append("      - Replicates Poropat (2009) meta-analysis with passive sensing era data")
    lines.append("")
    lines.append("   2. BEHAVIOR → WELLBEING (not GPA) — with a non-linear exception")
    lines.append("      - Behavior predicts PHQ-9 (R²=0.468) but NOT GPA via linear models")
    lines.append("      - PLS-SEM: Digital→Wellbeing (β=-0.49*), Mobility→Wellbeing (β=0.38*)")
    lines.append("      - KEY: SVR captures non-linear Behavior→GPA (R²=0.116, p=0.032)")
    lines.append("        missed by Elastic Net (R²=-0.130) and Ridge (R²=-0.516)")
    lines.append("      - SHAP: Mobility and Digital Usage are top behavioral drivers for PHQ-9")
    lines.append("")
    lines.append("   3. PERSONALITY MODERATES BEHAVIOR EFFECTS")
    lines.append("      - 7/120 significant moderation effects")
    lines.append("      - E×Activity→GPA (ΔR²=0.221): activity benefits extraverts more")
    lines.append("      - Explains why aggregate behavior→GPA path is null (heterogeneity)")
    lines.append("")
    lines.append("   4. BEHAVIORAL PATTERNS DIFFERENTIATE STUDENT SUBGROUPS")
    lines.append("      - 4 LPA profiles distinguish stress (p=0.024) and loneliness (p=0.023)")
    lines.append("      - But not depression (PHQ-9 p=0.154) — aligns with behavior→wellbeing")
    lines.append("")
    lines.append("   5. ML AS ANALYTICAL FRAMEWORK")
    lines.append("      - Optuna Bayesian optimization: data-driven hyperparameter selection")
    lines.append("      - SHAP: model-agnostic feature attribution across 4 methods")
    lines.append("      - Cross-model consistency: Kendall's τ=0.460 (feature ranking agreement)")
    lines.append("      - Non-linear discovery: SVR revealed behavior→GPA link invisible to linear models")
    lines.append("      - 8 complementary analyses converge on personality-behavior dissociation")
    lines.append("")
    lines.append("   Limitations:")
    lines.append("   - Small sample (N=28), single institution, single term")
    lines.append("   - Mediation paths underpowered (need N≥71 per Fritz & MacKinnon 2007)")
    lines.append("   - Cross-sectional design limits causal inference")
    lines.append("   - Android-only sample may not generalize")
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

    print("\n[1/7] Figure 1: Sample overview...")
    figure1_sample_overview(df)

    print("\n[2/7] Figure 2: Mediation summary...")
    figure2_mediation_summary(df)

    print("\n[3/7] Figure 3: Prediction comparison...")
    figure3_prediction_comparison()

    print("\n[4/7] Figure 4: Latent profiles...")
    figure4_lpa_profiles()

    print("\n[5/7] Figure 5: Multi-outcome prediction...")
    figure5_multi_outcome(df)

    print("\n[6/7] Figure 6: Effect size forest plot...")
    figure6_effect_size_forest()

    print("\n[7/8] Figure 7: SHAP ML interpretability...")
    figure7_shap_summary()

    print("\n[8/8] Summary report & narrative...")
    generate_summary_report(df)

    print("\n" + "=" * 60)
    print("PUBLICATION MATERIALS COMPLETE")
    print("=" * 60)
    print(f"\n  Figures: results/figures/figure1-7*.png")
    print(f"  Tables:  results/tables/*.csv")
    print(f"  Report:  results/reports/summary_report.txt")


if __name__ == '__main__':
    main()
