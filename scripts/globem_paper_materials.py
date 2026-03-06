#!/usr/bin/env python3
"""
Phase 13 Step 6: Study 3 Publication-Ready Materials

Generates figures and tables for the GLOBEM validation study:
  - Figure 13: GLOBEM sample overview (BFI-10 + 5 MH outcomes + behavior)
  - Figure 14: Three-study mental health prediction R² (forest plot)
  - Figure 15: SHAP consistency heatmap (3 studies × mental health)
  - Table 10:  Study 3 descriptive statistics
  - Table 11:  Mental health prediction results (Personality + Behavior)
  - Table 12:  Three-study replication summary
  - Report:    Study 3 narrative summary

Input:  results/globem/tables/*.csv
        results/comparison/three_study_*.csv
        data/processed/globem/globem_analysis_dataset.parquet
Output: results/globem/figures/figure_*.png
        results/globem/tables/table_*.csv
        results/globem/study3_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent

S3_DATA = project_root / 'data' / 'processed' / 'globem' / 'globem_analysis_dataset.parquet'
S3_TABLE_DIR = project_root / 'results' / 'globem' / 'tables'
S3_FIGURE_DIR = project_root / 'results' / 'globem' / 'figures'
COMP_DIR = project_root / 'results' / 'comparison'

for d in [S3_TABLE_DIR, S3_FIGURE_DIR]:
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

PERSONALITY = ['extraversion', 'agreeableness', 'conscientiousness',
               'neuroticism', 'openness']
TRAIT_SHORT = {'extraversion': 'E', 'agreeableness': 'A',
               'conscientiousness': 'C', 'neuroticism': 'N', 'openness': 'O'}
GB_BEHAVIOR_PC = ['activity_pc1', 'sleep_pc1', 'communication_pc1',
                  'digital_pc1', 'mobility_pc1']
BEH_SHORT = {'activity_pc1': 'Activity', 'sleep_pc1': 'Sleep',
             'communication_pc1': 'Comm', 'digital_pc1': 'Digital',
             'mobility_pc1': 'Mobility'}

MH_COLS = ['bdi2_total', 'stai_state', 'pss_10', 'cesd_total', 'ucla_loneliness']
MH_LABELS = {'bdi2_total': 'BDI-II', 'stai_state': 'STAI',
             'pss_10': 'PSS-10', 'cesd_total': 'CESD',
             'ucla_loneliness': 'UCLA'}


# ──────────────────────────────────────────────────────────────────────
# Figure 13: GLOBEM Sample Overview
# ──────────────────────────────────────────────────────────────────────

def figure13_sample_overview(df):
    """Figure 13: GLOBEM sample characteristics."""
    print("  Generating Figure 13: Sample Overview...")
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # A: Big Five distributions (BFI-10)
    ax1 = fig.add_subplot(gs[0, 0])
    trait_data = df[PERSONALITY].dropna().melt(var_name='Trait', value_name='Score')
    trait_data['Trait'] = trait_data['Trait'].map(TRAIT_SHORT)
    sns.boxplot(data=trait_data, x='Trait', y='Score', ax=ax1,
                palette='Set2', width=0.6)
    ax1.set_title('A. Big Five Personality (BFI-10)')
    ax1.set_ylabel('Score (1–5)')
    ax1.set_xlabel('')

    # B: Mental health distributions
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.preprocessing import StandardScaler
    available_mh = [c for c in MH_COLS if c in df.columns]
    mh_data = df[available_mh].dropna()
    if len(mh_data) > 10:
        scaler = StandardScaler()
        mh_z = pd.DataFrame(scaler.fit_transform(mh_data), columns=available_mh)
        mh_melt = mh_z.melt(var_name='Measure', value_name='Z-Score')
        mh_melt['Measure'] = mh_melt['Measure'].map(MH_LABELS)
        sns.boxplot(data=mh_melt, x='Measure', y='Z-Score', ax=ax2,
                    palette='Set3', width=0.6)
    ax2.set_title('B. Mental Health (Standardized)')
    ax2.set_xlabel('')
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.tick_params(axis='x', rotation=30)

    # C: BDI-II distribution
    ax3 = fig.add_subplot(gs[0, 2])
    bdi = df['bdi2_total'].dropna()
    ax3.hist(bdi, bins=25, color='#ef5350', edgecolor='white', alpha=0.8)
    ax3.axvline(bdi.mean(), color='#1a237e', linestyle='--', linewidth=2,
                label=f'Mean={bdi.mean():.1f}')
    ax3.axvline(14, color='#ff9800', linestyle=':', linewidth=1.5,
                label='BDI-II cutoff (14)')
    ax3.set_title(f'C. BDI-II Depression (N={len(bdi)})')
    ax3.set_xlabel('BDI-II Score')
    ax3.set_ylabel('Count')
    ax3.legend(fontsize=8)

    # D: Behavioral composites
    ax4 = fig.add_subplot(gs[1, 0])
    beh_avail = [c for c in GB_BEHAVIOR_PC if c in df.columns]
    beh_data = df[beh_avail].dropna().melt(var_name='Modality', value_name='PC1 Score')
    beh_data['Modality'] = beh_data['Modality'].map(BEH_SHORT)
    sns.boxplot(data=beh_data, x='Modality', y='PC1 Score', ax=ax4,
                palette='Paired', width=0.6)
    ax4.set_title('D. Behavioral Composites (PC1)')
    ax4.set_xlabel('')
    ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax4.tick_params(axis='x', rotation=30)

    # E: Neuroticism × Mental Health correlations
    ax5 = fig.add_subplot(gs[1, 1])
    corrs = []
    for mh_col in MH_COLS:
        if mh_col in df.columns:
            subset = df[['neuroticism', mh_col]].dropna()
            if len(subset) >= 10:
                r, p = sp_stats.pearsonr(subset['neuroticism'], subset[mh_col])
                corrs.append({'Outcome': MH_LABELS.get(mh_col, mh_col), 'r': r, 'p': p})
    if corrs:
        corr_df = pd.DataFrame(corrs)
        colors = ['#d32f2f' if p < 0.05 else '#90a4ae' for p in corr_df['p']]
        ax5.barh(corr_df['Outcome'], corr_df['r'], color=colors)
        ax5.set_title('E. Neuroticism × Mental Health')
        ax5.set_xlabel('Pearson r')
        ax5.axvline(0, color='black', linewidth=0.5)
        for i, row in corr_df.iterrows():
            sig = '*' if row['p'] < 0.05 else ''
            ax5.text(row['r'] + 0.01, i, f"r={row['r']:.2f}{sig}", va='center', fontsize=8)

    # F: Data availability by cohort
    ax6 = fig.add_subplot(gs[1, 2])
    if 'cohort' in df.columns:
        cohort_counts = df.groupby('cohort').size()
        cohort_labels = [f'W{int(c)} ({"2018" if c == 1 else "2019" if c == 2 else "2020*" if c == 3 else "2021"})'
                        for c in cohort_counts.index]
        bars = ax6.bar(cohort_labels, cohort_counts.values, color='#42a5f5', alpha=0.8)
        ax6.set_title('F. Participants by Cohort')
        ax6.set_ylabel('N')
        ax6.set_xlabel('')
        # Mark COVID cohort
        for i, c in enumerate(cohort_counts.index):
            if c == 3:
                bars[i].set_color('#ff9800')
                bars[i].set_alpha(0.8)
        for i, v in enumerate(cohort_counts.values):
            ax6.text(i, v + 2, str(v), ha='center', fontsize=9)

    plt.suptitle('Figure 13: GLOBEM Sample Characteristics (Study 3)',
                fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(S3_FIGURE_DIR / 'figure13_sample_overview.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure13_sample_overview.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 14: Three-Study Mental Health R² Forest Plot
# ──────────────────────────────────────────────────────────────────────

def figure14_forest_mh_r2():
    """Figure 14: Three-study mental health prediction R² comparison."""
    print("  Generating Figure 14: Three-Study MH R² Forest Plot...")

    ml_file = COMP_DIR / 'three_study_ml_mh.csv'
    if not ml_file.exists():
        print("  three_study_ml_mh.csv not found, skipping")
        return

    ml_df = pd.read_csv(ml_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'S1': '#2196F3', 'S2': '#FF9800', 'S3': '#4CAF50'}
    labels = {'S1': 'Study 1 (StudentLife)', 'S2': 'Study 2 (NetHealth)',
              'S3': 'Study 3 (GLOBEM)'}
    markers = {'S1': 's', 'S2': 'o', 'S3': 'D'}

    y = 0
    y_positions = []
    y_labels = []
    labeled = set()

    for _, row in ml_df.iterrows():
        construct = row['Construct']
        fset = row['Feature_Set']
        y_label = f"{construct}\n({fset})"

        for i, study in enumerate(['S1', 'S2', 'S3']):
            r2_col = f'{study}_R2'
            if r2_col in row and pd.notna(row.get(r2_col)):
                offset = (i - 1) * 0.2
                lbl = labels[study] if study not in labeled else ''
                ax.plot(row[r2_col], y + offset, markers[study],
                       color=colors[study], markersize=9,
                       label=lbl)
                if lbl:
                    labeled.add(study)

        y_positions.append(y)
        y_labels.append(y_label)
        y += 1

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('R² (Best Model, 10×10-fold CV)')
    ax.set_title('Figure 14: Mental Health Prediction Across Three Studies')
    ax.legend(loc='lower right', fontsize=8)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(S3_FIGURE_DIR / 'figure14_forest_mh_r2.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure14_forest_mh_r2.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 15: SHAP Consistency Heatmap (3 studies)
# ──────────────────────────────────────────────────────────────────────

def figure15_shap_heatmap():
    """Figure 15: SHAP feature ranking consistency across studies."""
    print("  Generating Figure 15: SHAP Consistency Heatmap...")

    shap_file = COMP_DIR / 'three_study_shap_mh.csv'
    if not shap_file.exists():
        print("  three_study_shap_mh.csv not found, skipping")
        return

    shap_df = pd.read_csv(shap_file)

    # Build a heatmap: rows = Study × Model, cols = whether N is top
    # Simpler: just show top feature per cell

    studies = ['S1', 'S2', 'S3']
    study_labels = ['Study 1', 'Study 2', 'Study 3']

    # For each construct, show which trait is #1 SHAP
    constructs = shap_df['Construct'].unique()

    fig, axes = plt.subplots(1, len(constructs), figsize=(6 * len(constructs), 5))
    if len(constructs) == 1:
        axes = [axes]

    for idx, construct in enumerate(constructs):
        ax = axes[idx]
        sub = shap_df[shap_df['Construct'] == construct]

        # Create matrix: models × studies
        models = sub['Model'].unique()
        matrix = np.zeros((len(models), len(studies)))
        annot = []

        for i, model in enumerate(models):
            row_annot = []
            for j, study in enumerate(studies):
                col = f'{study}_top'
                if col in sub.columns:
                    model_row = sub[sub['Model'] == model]
                    if len(model_row) > 0:
                        val = model_row[col].values[0]
                        if pd.notna(val):
                            row_annot.append(TRAIT_SHORT.get(val, val[:3]))
                            matrix[i, j] = 1 if val.lower() == 'neuroticism' else 0
                        else:
                            row_annot.append('')
                    else:
                        row_annot.append('')
                else:
                    row_annot.append('')
            annot.append(row_annot)

        sns.heatmap(matrix, annot=np.array(annot), fmt='', cmap=['#E3F2FD', '#1565C0'],
                    xticklabels=study_labels, yticklabels=models,
                    cbar=False, ax=ax, linewidths=1, linecolor='white',
                    vmin=0, vmax=1)
        ax.set_title(f'{construct}\n(Blue = Neuroticism #1)')

    plt.suptitle('Figure 15: SHAP Top Feature Across Studies',
                fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(S3_FIGURE_DIR / 'figure15_shap_heatmap.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: figure15_shap_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# Table 10: Study 3 Descriptive Statistics
# ──────────────────────────────────────────────────────────────────────

def table10_descriptive(df):
    """Table 10: GLOBEM descriptive statistics."""
    print("  Generating Table 10: Descriptive Statistics...")

    all_vars = PERSONALITY + MH_COLS + GB_BEHAVIOR_PC
    rows = []
    for col in all_vars:
        if col in df.columns:
            s = df[col].dropna()
            rows.append({
                'Variable': col,
                'Label': TRAIT_SHORT.get(col, MH_LABELS.get(col, BEH_SHORT.get(col, col))),
                'N': len(s),
                'Mean': s.mean(),
                'SD': s.std(),
                'Min': s.min(),
                'Max': s.max(),
                'Skew': s.skew(),
            })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(S3_TABLE_DIR / 'table10_descriptive.csv', index=False)
    print(f"  Saved: table10_descriptive.csv ({len(stats_df)} rows)")
    return stats_df


# ──────────────────────────────────────────────────────────────────────
# Table 11: Mental Health Prediction Results
# ──────────────────────────────────────────────────────────────────────

def table11_mh_prediction():
    """Table 11: Combined personality + behavior → mental health results."""
    print("  Generating Table 11: MH Prediction Results...")

    files = [
        S3_TABLE_DIR / 'personality_mental_health.csv',
        S3_TABLE_DIR / 'behavior_mental_health_all.csv',
    ]

    dfs = []
    for f in files:
        if f.exists():
            dfs.append(pd.read_csv(f))

    if not dfs:
        print("  No prediction results found, skipping")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Rename Feature_Set for consistency
    if 'Feature_Set' in combined.columns:
        combined['Features'] = combined['Feature_Set']
    elif 'Features' not in combined.columns:
        combined['Features'] = 'Personality'

    # Select best model per outcome × feature set
    best_rows = []
    for outcome in combined['Outcome'].unique():
        for fset in combined['Features'].unique():
            sub = combined[(combined['Outcome'] == outcome) &
                           (combined['Features'] == fset)]
            if len(sub) > 0:
                best = sub.loc[sub['R2_mean'].idxmax()]
                best_rows.append(best)

    if best_rows:
        result = pd.DataFrame(best_rows)
        result = result.sort_values(['Outcome', 'Features'])
        result.to_csv(S3_TABLE_DIR / 'table11_mh_prediction.csv', index=False)
        print(f"  Saved: table11_mh_prediction.csv ({len(result)} rows)")
        return result
    return None


# ──────────────────────────────────────────────────────────────────────
# Table 12: Three-Study Replication Summary
# ──────────────────────────────────────────────────────────────────────

def table12_replication():
    """Table 12: Copy/format three-study replication summary."""
    print("  Generating Table 12: Replication Summary...")

    src = COMP_DIR / 'three_study_replication_summary.csv'
    if not src.exists():
        print("  three_study_replication_summary.csv not found, skipping")
        return None

    df = pd.read_csv(src)
    df.to_csv(S3_TABLE_DIR / 'table12_replication.csv', index=False)
    print(f"  Saved: table12_replication.csv ({len(df)} rows)")
    return df


# ──────────────────────────────────────────────────────────────────────
# Report: Study 3 Narrative Summary
# ──────────────────────────────────────────────────────────────────────

def generate_report(df):
    """Generate a text report summarizing Study 3 findings."""
    print("  Generating Study 3 report...")

    lines = []
    lines.append("=" * 70)
    lines.append("STUDY 3: GLOBEM VALIDATION — NARRATIVE SUMMARY")
    lines.append("=" * 70)

    # Sample
    n_total = len(df)
    n_pers = df[PERSONALITY[0]].notna().sum()
    n_bdi = df['bdi2_total'].notna().sum() if 'bdi2_total' in df.columns else 0
    lines.append(f"\nSample: N={n_total} participants across 4 cohorts (2018-2021)")
    lines.append(f"  Personality (BFI-10): N={n_pers}")
    lines.append(f"  BDI-II depression: N={n_bdi}")

    if 'cohort' in df.columns:
        for c in sorted(df['cohort'].dropna().unique()):
            n_c = (df['cohort'] == c).sum()
            year = {1: '2018', 2: '2019', 3: '2020 (COVID)', 4: '2021'}.get(c, '?')
            lines.append(f"  Cohort {int(c)} ({year}): N={n_c}")

    # Key correlations
    lines.append("\n--- Key Personality × Mental Health Correlations ---")
    for mh_col in MH_COLS:
        if mh_col in df.columns:
            for trait in PERSONALITY:
                sub = df[[trait, mh_col]].dropna()
                if len(sub) > 10:
                    r, p = sp_stats.pearsonr(sub[trait], sub[mh_col])
                    if p < 0.05:
                        sig = '**' if p < 0.01 else '*'
                        lines.append(f"  {trait:20s} × {MH_LABELS.get(mh_col, mh_col):8s}: "
                                    f"r = {r:+.3f}{sig} (N={len(sub)})")

    # ML results
    pers_mh_file = S3_TABLE_DIR / 'personality_mental_health.csv'
    if pers_mh_file.exists():
        pers_mh = pd.read_csv(pers_mh_file)
        lines.append("\n--- Personality → Mental Health (ML) ---")
        for outcome in pers_mh['Outcome'].unique():
            sub = pers_mh[pers_mh['Outcome'] == outcome]
            best = sub.loc[sub['R2_mean'].idxmax()]
            sig = '*' if best['p_perm'] < 0.05 else ''
            lines.append(f"  {outcome:10s}: best R²={best['R2_mean']:.3f} ({best['Model']}) p={best['p_perm']:.4f}{sig}")

    # SHAP
    lines.append("\n--- SHAP: Neuroticism as #1 Predictor ---")
    for mh_col, mh_label in [('bdi2_total', 'BDI-II'), ('stai_state', 'STAI'), ('pss_10', 'PSS-10')]:
        safe = mh_label.lower().replace('-', '').replace(' ', '_')
        shap_file = S3_TABLE_DIR / f'shap_personality_{safe}.csv'
        if shap_file.exists():
            shap_df = pd.read_csv(shap_file, index_col=0)
            n_top = sum(1 for model in shap_df.index
                       if shap_df.loc[model].idxmax() == 'neuroticism')
            lines.append(f"  {mh_label}: Neuroticism = #1 in {n_top}/{len(shap_df)} models")

    # LPA
    lpa_file = S3_TABLE_DIR / 'lpa_outcomes.csv'
    if lpa_file.exists():
        lpa = pd.read_csv(lpa_file)
        lines.append("\n--- LPA: Behavioral Profile Differences ---")
        for _, r in lpa.iterrows():
            sig = '*' if r['p'] < 0.05 else ''
            lines.append(f"  {r['Outcome']:10s}: F={r['F']:.2f}, p={r['p']:.4f}{sig}, "
                        f"eta²={r.get('eta_sq', 0):.3f}")

    # COVID sensitivity
    covid_file = S3_TABLE_DIR / 'covid_sensitivity.csv'
    if covid_file.exists():
        covid = pd.read_csv(covid_file)
        lines.append("\n--- COVID Sensitivity (Excluding INS-W_3) ---")
        for _, r in covid.iterrows():
            lines.append(f"  {r['Outcome']:10s}: Full R²={r['Full_R2']:.3f} → "
                        f"No-COVID R²={r['NoCovid_R2']:.3f} (Δ={r['Delta']:+.3f})")

    lines.append("\n" + "=" * 70)
    lines.append("END OF STUDY 3 REPORT")
    lines.append("=" * 70)

    report_text = '\n'.join(lines)
    report_path = project_root / 'results' / 'globem' / 'study3_report.txt'
    report_path.write_text(report_text)
    print(f"  Saved: {report_path}")
    print(report_text)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 13 STEP 6: GLOBEM PAPER MATERIALS")
    print("=" * 60)

    df = pd.read_parquet(S3_DATA)
    print(f"Dataset: {df.shape[0]} × {df.shape[1]}")

    # Figure 13: Sample overview
    figure13_sample_overview(df)

    # Figure 14: Three-study forest plot
    figure14_forest_mh_r2()

    # Figure 15: SHAP heatmap
    figure15_shap_heatmap()

    # Table 10: Descriptive stats
    table10_descriptive(df)

    # Table 11: MH prediction
    table11_mh_prediction()

    # Table 12: Replication summary
    table12_replication()

    # Report
    generate_report(df)

    print(f"\n{'=' * 60}")
    print("PHASE 13 STEP 6 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
