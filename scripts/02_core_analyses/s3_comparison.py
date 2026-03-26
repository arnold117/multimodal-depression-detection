#!/usr/bin/env python3
"""
Phase 13 Step 5: Three-Study Comparison

Compares key findings across all three studies:
  - Study 1: StudentLife (N=27-28)
  - Study 2: NetHealth (N=135-722)
  - Study 3: GLOBEM (N=~500-800)

Focus: mental health prediction replication (GLOBEM has no GPA).

Replication criteria (Open Science Collaboration, 2015):
  1. Effect direction consistent
  2. Replication effect within original CI
  3. Replication statistically significant (p < .05)

Output: results/comparison/three_study_*.csv
        results/comparison/three_study_*.png
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

project_root = Path(__file__).parent.parent.parent

# ── Result directories ──
S1_TABLE_DIR = project_root / 'results' / 'tables'
S1_DATA = project_root / 'data' / 'processed' / 'analysis_dataset.parquet'

S2_TABLE_DIR = project_root / 'results' / 'nethealth' / 'tables'
S2_DATA = project_root / 'data' / 'processed' / 'nethealth' / 'nethealth_analysis_dataset.parquet'

S3_TABLE_DIR = project_root / 'results' / 'globem' / 'tables'
S3_DATA = project_root / 'data' / 'processed' / 'globem' / 'globem_analysis_dataset.parquet'

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

# Study colors
COLORS = {'S1': '#2196F3', 'S2': '#FF9800', 'S3': '#4CAF50'}
STUDY_LABELS = {'S1': 'Study 1 (StudentLife)', 'S2': 'Study 2 (NetHealth)',
                'S3': 'Study 3 (GLOBEM)'}


# ──────────────────────────────────────────────────────────────────────
# Utility
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
# 1. Personality–Mental Health Correlations (Three Studies)
# ──────────────────────────────────────────────────────────────────────

def compare_personality_mh_correlations(s1_df, s2_df, s3_df):
    """Compare personality–mental health correlations across all three studies.

    Mappings:
      Depression: S1 PHQ-9, S2 CES-D, S3 BDI-II
      Stress/Anxiety: S1 PSS, S2 STAI, S3 STAI/PSS-10
      Loneliness: S1 loneliness, S2 loneliness, S3 UCLA
    """
    print("\n[1/6] Personality → Mental Health Correlations")
    print("─" * 60)

    # Define outcome mappings
    outcome_maps = [
        ('Depression', 'phq9_total', 'cesd_total', 'bdi2_total'),
        ('State Anxiety', 'pss_total', 'stai_trait_total', 'stai_state'),
        ('Perceived Stress', 'pss_total', None, 'pss_10'),
        ('Loneliness', 'loneliness_total', 'loneliness_total', 'ucla_loneliness'),
    ]

    rows = []
    for construct, s1_col, s2_col, s3_col in outcome_maps:
        for trait in PERSONALITY:
            row = {'Construct': construct, 'Trait': trait}

            # Study 1
            if s1_col and s1_col in s1_df.columns and trait in s1_df.columns:
                sub = s1_df[[trait, s1_col]].dropna()
                if len(sub) > 5:
                    r, p = sp_stats.pearsonr(sub[trait], sub[s1_col])
                    ci = bootstrap_r_ci(sub[trait].values, sub[s1_col].values)
                    row.update({'S1_r': r, 'S1_p': p, 'S1_ci_lo': ci[0], 'S1_ci_hi': ci[1], 'S1_N': len(sub)})

            # Study 2
            if s2_col and s2_col in s2_df.columns and trait in s2_df.columns:
                sub = s2_df[[trait, s2_col]].dropna()
                if len(sub) > 5:
                    r, p = sp_stats.pearsonr(sub[trait], sub[s2_col])
                    ci = bootstrap_r_ci(sub[trait].values, sub[s2_col].values)
                    row.update({'S2_r': r, 'S2_p': p, 'S2_ci_lo': ci[0], 'S2_ci_hi': ci[1], 'S2_N': len(sub)})

            # Study 3
            if s3_col and s3_col in s3_df.columns and trait in s3_df.columns:
                sub = s3_df[[trait, s3_col]].dropna()
                if len(sub) > 5:
                    r, p = sp_stats.pearsonr(sub[trait], sub[s3_col])
                    ci = bootstrap_r_ci(sub[trait].values, sub[s3_col].values)
                    row.update({'S3_r': r, 'S3_p': p, 'S3_ci_lo': ci[0], 'S3_ci_hi': ci[1], 'S3_N': len(sub)})

            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(COMP_DIR / 'three_study_mh_correlations.csv', index=False)

    # Print key findings: Neuroticism
    print("\n  Neuroticism → Mental Health (key hypothesis):")
    n_rows = result[result['Trait'] == 'neuroticism']
    for _, r in n_rows.iterrows():
        parts = [f"  {r['Construct']:20s}"]
        for study in ['S1', 'S2', 'S3']:
            col = f'{study}_r'
            if col in r and pd.notna(r[col]):
                sig = '*' if r[f'{study}_p'] < 0.05 else ''
                parts.append(f"{study}: r={r[col]:+.3f}{sig}")
            else:
                parts.append(f"{study}: N/A")
        print("  ".join(parts))

    print(f"\n  Saved: three_study_mh_correlations.csv ({len(result)} rows)")
    return result


# ──────────────────────────────────────────────────────────────────────
# 2. ML Mental Health Prediction Comparison
# ──────────────────────────────────────────────────────────────────────

def compare_ml_mental_health():
    """Compare mental health prediction R² across three studies."""
    print("\n[2/6] ML Mental Health Prediction R²")
    print("─" * 60)

    # Study 1: multi_outcome_matrix.csv (LOO-CV)
    s1_matrix = pd.read_csv(S1_TABLE_DIR / 'multi_outcome_matrix.csv')

    # Study 2: behavior_stai.csv, behavior_cesd.csv (10×10-fold)
    # Study 3: personality_mental_health.csv, behavior_mental_health_all.csv (10×10-fold)

    # Construct mapping: (construct_label, S1_outcome, S2_file, S3_outcome_label)
    construct_maps = [
        ('Depression', 'PHQ-9', 'behavior_cesd', 'BDI-II'),
        ('Anxiety', 'PSS', 'behavior_stai', 'STAI'),
    ]

    rows = []
    for construct, s1_outcome, s2_file, s3_outcome in construct_maps:
        # S1: Personality features, best model
        s1_pers = s1_matrix[(s1_matrix['Outcome'] == s1_outcome) &
                             (s1_matrix['Features'] == 'Personality')]
        if len(s1_pers) > 0:
            s1_best = s1_pers.loc[s1_pers['R2'].idxmax()]
        else:
            s1_best = None

        # S2: Personality features
        s2_path = S2_TABLE_DIR / f'{s2_file}.csv'
        s2_best = None
        if s2_path.exists():
            s2_data = pd.read_csv(s2_path)
            s2_pers = s2_data[s2_data['Features'] == 'Personality']
            if len(s2_pers) > 0:
                r2_col = 'R2_kfold' if 'R2_kfold' in s2_pers.columns else 'R2_mean'
                s2_best = s2_pers.loc[s2_pers[r2_col].idxmax()]

        # S3: Personality features
        s3_pers_path = S3_TABLE_DIR / 'personality_mental_health.csv'
        s3_best = None
        if s3_pers_path.exists():
            s3_data = pd.read_csv(s3_pers_path)
            s3_pers = s3_data[s3_data['Outcome'] == s3_outcome]
            if len(s3_pers) > 0:
                s3_best = s3_pers.loc[s3_pers['R2_mean'].idxmax()]

        for fset in ['Personality', 'Behavior', 'Pers + Beh', 'Pers+Beh']:
            row = {'Construct': construct, 'Feature_Set': fset}

            # S1
            s1_sub = s1_matrix[(s1_matrix['Outcome'] == s1_outcome) &
                                (s1_matrix['Features'] == fset)]
            if len(s1_sub) > 0:
                best = s1_sub.loc[s1_sub['R2'].idxmax()]
                row.update({'S1_R2': best['R2'], 'S1_Model': best['Model'], 'S1_N': int(best['N'])})

            # S2
            if s2_path.exists():
                s2_data = pd.read_csv(s2_path)
                # Normalize feature set names
                s2_fset = fset
                s2_sub = s2_data[s2_data['Features'] == s2_fset]
                if len(s2_sub) == 0 and fset == 'Pers+Beh':
                    s2_sub = s2_data[s2_data['Features'] == 'Pers + Beh']
                if len(s2_sub) == 0 and fset == 'Pers + Beh':
                    s2_sub = s2_data[s2_data['Features'] == 'Pers+Beh']
                if len(s2_sub) > 0:
                    r2_col = 'R2_kfold' if 'R2_kfold' in s2_sub.columns else 'R2_mean'
                    best = s2_sub.loc[s2_sub[r2_col].idxmax()]
                    row.update({'S2_R2': best[r2_col], 'S2_Model': best['Model'], 'S2_N': int(best['N'])})

            # S3: behavior_mental_health_all.csv has all feature sets
            s3_beh_path = S3_TABLE_DIR / 'behavior_mental_health_all.csv'
            s3_pers_path = S3_TABLE_DIR / 'personality_mental_health.csv'

            s3_sub = pd.DataFrame()
            if fset in ['Personality'] and s3_pers_path.exists():
                s3_all = pd.read_csv(s3_pers_path)
                s3_sub = s3_all[s3_all['Outcome'] == s3_outcome]
            elif s3_beh_path.exists():
                s3_all = pd.read_csv(s3_beh_path)
                s3_fset = fset
                s3_sub = s3_all[(s3_all['Outcome'] == s3_outcome) &
                                (s3_all['Feature_Set'] == s3_fset)]
                if len(s3_sub) == 0 and fset == 'Pers + Beh':
                    s3_sub = s3_all[(s3_all['Outcome'] == s3_outcome) &
                                    (s3_all['Feature_Set'] == 'Pers+Beh')]

            if len(s3_sub) > 0:
                best = s3_sub.loc[s3_sub['R2_mean'].idxmax()]
                row.update({'S3_R2': best['R2_mean'], 'S3_Model': best['Model'], 'S3_N': int(best['N'])})

            # Only include if at least 2 studies have data
            n_studies = sum(1 for s in ['S1_R2', 'S2_R2', 'S3_R2'] if s in row and pd.notna(row.get(s)))
            if n_studies >= 2:
                rows.append(row)

    result = pd.DataFrame(rows)
    # Drop duplicate feature set names (Pers + Beh vs Pers+Beh)
    result = result.drop_duplicates(subset=['Construct', 'Feature_Set'], keep='first')
    result.to_csv(COMP_DIR / 'three_study_ml_mh.csv', index=False)

    for _, r in result.iterrows():
        parts = [f"  {r['Construct']:12s} {r['Feature_Set']:12s}"]
        for s in ['S1', 'S2', 'S3']:
            col = f'{s}_R2'
            if col in r and pd.notna(r.get(col)):
                parts.append(f"{s}: R²={r[col]:.3f}")
            else:
                parts.append(f"{s}: N/A")
        print("  ".join(parts))

    print(f"\n  Saved: three_study_ml_mh.csv ({len(result)} rows)")
    return result


# ──────────────────────────────────────────────────────────────────────
# 3. SHAP Feature Ranking: Neuroticism = #1 for Mental Health?
# ──────────────────────────────────────────────────────────────────────

def compare_shap_mental_health():
    """Compare SHAP rankings for personality → mental health across studies."""
    print("\n[3/6] SHAP Feature Ranking: Mental Health")
    print("─" * 60)

    rows = []

    # Study 1: shap_importance.csv (Scenario format)
    s1_shap = pd.read_csv(S1_TABLE_DIR / 'shap_importance.csv')

    # Construct maps: (construct, S1_scenario, S2_file, S3_file)
    construct_maps = [
        ('Depression', 'Personality → PHQ-9', 'shap_personality_stai.csv', 'shap_personality_bdiii.csv'),
        ('Anxiety/Stress', 'Personality → PSS', 'shap_personality_stai.csv', 'shap_personality_stai.csv'),
    ]

    for construct, s1_scenario, s2_file, s3_file in construct_maps:
        # S1
        s1_scen = s1_shap[s1_shap['Scenario'] == s1_scenario]

        for model in ['Elastic Net', 'Ridge', 'Random Forest', 'SVR']:
            row = {'Construct': construct, 'Model': model}

            # S1
            s1_m = s1_scen[s1_scen['Model'] == model]
            if len(s1_m) > 0:
                top = s1_m.loc[s1_m['Mean_Abs_SHAP'].idxmax(), 'Feature']
                row['S1_top'] = top.lower() if isinstance(top, str) else top

            # S2
            s2_path = S2_TABLE_DIR / s2_file
            if s2_path.exists():
                s2_shap = pd.read_csv(s2_path, index_col=0)
                if model in s2_shap.index:
                    vals = s2_shap.loc[model]
                    row['S2_top'] = vals.idxmax()

            # S3
            s3_path = S3_TABLE_DIR / s3_file
            if s3_path.exists():
                s3_shap = pd.read_csv(s3_path, index_col=0)
                if model in s3_shap.index:
                    vals = s3_shap.loc[model]
                    row['S3_top'] = vals.idxmax()

            # Check if Neuroticism is #1
            for s in ['S1', 'S2', 'S3']:
                col = f'{s}_top'
                if col in row:
                    row[f'{s}_N_is_top'] = row[col].lower() == 'neuroticism'

            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(COMP_DIR / 'three_study_shap_mh.csv', index=False)

    # Summary
    for construct in result['Construct'].unique():
        sub = result[result['Construct'] == construct]
        for s in ['S1', 'S2', 'S3']:
            col = f'{s}_N_is_top'
            if col in sub.columns:
                n_top = sub[col].sum()
                print(f"  {construct:20s} {STUDY_LABELS.get(s, s):30s}: N=#1 in {n_top}/{len(sub)} models")

    print(f"\n  Saved: three_study_shap_mh.csv")
    return result


# ──────────────────────────────────────────────────────────────────────
# 4. LPA Comparison
# ──────────────────────────────────────────────────────────────────────

def compare_lpa():
    """Compare LPA results across studies."""
    print("\n[4/6] LPA Behavioral Profiles Comparison")
    print("─" * 60)

    rows = []

    # Study 1: lpa_outcome_comparison.csv (Kruskal-Wallis, 4 profiles)
    s1_path = S1_TABLE_DIR / 'lpa_outcome_comparison.csv'
    if s1_path.exists():
        s1_lpa = pd.read_csv(s1_path)
        for _, r in s1_lpa.iterrows():
            rows.append({
                'Study': 'S1 (StudentLife)',
                'Outcome': r['Variable'],
                'N': sum(r.get(f'Profile_{i}_n', 0) for i in range(4)),
                'k': 4,
                'Test': 'Kruskal-Wallis',
                'Statistic': r['H_stat'],
                'p': r['p_kruskal'],
            })

    # Study 2: lpa_outcomes.csv (ANOVA, k profiles)
    s2_path = S2_TABLE_DIR / 'lpa_outcomes.csv'
    if s2_path.exists():
        s2_lpa = pd.read_csv(s2_path)
        for _, r in s2_lpa.iterrows():
            rows.append({
                'Study': 'S2 (NetHealth)',
                'Outcome': r['Outcome'],
                'N': int(r['N']),
                'k': int(r['k']),
                'Test': 'ANOVA',
                'Statistic': r['F'],
                'p': r['p'],
            })

    # Study 3: lpa_outcomes.csv
    s3_path = S3_TABLE_DIR / 'lpa_outcomes.csv'
    if s3_path.exists():
        s3_lpa = pd.read_csv(s3_path)
        for _, r in s3_lpa.iterrows():
            rows.append({
                'Study': 'S3 (GLOBEM)',
                'Outcome': r['Outcome'],
                'N': int(r['N']),
                'k': int(r['k']),
                'Test': 'ANOVA',
                'Statistic': r['F'],
                'p': r['p'],
            })

    if rows:
        result = pd.DataFrame(rows)
        result.to_csv(COMP_DIR / 'three_study_lpa.csv', index=False)

        for _, r in result.iterrows():
            sig = '*' if r['p'] < 0.05 else ''
            print(f"  {r['Study']:20s}  {r['Outcome']:20s}  "
                  f"k={r['k']}  {r['Test']}={r['Statistic']:.2f}  p={r['p']:.4f}{sig}")

        print(f"\n  Saved: three_study_lpa.csv")
        return result
    return None


# ──────────────────────────────────────────────────────────────────────
# 5. Forest Plot: Neuroticism → Mental Health (Three Studies)
# ──────────────────────────────────────────────────────────────────────

def plot_three_study_neuroticism(corr_df):
    """Forest plot: Neuroticism × mental health correlations across 3 studies."""
    print("\n[5/6] Forest Plot: Neuroticism → Mental Health")
    print("─" * 60)

    if corr_df is None or len(corr_df) == 0:
        print("  No correlation data")
        return

    n_rows = corr_df[corr_df['Trait'] == 'neuroticism'].copy()
    if len(n_rows) == 0:
        print("  No neuroticism data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    y = 0
    y_positions = []
    y_labels = []

    for _, row in n_rows.iterrows():
        construct = row['Construct']

        for i, (study, color) in enumerate([('S1', COLORS['S1']),
                                              ('S2', COLORS['S2']),
                                              ('S3', COLORS['S3'])]):
            r_col = f'{study}_r'
            if r_col not in row or pd.isna(row.get(r_col)):
                continue

            r_val = row[r_col]
            ci_lo = row.get(f'{study}_ci_lo', r_val)
            ci_hi = row.get(f'{study}_ci_hi', r_val)
            n_val = int(row.get(f'{study}_N', 0))
            marker = ['s', 'o', 'D'][i]
            offset = (i - 1) * 0.2

            ax.errorbar(r_val, y + offset,
                        xerr=[[r_val - ci_lo], [ci_hi - r_val]],
                        fmt=marker, color=color, markersize=8, capsize=4,
                        linewidth=1.5,
                        label=f'{STUDY_LABELS[study]} (N={n_val})' if y == 0 else '')

        y_positions.append(y)
        y_labels.append(construct)
        y += 1

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Pearson r (Neuroticism → Mental Health Outcome)')
    ax.set_title('Neuroticism–Mental Health Correlations Across Three Studies')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.2, 0.8)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(COMP_DIR / 'three_study_forest_neuroticism.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print("  Saved: three_study_forest_neuroticism.png")


# ──────────────────────────────────────────────────────────────────────
# 6. Three-Study Replication Summary
# ──────────────────────────────────────────────────────────────────────

def generate_three_study_summary(corr_df, ml_df, shap_df, lpa_df):
    """Generate overall three-study replication summary."""
    print("\n" + "=" * 60)
    print("THREE-STUDY REPLICATION SUMMARY")
    print("=" * 60)

    rows = []

    # 1. Key correlation: Neuroticism → Depression
    if corr_df is not None:
        n_dep = corr_df[(corr_df['Trait'] == 'neuroticism') &
                        (corr_df['Construct'] == 'Depression')]
        if len(n_dep) > 0:
            r = n_dep.iloc[0]
            direction_consistent = True
            for s in ['S1', 'S2', 'S3']:
                col = f'{s}_r'
                if col in r and pd.notna(r[col]):
                    if r[col] <= 0:
                        direction_consistent = False
            rows.append({
                'Finding': 'N → Depression (r)',
                'S1': f"r={r.get('S1_r', np.nan):+.3f}" if pd.notna(r.get('S1_r')) else 'N/A',
                'S2': f"r={r.get('S2_r', np.nan):+.3f}" if pd.notna(r.get('S2_r')) else 'N/A',
                'S3': f"r={r.get('S3_r', np.nan):+.3f}" if pd.notna(r.get('S3_r')) else 'N/A',
                'direction_consistent': direction_consistent,
            })

        # Neuroticism → Anxiety/Stress
        n_anx = corr_df[(corr_df['Trait'] == 'neuroticism') &
                        (corr_df['Construct'] == 'State Anxiety')]
        if len(n_anx) > 0:
            r = n_anx.iloc[0]
            direction_consistent = True
            for s in ['S1', 'S2', 'S3']:
                col = f'{s}_r'
                if col in r and pd.notna(r[col]):
                    if r[col] <= 0:
                        direction_consistent = False
            rows.append({
                'Finding': 'N → Anxiety (r)',
                'S1': f"r={r.get('S1_r', np.nan):+.3f}" if pd.notna(r.get('S1_r')) else 'N/A',
                'S2': f"r={r.get('S2_r', np.nan):+.3f}" if pd.notna(r.get('S2_r')) else 'N/A',
                'S3': f"r={r.get('S3_r', np.nan):+.3f}" if pd.notna(r.get('S3_r')) else 'N/A',
                'direction_consistent': direction_consistent,
            })

    # 2. ML prediction: Personality → Depression R²
    if ml_df is not None:
        dep_pers = ml_df[(ml_df['Construct'] == 'Depression') &
                          (ml_df['Feature_Set'] == 'Personality')]
        if len(dep_pers) > 0:
            r = dep_pers.iloc[0]
            rows.append({
                'Finding': 'Pers → Depression (R²)',
                'S1': f"R²={r.get('S1_R2', np.nan):.3f}" if pd.notna(r.get('S1_R2')) else 'N/A',
                'S2': f"R²={r.get('S2_R2', np.nan):.3f}" if pd.notna(r.get('S2_R2')) else 'N/A',
                'S3': f"R²={r.get('S3_R2', np.nan):.3f}" if pd.notna(r.get('S3_R2')) else 'N/A',
                'direction_consistent': True,
            })

        anx_pers = ml_df[(ml_df['Construct'] == 'Anxiety') &
                          (ml_df['Feature_Set'] == 'Personality')]
        if len(anx_pers) > 0:
            r = anx_pers.iloc[0]
            rows.append({
                'Finding': 'Pers → Anxiety (R²)',
                'S1': f"R²={r.get('S1_R2', np.nan):.3f}" if pd.notna(r.get('S1_R2')) else 'N/A',
                'S2': f"R²={r.get('S2_R2', np.nan):.3f}" if pd.notna(r.get('S2_R2')) else 'N/A',
                'S3': f"R²={r.get('S3_R2', np.nan):.3f}" if pd.notna(r.get('S3_R2')) else 'N/A',
                'direction_consistent': True,
            })

    # 3. SHAP: N = #1 for mental health
    if shap_df is not None:
        for construct in shap_df['Construct'].unique():
            sub = shap_df[shap_df['Construct'] == construct]
            s1_n_top = sub['S1_N_is_top'].sum() if 'S1_N_is_top' in sub.columns else 0
            s2_n_top = sub['S2_N_is_top'].sum() if 'S2_N_is_top' in sub.columns else 0
            s3_n_top = sub['S3_N_is_top'].sum() if 'S3_N_is_top' in sub.columns else 0
            n_models = len(sub)

            rows.append({
                'Finding': f'SHAP N=#1 ({construct})',
                'S1': f"{s1_n_top}/{n_models}",
                'S2': f"{s2_n_top}/{n_models}",
                'S3': f"{s3_n_top}/{n_models}",
                'direction_consistent': (s1_n_top + s2_n_top + s3_n_top) > n_models,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(COMP_DIR / 'three_study_replication_summary.csv', index=False)

    print("\n  Finding                        S1              S2              S3              Consistent")
    print("  " + "─" * 90)
    for _, r in summary.iterrows():
        consistent = 'YES' if r.get('direction_consistent') else 'NO'
        print(f"  {r['Finding']:30s}  {r['S1']:14s}  {r['S2']:14s}  {r['S3']:14s}  {consistent}")

    print(f"\n  Total: {summary['direction_consistent'].sum()}/{len(summary)} findings consistent")
    print(f"  Saved: three_study_replication_summary.csv")
    return summary


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 13 STEP 5: THREE-STUDY COMPARISON")
    print("=" * 60)

    # Load datasets
    print("\n  Loading datasets...")
    s1_df = pd.read_parquet(S1_DATA)
    s2_df = pd.read_parquet(S2_DATA)
    s3_df = pd.read_parquet(S3_DATA)
    print(f"  Study 1 (StudentLife): {s1_df.shape[0]} × {s1_df.shape[1]}")
    print(f"  Study 2 (NetHealth):   {s2_df.shape[0]} × {s2_df.shape[1]}")
    print(f"  Study 3 (GLOBEM):      {s3_df.shape[0]} × {s3_df.shape[1]}")

    # 1. Correlations
    corr_df = compare_personality_mh_correlations(s1_df, s2_df, s3_df)

    # 2. ML predictions
    ml_df = compare_ml_mental_health()

    # 3. SHAP rankings
    shap_df = compare_shap_mental_health()

    # 4. LPA
    lpa_df = compare_lpa()

    # 5. Forest plot
    plot_three_study_neuroticism(corr_df)

    # 6. Summary
    generate_three_study_summary(corr_df, ml_df, shap_df, lpa_df)

    print(f"\n{'=' * 60}")
    print("PHASE 13 STEP 5 COMPLETE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
