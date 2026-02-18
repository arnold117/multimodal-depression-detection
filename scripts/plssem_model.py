#!/usr/bin/env python3
"""
PLS-SEM Structural Equation Model

Research Question: What are the structural relationships between
personality, smartphone behavior, psychological wellbeing, and GPA?

Measurement Model:
  - PERSONALITY (reflective): E, A, C, N, O
  - WELLBEING (reflective): PHQ-9(r), PSS(r), Loneliness(r), Flourishing, PA, NA(r)
  - MOBILITY (formative): mobility_pc1
  - DIGITAL (formative): digital_pc1
  - SOCIAL (formative): social_pc1
  - ACTIVITY (formative): activity_pc1
  - SCREEN (formative): screen_pc1
  - PROXIMITY (formative): proximity_pc1
  - FACE2FACE (formative): face2face_pc1
  - AUDIO (formative): audio_pc1
  - GPA (single indicator): gpa_overall

Structural Paths:
  PERSONALITY → all behavioral constructs → WELLBEING → GPA
  PERSONALITY → GPA (direct)

Method: PLS-PM with Bootstrap 5000 for path significance.
Uses semopy for SEM estimation.

Input:  data/processed/analysis_dataset.parquet
Output: results/tables/plssem_results.csv
        results/tables/plssem_effects.csv
        results/figures/plssem_path_diagram.png
        results/figures/plssem_r_squared.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
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

N_BOOT = 5000
RANDOM_STATE = 42


def prepare_data(df):
    """Prepare standardized variables for SEM."""
    # Personality indicators
    personality_cols = ['extraversion', 'agreeableness', 'conscientiousness',
                        'neuroticism', 'openness']
    # Wellbeing indicators (reverse-code so higher = better wellbeing)
    wellbeing_pos = ['flourishing_total', 'panas_positive']
    wellbeing_neg = ['phq9_total', 'pss_total', 'loneliness_total', 'panas_negative']
    wellbeing_cols = wellbeing_pos + wellbeing_neg

    # Behavioral composites
    behavior_cols = ['mobility_pc1', 'digital_pc1', 'social_pc1', 'activity_pc1',
                     'screen_pc1', 'proximity_pc1', 'face2face_pc1', 'audio_pc1']

    outcome_col = 'gpa_overall'

    all_cols = personality_cols + wellbeing_cols + behavior_cols + [outcome_col]
    subset = df[all_cols].dropna()
    print(f"  Complete cases: {len(subset)}")

    # Standardize
    scaler = StandardScaler()
    data = pd.DataFrame(
        scaler.fit_transform(subset),
        columns=subset.columns,
        index=subset.index
    )

    # Reverse-code negative wellbeing indicators
    for col in wellbeing_neg:
        data[col] = -data[col]

    return data, personality_cols, wellbeing_cols, behavior_cols, outcome_col


def compute_construct_scores(data, personality_cols, wellbeing_cols):
    """Compute composite scores for multi-indicator constructs."""
    scores = data.copy()
    scores['PERSONALITY'] = data[personality_cols].mean(axis=1)
    scores['WELLBEING'] = data[wellbeing_cols].mean(axis=1)
    return scores


def estimate_paths(data, personality_cols, wellbeing_cols, behavior_cols, outcome_col):
    """Estimate structural paths using OLS regression (PLS-PM approach)."""
    from numpy.linalg import lstsq

    scores = compute_construct_scores(data, personality_cols, wellbeing_cols)
    personality = scores['PERSONALITY'].values
    wellbeing = scores['WELLBEING'].values
    gpa = scores[outcome_col].values
    n = len(scores)

    results = []

    # Path 1: PERSONALITY → each behavioral composite
    for beh in behavior_cols:
        X = np.column_stack([np.ones(n), personality])
        y = scores[beh].values
        coefs, _, _, _ = lstsq(X, y, rcond=None)
        y_pred = X @ coefs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results.append({
            'From': 'PERSONALITY', 'To': beh.replace('_pc1', '').upper(),
            'path_coef': coefs[1], 'R²': r2,
        })

    # Path 2: PERSONALITY → WELLBEING
    X = np.column_stack([np.ones(n), personality])
    coefs, _, _, _ = lstsq(X, wellbeing, rcond=None)
    y_pred = X @ coefs
    r2 = 1 - np.sum((wellbeing - y_pred) ** 2) / np.sum((wellbeing - wellbeing.mean()) ** 2)
    results.append({
        'From': 'PERSONALITY', 'To': 'WELLBEING',
        'path_coef': coefs[1], 'R²': r2,
    })

    # Path 3: All behaviors + PERSONALITY + WELLBEING → GPA
    beh_data = np.column_stack([scores[b].values for b in behavior_cols])
    X_full = np.column_stack([np.ones(n), personality, wellbeing, beh_data])
    coefs_full, _, _, _ = lstsq(X_full, gpa, rcond=None)
    y_pred = X_full @ coefs_full
    r2_gpa = 1 - np.sum((gpa - y_pred) ** 2) / np.sum((gpa - gpa.mean()) ** 2)

    results.append({
        'From': 'PERSONALITY', 'To': 'GPA',
        'path_coef': coefs_full[1], 'R²': r2_gpa,
    })
    results.append({
        'From': 'WELLBEING', 'To': 'GPA',
        'path_coef': coefs_full[2], 'R²': r2_gpa,
    })
    for i, beh in enumerate(behavior_cols):
        results.append({
            'From': beh.replace('_pc1', '').upper(), 'To': 'GPA',
            'path_coef': coefs_full[3 + i], 'R²': r2_gpa,
        })

    # Path 4: Behaviors → WELLBEING (controlling for PERSONALITY)
    X_wb = np.column_stack([np.ones(n), personality, beh_data])
    coefs_wb, _, _, _ = lstsq(X_wb, wellbeing, rcond=None)
    y_pred = X_wb @ coefs_wb
    r2_wb = 1 - np.sum((wellbeing - y_pred) ** 2) / np.sum((wellbeing - wellbeing.mean()) ** 2)

    for i, beh in enumerate(behavior_cols):
        results.append({
            'From': beh.replace('_pc1', '').upper(), 'To': 'WELLBEING',
            'path_coef': coefs_wb[2 + i], 'R²': r2_wb,
        })

    return pd.DataFrame(results), r2_gpa, r2_wb


def bootstrap_paths(data, personality_cols, wellbeing_cols, behavior_cols, outcome_col,
                    n_boot=N_BOOT, seed=RANDOM_STATE):
    """Bootstrap for path coefficient CIs."""
    rng = np.random.RandomState(seed)
    n = len(data)

    # Collect all bootstrap path estimates
    all_boot = []
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_data = data.iloc[idx].reset_index(drop=True)
        try:
            paths_df, _, _ = estimate_paths(
                boot_data, personality_cols, wellbeing_cols, behavior_cols, outcome_col)
            all_boot.append(paths_df[['From', 'To', 'path_coef']])
        except Exception:
            continue

    if len(all_boot) == 0:
        return None

    boot_combined = pd.concat(all_boot, ignore_index=True)
    summary = boot_combined.groupby(['From', 'To'])['path_coef'].agg(
        boot_mean='mean', boot_std='std',
        ci_lo=lambda x: np.percentile(x, 2.5),
        ci_hi=lambda x: np.percentile(x, 97.5),
    ).reset_index()

    # p-value: proportion of bootstrap crossing zero
    p_vals = []
    for _, row in summary.iterrows():
        subset = boot_combined[
            (boot_combined['From'] == row['From']) &
            (boot_combined['To'] == row['To'])
        ]['path_coef']
        if row['boot_mean'] >= 0:
            p = 2 * np.mean(subset <= 0)
        else:
            p = 2 * np.mean(subset >= 0)
        p_vals.append(min(p, 1.0))
    summary['p_value'] = p_vals
    summary['sig'] = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                       for p in p_vals]

    return summary


def compute_effects(paths_df, behavior_cols):
    """Compute total, direct, and indirect effects."""
    beh_names = [b.replace('_pc1', '').upper() for b in behavior_cols]
    effects = []

    # PERSONALITY → GPA: direct + indirect via behaviors + indirect via wellbeing
    direct_pg = paths_df[
        (paths_df['From'] == 'PERSONALITY') & (paths_df['To'] == 'GPA')
    ]['path_coef'].values[0]

    # Indirect via each behavior
    total_indirect_beh = 0
    for beh_name in beh_names:
        a = paths_df[
            (paths_df['From'] == 'PERSONALITY') & (paths_df['To'] == beh_name)
        ]['path_coef'].values
        b = paths_df[
            (paths_df['From'] == beh_name) & (paths_df['To'] == 'GPA')
        ]['path_coef'].values
        if len(a) > 0 and len(b) > 0:
            indirect = a[0] * b[0]
            total_indirect_beh += indirect
            effects.append({
                'Effect': f'PERSONALITY → {beh_name} → GPA',
                'Type': 'specific_indirect',
                'Value': indirect,
            })

    # Indirect via WELLBEING
    a_pw = paths_df[
        (paths_df['From'] == 'PERSONALITY') & (paths_df['To'] == 'WELLBEING')
    ]['path_coef'].values[0]
    b_wg = paths_df[
        (paths_df['From'] == 'WELLBEING') & (paths_df['To'] == 'GPA')
    ]['path_coef'].values[0]
    indirect_wb = a_pw * b_wg

    effects.append({
        'Effect': 'PERSONALITY → WELLBEING → GPA',
        'Type': 'specific_indirect',
        'Value': indirect_wb,
    })

    total_indirect = total_indirect_beh + indirect_wb
    total_effect = direct_pg + total_indirect

    effects.extend([
        {'Effect': 'PERSONALITY → GPA (direct)', 'Type': 'direct', 'Value': direct_pg},
        {'Effect': 'PERSONALITY → GPA (total indirect)', 'Type': 'total_indirect', 'Value': total_indirect},
        {'Effect': 'PERSONALITY → GPA (total)', 'Type': 'total', 'Value': total_effect},
    ])

    return pd.DataFrame(effects)


def plot_path_diagram(paths_df, boot_summary, r2_gpa, r2_wb, output_path):
    """Visualize path model as a diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 9)
    ax.axis('off')

    # Node positions
    nodes = {
        'PERSONALITY': (0.5, 4.5),
        'MOBILITY': (4, 8), 'DIGITAL': (4, 7), 'SOCIAL': (4, 6),
        'ACTIVITY': (4, 5), 'SCREEN': (4, 4), 'PROXIMITY': (4, 3),
        'FACE2FACE': (4, 2), 'AUDIO': (4, 1),
        'WELLBEING': (7, 6.5),
        'GPA': (9.5, 4.5),
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        if name in ['PERSONALITY', 'WELLBEING']:
            circle = plt.Circle((x, y), 0.6, fill=True, facecolor='#bbdefb',
                               edgecolor='#1565c0', linewidth=2)
            ax.add_patch(circle)
        elif name == 'GPA':
            rect = mpatches.FancyBboxPatch(
                (x - 0.5, y - 0.35), 1.0, 0.7, boxstyle="round,pad=0.1",
                facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=2)
            ax.add_patch(rect)
            r2_text = f'\nR²={r2_gpa:.2f}'
            ax.text(x, y, f'{name}{r2_text}', ha='center', va='center',
                    fontsize=9, fontweight='bold')
            continue
        else:
            rect = mpatches.FancyBboxPatch(
                (x - 0.55, y - 0.3), 1.1, 0.6, boxstyle="round,pad=0.1",
                facecolor='#fff9c4', edgecolor='#f57f17', linewidth=1.5)
            ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw paths with coefficients
    if boot_summary is not None:
        merged = boot_summary
    else:
        merged = paths_df

    for _, row in merged.iterrows():
        from_name = row['From']
        to_name = row['To']
        coef = row['path_coef'] if 'path_coef' in row.index else row['boot_mean']
        sig = row.get('sig', '')

        if from_name in nodes and to_name in nodes:
            x1, y1 = nodes[from_name]
            x2, y2 = nodes[to_name]

            # Offset start/end for node boundaries
            dx, dy = x2 - x1, y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                x1_adj = x1 + 0.6 * dx / dist
                y1_adj = y1 + 0.6 * dy / dist
                x2_adj = x2 - 0.6 * dx / dist
                y2_adj = y2 - 0.6 * dy / dist
            else:
                continue

            color = '#d32f2f' if coef < 0 else '#1565c0'
            alpha = min(1.0, abs(coef) * 2 + 0.3)
            lw = max(0.5, abs(coef) * 3)

            if sig:
                ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                           arrowprops=dict(arrowstyle='->', color=color,
                                          lw=lw, alpha=alpha))
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y + 0.15, f'{coef:.2f}{sig}',
                       ha='center', va='bottom', fontsize=7, color=color,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                edgecolor='none', alpha=0.8))

    ax.set_title('PLS Path Model: Personality → Behavior → Wellbeing → GPA',
                fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_r_squared(paths_df, behavior_cols, r2_gpa, r2_wb, output_path):
    """Bar chart of R² values for endogenous constructs."""
    beh_names = [b.replace('_pc1', '').upper() for b in behavior_cols]

    r2_data = {'GPA': r2_gpa, 'WELLBEING': r2_wb}
    for beh_name in beh_names:
        row = paths_df[
            (paths_df['From'] == 'PERSONALITY') & (paths_df['To'] == beh_name)
        ]
        if len(row) > 0:
            r2_data[beh_name] = row['R²'].values[0]

    names = list(r2_data.keys())
    values = list(r2_data.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2e7d32' if v > 0.25 else '#ff8f00' if v > 0.1 else '#757575'
              for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor='white')
    ax.set_xlabel('R² (Variance Explained)')
    ax.set_title('Endogenous Construct R² Values')
    ax.axvline(0.25, color='green', linestyle='--', alpha=0.5, label='Moderate (0.25)')
    ax.axvline(0.50, color='blue', linestyle='--', alpha=0.5, label='Substantial (0.50)')
    ax.legend(fontsize=8)

    for bar, v in zip(bars, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
               f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("PLS-SEM STRUCTURAL EQUATION MODEL")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    # Prepare data
    print("\n[1/5] Preparing data...")
    data, personality_cols, wellbeing_cols, behavior_cols, outcome_col = prepare_data(df)

    # Estimate paths
    print("\n[2/5] Estimating structural paths...")
    paths_df, r2_gpa, r2_wb = estimate_paths(
        data, personality_cols, wellbeing_cols, behavior_cols, outcome_col)

    print(f"\n  R² GPA model: {r2_gpa:.3f}")
    print(f"  R² Wellbeing model: {r2_wb:.3f}")
    print("\n  Path Coefficients:")
    for _, row in paths_df.iterrows():
        print(f"    {row['From']:15s} → {row['To']:15s}: β = {row['path_coef']:.3f}")

    # Bootstrap
    print(f"\n[3/5] Bootstrap validation ({N_BOOT} resamples)...")
    boot_summary = bootstrap_paths(
        data, personality_cols, wellbeing_cols, behavior_cols, outcome_col)

    if boot_summary is not None:
        print("\n  Significant paths (p < 0.05):")
        sig_paths = boot_summary[boot_summary['sig'] != '']
        for _, row in sig_paths.iterrows():
            print(f"    {row['From']:15s} → {row['To']:15s}: "
                  f"β = {row['boot_mean']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] "
                  f"{row['sig']}")

        boot_summary.to_csv(TABLE_DIR / 'plssem_results.csv', index=False)

    # Effect decomposition
    print("\n[4/5] Effect decomposition...")
    effects = compute_effects(paths_df, behavior_cols)
    effects.to_csv(TABLE_DIR / 'plssem_effects.csv', index=False)
    print(effects.to_string(index=False))

    # Figures
    print("\n[5/5] Generating figures...")
    plot_path_diagram(paths_df, boot_summary, r2_gpa, r2_wb,
                      FIGURE_DIR / 'plssem_path_diagram.png')
    plot_r_squared(paths_df, behavior_cols, r2_gpa, r2_wb,
                   FIGURE_DIR / 'plssem_r_squared.png')

    print("\n" + "=" * 60)
    print("PLS-SEM ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
