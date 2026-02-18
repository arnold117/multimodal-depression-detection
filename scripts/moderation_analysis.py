#!/usr/bin/env python3
"""
Moderation Analysis: Does Personality Moderate Behavior → Outcome Links?

Research Question: Does the effect of smartphone behavior on GPA/wellbeing
depend on personality traits? (e.g., high screen time may only hurt GPA
for low-conscientiousness students)

Method:
  - Hierarchical regression (Step 1: main effects, Step 2: + interaction)
  - Report ΔR² and interaction β with bootstrap CIs
  - Simple slopes at ±1 SD of moderator
  - Visualization: simple slopes plots

Input:  data/processed/analysis_dataset.parquet
Output: results/tables/moderation_results.csv
        results/figures/moderation_simple_slopes.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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

# Moderators (personality traits)
MODERATORS = ['conscientiousness', 'neuroticism', 'extraversion',
              'agreeableness', 'openness']

# Focal predictors (behavioral composites)
BEHAVIORS = ['digital_pc1', 'mobility_pc1', 'screen_pc1', 'activity_pc1',
             'proximity_pc1', 'social_pc1', 'face2face_pc1', 'audio_pc1']

# Outcomes
OUTCOMES = {'gpa_overall': 'GPA', 'pss_total': 'PSS',
            'loneliness_total': 'Loneliness'}

N_BOOT = 5000
RANDOM_STATE = 42


def hierarchical_regression(X_step1, X_step2, y):
    """Compare two nested OLS models. Returns R², ΔR², F-change, p."""
    from numpy.linalg import lstsq

    n = len(y)

    # Step 1
    X1 = np.column_stack([np.ones(n), X_step1])
    b1, _, _, _ = lstsq(X1, y, rcond=None)
    y_pred1 = X1 @ b1
    ss_res1 = np.sum((y - y_pred1) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_1 = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0

    # Step 2
    X2 = np.column_stack([np.ones(n), X_step2])
    b2, _, _, _ = lstsq(X2, y, rcond=None)
    y_pred2 = X2 @ b2
    ss_res2 = np.sum((y - y_pred2) ** 2)
    r2_2 = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0

    # ΔR² and F-change
    delta_r2 = r2_2 - r2_1
    p1 = X_step1.shape[1] if X_step1.ndim > 1 else 1
    p2 = X_step2.shape[1] if X_step2.ndim > 1 else 1
    df_num = p2 - p1
    df_den = n - p2 - 1

    if df_den > 0 and ss_res2 > 0 and df_num > 0:
        f_change = (delta_r2 / df_num) / ((1 - r2_2) / df_den)
        p_change = 1 - sp_stats.f.cdf(f_change, df_num, df_den)
    else:
        f_change = 0
        p_change = 1.0

    return {
        'r2_step1': r2_1, 'r2_step2': r2_2, 'delta_r2': delta_r2,
        'f_change': f_change, 'p_change': p_change,
        'coefs': b2,  # [intercept, X_pred, moderator, interaction]
    }


def bootstrap_interaction(x_pred, x_mod, y, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Bootstrap CIs for interaction coefficient."""
    from numpy.linalg import lstsq

    rng = np.random.RandomState(seed)
    n = len(y)
    interaction = x_pred * x_mod
    X = np.column_stack([np.ones(n), x_pred, x_mod, interaction])

    # Point estimate
    b, _, _, _ = lstsq(X, y, rcond=None)
    interaction_coef = b[3]

    # Bootstrap
    boot_coefs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        b_boot, _, _, _ = lstsq(X[idx], y[idx], rcond=None)
        boot_coefs[i] = b_boot[3]

    ci = np.percentile(boot_coefs, [2.5, 97.5])
    sig = (ci[0] > 0) or (ci[1] < 0)

    return {
        'interaction_beta': interaction_coef,
        'ci_lo': ci[0], 'ci_hi': ci[1],
        'significant': sig,
        'boot_coefs': boot_coefs,
    }


def simple_slopes(x_pred, x_mod, y):
    """Compute simple slopes at low (-1SD), mean, high (+1SD) of moderator."""
    from numpy.linalg import lstsq

    n = len(y)
    interaction = x_pred * x_mod
    X = np.column_stack([np.ones(n), x_pred, x_mod, interaction])
    b, _, _, _ = lstsq(X, y, rcond=None)

    # Simple slope of x_pred at different levels of moderator
    # y = b0 + b1*x_pred + b2*x_mod + b3*x_pred*x_mod
    # dy/dx_pred = b1 + b3 * x_mod
    slopes = {}
    for label, mod_val in [('Low (-1SD)', -1), ('Mean', 0), ('High (+1SD)', 1)]:
        slope = b[1] + b[3] * mod_val
        # SE via delta method approximation
        slopes[label] = {'slope': slope, 'mod_value': mod_val}

    return slopes, b


def run_moderation(df):
    """Test all moderator × behavior → outcome combinations."""
    print("\n[1/2] Running Moderation Analyses")
    print("─" * 60)

    rows = []

    for outcome, outcome_label in OUTCOMES.items():
        for moderator in MODERATORS:
            for behavior in BEHAVIORS:
                cols = [moderator, behavior, outcome]
                subset = df[cols].dropna()
                if len(subset) < 15:
                    continue

                # Standardize
                scaler = StandardScaler()
                z = pd.DataFrame(
                    scaler.fit_transform(subset),
                    columns=cols, index=subset.index)
                x_pred = z[behavior].values
                x_mod = z[moderator].values
                y = z[outcome].values

                # Hierarchical regression
                X_step1 = np.column_stack([x_pred, x_mod])
                interaction = x_pred * x_mod
                X_step2 = np.column_stack([x_pred, x_mod, interaction])
                hier = hierarchical_regression(X_step1, X_step2, y)

                # Bootstrap interaction CI
                boot = bootstrap_interaction(x_pred, x_mod, y)

                # Simple slopes
                ss, _ = simple_slopes(x_pred, x_mod, y)

                sig_marker = '*' if boot['significant'] else ''

                rows.append({
                    'Outcome': outcome, 'Outcome_Label': outcome_label,
                    'Moderator': moderator, 'Behavior': behavior,
                    'N': len(subset),
                    'R2_main': hier['r2_step1'],
                    'R2_interaction': hier['r2_step2'],
                    'Delta_R2': hier['delta_r2'],
                    'F_change': hier['f_change'],
                    'p_change': hier['p_change'],
                    'Interaction_beta': boot['interaction_beta'],
                    'CI_lo': boot['ci_lo'],
                    'CI_hi': boot['ci_hi'],
                    'Slope_low': ss['Low (-1SD)']['slope'],
                    'Slope_mean': ss['Mean']['slope'],
                    'Slope_high': ss['High (+1SD)']['slope'],
                    'sig': sig_marker,
                })

    results_df = pd.DataFrame(rows)

    # Report significant interactions
    sig = results_df[results_df['sig'] == '*']
    print(f"  Total tests: {len(results_df)}")
    print(f"  Significant interactions (bootstrap 95% CI): {len(sig)}")

    # Show top by |ΔR²|
    top = results_df.nlargest(10, 'Delta_R2')
    for _, row in top.iterrows():
        print(f"  {row['Moderator']:18s} × {row['Behavior']:15s} → {row['Outcome_Label']:12s}  "
              f"ΔR²={row['Delta_R2']:.3f} β={row['Interaction_beta']:.3f} "
              f"[{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] {row['sig']}")

    return results_df


def plot_simple_slopes(df, results_df, output_path):
    """Simple slopes plots for top interactions."""
    # Select top interactions by |ΔR²|
    top = results_df.nlargest(6, 'Delta_R2')

    n_plots = min(len(top), 6)
    if n_plots == 0:
        print("  No interactions to plot")
        return

    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (_, row) in enumerate(top.iterrows()):
        if idx >= n_plots:
            break

        ax = axes[idx]
        moderator = row['Moderator']
        behavior = row['Behavior']
        outcome = row['Outcome']

        subset = df[[moderator, behavior, outcome]].dropna()
        scaler = StandardScaler()
        z = pd.DataFrame(scaler.fit_transform(subset),
                         columns=[moderator, behavior, outcome],
                         index=subset.index)

        # Split by moderator level
        mod_vals = z[moderator]
        low_mask = mod_vals <= -0.5
        high_mask = mod_vals >= 0.5

        x_range = np.linspace(z[behavior].min(), z[behavior].max(), 50)

        # Compute regression lines at ±1SD
        from numpy.linalg import lstsq
        x_pred = z[behavior].values
        x_mod = z[moderator].values
        y = z[outcome].values
        interaction = x_pred * x_mod
        X = np.column_stack([np.ones(len(y)), x_pred, x_mod, interaction])
        b, _, _, _ = lstsq(X, y, rcond=None)

        for mod_level, label, color in [(-1, f'Low {moderator[:4]}', '#d32f2f'),
                                         (1, f'High {moderator[:4]}', '#1565c0')]:
            y_line = b[0] + b[1] * x_range + b[2] * mod_level + b[3] * x_range * mod_level
            ax.plot(x_range, y_line, color=color, linewidth=2, label=label)

        # Scatter by group
        if low_mask.sum() > 0:
            ax.scatter(z.loc[low_mask, behavior], z.loc[low_mask, outcome],
                      c='#d32f2f', alpha=0.4, s=30)
        if high_mask.sum() > 0:
            ax.scatter(z.loc[high_mask, behavior], z.loc[high_mask, outcome],
                      c='#1565c0', alpha=0.4, s=30)

        sig_str = ' *' if row['sig'] == '*' else ''
        ax.set_title(f"{moderator[:5]} × {behavior.replace('_pc1', '')}"
                     f" → {row['Outcome_Label']}\n"
                     f"ΔR²={row['Delta_R2']:.3f}{sig_str}", fontsize=9)
        ax.set_xlabel(behavior.replace('_pc1', ''), fontsize=8)
        ax.set_ylabel(row['Outcome_Label'], fontsize=8)
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Moderation: Personality × Behavior → Outcome\n'
                 '(Simple slopes at ±1SD of moderator)', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("MODERATION ANALYSIS")
    print("=" * 60)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Dataset: {len(df)} participants × {len(df.columns)} variables")

    # Run moderation
    results_df = run_moderation(df)
    results_df.to_csv(TABLE_DIR / 'moderation_results.csv', index=False)

    # Plots
    print("\n[2/2] Generating figures...")
    plot_simple_slopes(df, results_df, FIGURE_DIR / 'moderation_simple_slopes.png')

    print("\n" + "=" * 60)
    print("MODERATION ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
