# Cross-Study Synthesis: Meta-Analysis + Longitudinal

> Phase 14 | Scripts: `meta_analysis.py`, `globem_longitudinal.py`, `globem_comparison.py`, `nethealth_comparison.py`

## Meta-Analysis (Random-Effects, DerSimonian-Laird)

| Pool | k | N | Pooled r | 95% CI | I² |
|------|---|---|----------|--------|-----|
| **N → Anxiety/Stress** | 3 | 1,324 | **0.632*** | [0.384, 0.795] | 96.2% |
| **N → Depression** | 3 | 1,304 | **0.444*** | [0.235, 0.614] | 91.8% |
| **C → GPA** | 2 | 248 | **0.376*** | [0.064, 0.620] | 64.0% |
| N → Perceived Stress | 2 | 827 | 0.576* | [0.071, 0.846] | 88.1% |
| C → Depression | 3 | 1,304 | -0.209*** | [-0.291, -0.124] | 44.8% |

High I² expected: different instruments (PHQ-9 vs CES-D vs BDI-II) measure same construct differently.

**Output**: `results/comparison/meta_analysis.csv`, `results/comparison/meta_analysis_forest.png`

## Longitudinal Analyses (S3 GLOBEM)

### Weekly PHQ-4 Trajectory (N=540, ~10 weeks)

- **N → PHQ-4 mean level: r=+0.339*** — replicates cross-sectional finding**
- **N → PHQ-4 slope: r=-0.012, n.s.** — personality predicts *where you start*, not *whether you worsen*

**Output**: `results/globem/tables/phq4_trajectory.csv`, `results/globem/figures/figure16_phq4_trajectory.png`

### Pre → Post Change (N=748)

- All personality × Δ(Post-Pre) correlations near zero (all r < 0.10, all n.s.)
- Outcomes tested: ΔSTAI, ΔPSS-10, ΔCESD-10, ΔUCLA
- Interpretation: personality predicts stable between-person levels, not within-semester fluctuation

**Output**: `results/globem/tables/pre_post_change.csv`, `results/globem/figures/figure17_pre_post_change.png`

## Three-Study Replication Summary

| Finding | S1 | S2 | S3 | Consistent? |
|---------|----|----|-----|-------------|
| N → Depression (r) | +0.522 | +0.534 | +0.305 | YES |
| N → Anxiety (r) | +0.757 | +0.699 | +0.432 | YES |
| Pers → Depression R² | 0.071 | 0.284 | 0.087 | YES |
| Pers → Anxiety R² | 0.559 | 0.516 | 0.195 | YES |
| SHAP N=#1 (Depression) | — | 4/4 | 4/4 | YES |
| SHAP N=#1 (Anxiety) | — | 4/4 | 4/4 | YES |
| C=#1 for GPA | 4/4 | 4/4 | — | YES |
| Beh alone R² | ~0 | ≤0 | ≤0 | YES |

**6/6 MH findings consistent across 3 universities. Zero contradictory evidence.**

**Output**: `results/comparison/three_study_replication_summary.csv`, `results/comparison/three_study_mh_correlations.csv`, `results/comparison/three_study_ml_mh.csv`, `results/comparison/three_study_shap_mh.csv`, `results/comparison/three_study_forest_neuroticism.png`

## The Trait-Specific Dissociation

Across all three studies:
- **Conscientiousness → Academic performance** (SHAP #1 in 8/8 models)
- **Neuroticism → Mental health** (SHAP #1 in 16/16 models for depression/anxiety/stress)
- **Behavior → Neither** (R² ≤ 0 in S2/S3)
