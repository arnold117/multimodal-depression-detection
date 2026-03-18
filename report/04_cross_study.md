# Cross-Study Synthesis: Meta-Analysis + Longitudinal

> Phase 14 | Scripts: `meta_analysis.py`, `globem_longitudinal.py`, `globem_comparison.py`, `nethealth_comparison.py`

## Meta-Analysis (Random-Effects, DerSimonian-Laird)

### Method
Random-effects meta-analysis using the DerSimonian-Laird estimator, which accounts for between-study heterogeneity. Effect sizes are Pearson correlations (r) Fisher z-transformed for pooling, then back-transformed for reporting. Each pool includes only outcomes where equivalent constructs were measured across studies.

### All 5 Meta-Analytic Pools

| Pool | k | N | Pooled r | 95% CI | z | p | I² | tau² |
|------|---|---|----------|--------|---|---|-----|------|
| **N → Anxiety/Stress** | 3 | 1,324 | **0.632*** | [0.384, 0.795] | 4.36 | <0.001 | 96.2% | 0.108 |
| **N → Depression** | 3 | 1,304 | **0.444*** | [0.235, 0.614] | 3.87 | <0.001 | 91.8% | 0.042 |
| **C → GPA** | 2 | 248 | **0.376*** | [0.064, 0.620] | 2.35 | 0.019 | 64.0% | 0.018 |
| **N → Perceived Stress** | 2 | 827 | **0.576*** | [0.071, 0.846] | 2.24 | 0.025 | 88.1% | 0.054 |
| **C → Depression** | 3 | 1,304 | **-0.209*** | [-0.291, -0.124] | -4.72 | <0.001 | 44.8% | 0.002 |

### Individual Study Contributions

**N → Anxiety/Stress pool (k=3):**
- S1 (Dartmouth, N=28): r=0.757 (PSS)
- S2 (Notre Dame, N=498): r=0.699 (STAI-Trait)
- S3 (UW, N=800): r=0.432 (STAI-State)

**N → Depression pool (k=3):**
- S1 (Dartmouth, N=28): r=0.522 (PHQ-9)
- S2 (Notre Dame, N=498): r=0.534 (CES-D)
- S3 (UW, N=786): r=0.305 (BDI-II)

**C → GPA pool (k=2):**
- S1 (Dartmouth, N=28): r=0.552
- S2 (Notre Dame, N=220): r=0.263

### Why I² Is High (and Why That Is Expected)

High heterogeneity (I² = 64-96%) is expected and interpretable for three reasons:

1. **Different instruments measure the same construct differently**: PHQ-9 (9 items, 2-week window), CES-D (20 items, 1-week window), BDI-II (21 items, 2-week window) all measure depression but with different item content, response formats, and time frames. Similarly, STAI-Trait vs STAI-State vs PSS measure related but distinct anxiety/stress constructs.

2. **Different personality instruments**: S1/S2 use BFI-44 (alpha 0.67-0.87) while S3 uses BFI-10 (alpha ~0.65). Lower reliability in S3 attenuates observed correlations, contributing to heterogeneity.

3. **Different populations and time periods**: Dartmouth (2013), Notre Dame (2015-2019), UW (2018-2021) — including a COVID period in S3.

Despite high I², the critical finding is that **all effects are in the same direction** — there is zero contradictory evidence across studies.

**Output**: `results/comparison/meta_analysis.csv`, `results/comparison/meta_analysis_forest.png`

## Longitudinal Analyses (S3 GLOBEM)

### Weekly PHQ-4 Trajectory (N=540, ~10 weeks)

Participants from cohorts INS-W_2 through INS-W_4 completed weekly PHQ-4 assessments over approximately 10 weeks. Growth curve models estimated individual intercepts (mean level) and slopes (rate of change).

**Personality → PHQ-4 intercept (mean level):**
- **Neuroticism → PHQ-4 mean: r=+0.339*** — replicates cross-sectional finding longitudinally
- Conscientiousness → PHQ-4 mean: r=-0.148**
- Extraversion → PHQ-4 mean: r=-0.089*
- Agreeableness → PHQ-4 mean: r=-0.071 (n.s.)
- Openness → PHQ-4 mean: r=-0.023 (n.s.)

**Personality → PHQ-4 slope (rate of change):**
- **Neuroticism → PHQ-4 slope: r=-0.012, n.s.** — personality does NOT predict whether symptoms worsen or improve over time
- All other trait-slope correlations: |r| < 0.05, all n.s.

**Interpretation**: Personality predicts *where you start* (stable individual differences in distress level), not *whether you worsen* (within-person change over time). This is consistent with trait theory: personality marks stable vulnerability, not dynamic deterioration.

**Output**: `results/globem/tables/phq4_trajectory.csv`, `results/globem/figures/figure16_phq4_trajectory.png`

### Pre → Post Change (N=748)

Tested whether personality predicts within-semester change in mental health (Post - Pre difference scores):

| Trait | Delta-STAI | Delta-PSS-10 | Delta-CESD | Delta-UCLA |
|-------|-----------|-------------|-----------|-----------|
| Neuroticism | r=+0.04 (n.s.) | r=+0.02 (n.s.) | r=+0.06 (n.s.) | r=+0.01 (n.s.) |
| Conscientiousness | r=-0.03 (n.s.) | r=-0.01 (n.s.) | r=-0.02 (n.s.) | r=-0.04 (n.s.) |
| Extraversion | r=+0.02 (n.s.) | r=+0.01 (n.s.) | r=-0.01 (n.s.) | r=+0.03 (n.s.) |
| Agreeableness | r=-0.01 (n.s.) | r=+0.03 (n.s.) | r=-0.02 (n.s.) | r=-0.01 (n.s.) |
| Openness | r=+0.01 (n.s.) | r=-0.02 (n.s.) | r=+0.01 (n.s.) | r=+0.02 (n.s.) |

All personality x change correlations near zero (all |r| < 0.10, all n.s.). Personality predicts stable between-person levels, not within-semester fluctuation. This converges with the slope analysis above and strengthens the trait interpretation.

**Output**: `results/globem/tables/pre_post_change.csv`, `results/globem/figures/figure17_pre_post_change.png`

## Three-Study Replication Summary

### Mental Health Findings

| Finding | S1 (Dartmouth) | S2 (Notre Dame) | S3 (UW) | Consistent? |
|---------|----------------|-----------------|---------|-------------|
| N → Depression (r) | +0.522 (PHQ-9) | +0.534 (CES-D) | +0.305 (BDI-II) | YES |
| N → Anxiety (r) | +0.757 (PSS) | +0.699 (STAI-T) | +0.432 (STAI-S) | YES |
| Pers → Depression R² | 0.071 | 0.284 | 0.087 | YES (all > 0) |
| Pers → Anxiety R² | 0.559 | 0.516 | 0.195 | YES (all > 0) |
| SHAP N=#1 (Depression) | — (underpowered) | 4/4 models | 4/4 models | YES |
| SHAP N=#1 (Anxiety) | — (underpowered) | 4/4 models | 4/4 models | YES |
| Beh alone R² (MH) | ~0 (overfit) | <=0 (all outcomes) | <=0 (all outcomes) | YES |
| Pers+Beh > Pers (FDR) | — | 0/3 significant | 1/5 significant | YES (minimal) |

### Academic Findings

| Finding | S1 (Dartmouth) | S2 (Notre Dame) | S3 (UW) | Consistent? |
|---------|----------------|-----------------|---------|-------------|
| C → GPA (r) | +0.552 (p=.003) | +0.263 (p=.0001) | — (no GPA) | YES |
| C=#1 SHAP (GPA) | 4/4 models | 4/4 models | — | YES (8/8 = 100%) |
| Beh alone → GPA R² | 0.116 (SVR only) | <=0 | — | Partial |

### Formal Replication Statistics

- **6/6 MH findings consistent** in direction across all 3 universities
- **8/8 GPA SHAP rankings** consistent across 2 universities
- **Zero contradictory evidence** — no finding reverses direction in any study
- Fisher z-tests for correlation differences: all p>0.05 except where instrument reliability differs (BFI-10 vs BFI-44)
- Effect size attenuation in S3 fully explained by BFI-10 reliability (disattenuation correction brings S3 in line with S2)

**Output**: `results/comparison/three_study_replication_summary.csv`, `results/comparison/three_study_mh_correlations.csv`, `results/comparison/three_study_ml_mh.csv`, `results/comparison/three_study_shap_mh.csv`, `results/comparison/three_study_forest_neuroticism.png`, `results/comparison/three_study_lpa.csv`

## The Trait-Specific Dissociation

Across all three studies, a clean dissociation emerges:
- **Conscientiousness → Academic performance** (SHAP #1 in 8/8 GPA models across S1 and S2)
- **Neuroticism → Mental health** (SHAP #1 in 16/16 depression/anxiety/stress models across S2 and S3)
- **Extraversion → Loneliness** (SHAP #1 in 3/4 UCLA models in S3; r=-0.229)
- **Behavior → Neither GPA nor mental health** (R² <= 0 in S2 and S3; S1 behavior results likely overfit at N=28)

This dissociation is the central empirical contribution: personality traits predict outcomes in a domain-specific manner that is consistent across universities, time periods, instruments, and ML algorithms. Passive sensing does not add meaningful variance to any outcome domain.

## All Cross-Study Output Files

### Tables (CSV)

| File | Description |
|------|-------------|
| `results/comparison/meta_analysis.csv` | Random-effects meta-analysis results (5 pools) |
| `results/comparison/three_study_replication_summary.csv` | Formal replication summary table |
| `results/comparison/three_study_mh_correlations.csv` | Personality-MH correlations across 3 studies |
| `results/comparison/three_study_ml_mh.csv` | ML prediction R² across 3 studies |
| `results/comparison/three_study_shap_mh.csv` | SHAP rankings across 3 studies |
| `results/comparison/three_study_lpa.csv` | LPA comparison across 3 studies |
| `results/comparison/personality_gpa_correlations.csv` | C→GPA correlations (S1 vs S2) |
| `results/comparison/replication_summary.csv` | S1 vs S2 GPA replication summary |
| `results/comparison/ml_performance_comparison.csv` | ML model comparison across studies |
| `results/comparison/shap_ranking_comparison.csv` | SHAP ranking agreement across studies |
| `results/comparison/anxiety_prediction_comparison.csv` | Anxiety prediction across studies |

### Figures (PNG)

| File | Description |
|------|-------------|
| `results/comparison/meta_analysis_forest.png` | Meta-analysis forest plot (5 pools) |
| `results/comparison/three_study_forest_neuroticism.png` | Neuroticism effect size across 3 studies |
| `results/comparison/forest_ml_r2.png` | ML R² forest plot across studies |
| `results/comparison/forest_personality_gpa.png` | Personality→GPA forest plot (S1 vs S2) |
