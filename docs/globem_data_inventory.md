# GLOBEM — University of Washington — Data Inventory

> Institution: University of Washington
> Funding: NIH, NSF
> Period: 2018–2021 (4 annual cohorts, ~10 weeks per year)
> N: 497 unique participants (~705 person-years)
> Paper: Xu et al. "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling" (NeurIPS 2022)
> Data: https://physionet.org/content/globem/1.1/
> Access: **PhysioNet credentialed access** — sign DUA, free

---

## Overview

GLOBEM (Global-Multi-Year Behavioral Modeling) is a multi-year longitudinal mobile sensing dataset from the University of Washington, designed to study cross-dataset generalization in human behavior modeling. It combines smartphone passive sensing (via AWARE framework), Fitbit wearable data, and comprehensive psychological surveys across 4 annual cohorts of college students.

**Value for our project**: Has BFI-10 personality + BDI-II depression + rich smartphone + Fitbit sensing in N=497 college students. Strong complement to NetHealth for validating Personality→Mental Health and Behavior→Depression pathways.

---

## Available Data

### Smartphone Sensing (AWARE Framework)

| Sensor | Details |
|---|---|
| **GPS/Location** | Latitude, longitude, altitude, accuracy |
| **Screen status** | Screen on/off events, lock/unlock |
| **Bluetooth scans** | Nearby devices (social proximity) |
| **Phone calls** | Timestamped call logs |
| **WiFi** | Connected/scanned access points |

### Wearable Sensing (Fitbit)

| Measure | Details |
|---|---|
| **Steps** | Daily/intraday step count |
| **Heart rate** | Continuous heart rate monitoring |
| **Sleep** | Sleep duration, stages, efficiency |
| **Physical activity** | Active minutes, calories |

### Survey Instruments

| Measure | Acronym | Items | Frequency | Description |
|---|---|---|---|---|
| **Big Five Inventory** | **BFI-10** | 10 | Baseline | Big Five personality (short form) |
| **Beck Depression Inventory** | **BDI-II** | 21 | Per wave | Depression severity (gold standard) |
| **PHQ-4** | **PHQ-4** | 4 | EMA | Ultra-brief depression + anxiety screen |
| **Perceived Stress Scale** | **PSS-4** | 4 | Per wave | Perceived stress (short form) |
| **PANAS** | **PANAS** | 20 | Per wave | Positive and Negative Affect |
| **Emotion Regulation** | **ERQ** | 10 | Per wave | Emotion regulation strategies |
| **Physical Symptoms** | **CHIPS** | — | Per wave | Cohen-Hoberman physical symptoms |
| **Discrimination** | **EDS** | — | Per wave | Everyday Discrimination Scale |
| **Academic Fit** | — | — | Per wave | Sense of Social and Academic Fit |

---

## Detailed Variable Inventory

### Personality

| Measure | Items | Scales | Notes |
|---|---|---|---|
| **BFI-10** | 10 items | Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness | Short form (2 items per dimension). **NOT directly comparable to BFI-44** — reduced reliability, broader constructs |

### Mental Health

| Measure | What It Measures | Notes |
|---|---|---|
| **BDI-II** | Depression severity (0-63) | Gold standard depression measure; more comprehensive than PHQ-9 |
| **PHQ-4** | Depression + Anxiety screen | Ultra-brief 4-item screener (2 depression + 2 anxiety) |
| **PSS-4** | Perceived stress | Short form of PSS-10 (4 items vs our 10 items) |
| **PANAS** | Positive/Negative affect | Same instrument as StudentLife |

### Academic

| Measure | Details |
|---|---|
| **Sense of Academic Fit** | Self-reported belonging and fit in academic environment |
| **No raw GPA** | No registrar or self-reported GPA available |

---

## Comparison with StudentLife (Our Dataset)

| Measure | StudentLife (N=28) | GLOBEM (N=497) | Compatible |
|---|---|---|---|
| **Big Five** | BFI-44 (44 items) | BFI-10 (10 items) | ⚠️ Same construct, different length |
| **GPA** | Registrar | **Not collected** | **No** |
| **Depression** | PHQ-9 | BDI-II + PHQ-4 | ⚠️ Different instruments, same construct |
| **Stress** | PSS-10 | PSS-4 | ⚠️ Short form vs full |
| **Affect** | PANAS-20 | PANAS-20 | ✅ Same instrument |
| **Loneliness** | UCLA-20 | Not collected | No |
| **Flourishing** | Flourishing Scale | Not collected | No |
| **Phone sensors** | StudentLife app (13 modalities) | AWARE (GPS/screen/BT/calls/WiFi) | ⚠️ Partial overlap |
| **Wearable** | Not collected | Fitbit (steps/HR/sleep) | New modality |
| **Duration** | 10 weeks | ~10 weeks × 4 years | GLOBEM much longer |

---

## What GLOBEM Can Validate from Our Study

### Directly Validable

| Our Finding | GLOBEM Variables | Validation Approach |
|---|---|---|
| **Personality → Mental Health** | BFI-10 + BDI-II | Test with N=497 (adequate power) |
| **Behavior → Depression** (R²=0.468) | Phone/Fitbit + BDI-II | Replicate with richer behavioral data |
| **Behavior → Stress** | Phone/Fitbit + PSS-4 | Replicate with N=497 |
| **Behavior → Affect** | Phone/Fitbit + PANAS | Same instrument, direct replication |
| **LPA behavioral profiles** | Phone/Fitbit features | Replicate clustering with N=497 |
| **ML framework** (SHAP, Optuna, cross-model) | All variables | Full methodological replication |
| **Cross-model consistency** (Kendall's τ) | All variables | 4 models × multiple outcomes |

### Not Validable

| Our Finding | Reason |
|---|---|
| **Personality → GPA** (core finding) | No GPA in GLOBEM |
| **Conscientiousness = #1 GPA predictor** | No GPA |
| **Personality moderates behavior → GPA** | No GPA |
| **BFI-44 specific findings** | BFI-10 has lower reliability per dimension |

---

## Key Advantages of GLOBEM

1. **N=497** — 18× our sample; adequate power for virtually all analyses
2. **Both smartphone + Fitbit** — bridges the gap between phone-only (StudentLife) and Fitbit-only (NetHealth)
3. **4-year longitudinal** — can test temporal stability and cross-cohort replication
4. **BDI-II** — gold standard depression measure, more reliable than PHQ-9
5. **PANAS** — same instrument as StudentLife, enables direct comparison
6. **PhysioNet access** — credentialed but free, well-established process
7. **NeurIPS venue** — high-impact ML venue, adds credibility to methodology

## Limitations

1. **No GPA** — cannot validate our core Personality→GPA finding
2. **BFI-10 vs BFI-44** — short form has lower reliability; 2 items per dimension means Conscientiousness score is based on only 2 items (vs 9 in BFI-44)
3. **AWARE ≠ StudentLife** — different sensing framework, different feature extraction
4. **UW vs Dartmouth** — different institution, different student demographics
5. **PSS-4 vs PSS-10** — short form may miss nuance

---

## Access Instructions

1. Create a PhysioNet account at https://physionet.org/
2. Complete the CITI training for human subjects research
3. Sign the Data Use Agreement for GLOBEM
4. Download from https://physionet.org/content/globem/1.1/
5. Data available in CSV format with documentation

---

## Impact on Publication Strategy

GLOBEM is best positioned as a **complementary validation dataset** alongside NetHealth:

| Role | Dataset | Validates |
|---|---|---|
| **Discovery** | StudentLife (N=28) | Full 8-method exploratory analysis |
| **Personality→GPA validation** | NetHealth (N=722) | Core finding + SHAP + moderation |
| **Behavior→Depression validation** | GLOBEM (N=497) | Behavior→PHQ pathway + LPA + ML framework |

This three-dataset structure gives the paper a powerful "triangulation" narrative across different institutions, instruments, and sensing modalities.
