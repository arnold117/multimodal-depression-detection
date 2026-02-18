# College Experience Study (CES) — Data Inventory

> NDA Collection #2494 | Dartmouth College | 2017–2022 | N=200+
> Source: https://nda.nih.gov/edit_collection.html?id=2494
> Kaggle: https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset

## Overview

The College Experience Study (CES) is a 5-year longitudinal mobile sensing study tracking 200+ Dartmouth undergraduates (two cohorts of first-year students) using the same StudentLife app as our current dataset. Data includes continuous smartphone sensing, weekly EMAs, and yearly surveys, spanning both on-campus and off-campus periods including COVID-19.

**Principal Investigator**: Andrew T. Campbell, Dartmouth CS
**Key Publication**: Nepal et al. (2024) "Capturing the College Experience" (IMWUT/UbiComp)

---

## Data Structures in NDA Collection #2494

| # | Data Structure | Short Name | N (expected) | N (submitted) | Status |
|---|---|---|---|---|---|
| 1 | Perceived Stress Scale | pss01 | 200 | 214 | Approved |
| 2 | Patient Health Questionnaire | phq01 | 200 | 214 | Approved |
| 3 | Generalized Anxiety Disorder Screener | gad01 | 200 | 215 | Approved |
| 4 | State Self-Esteem Scale | sse01 | 104 | 213 | Approved |
| 5 | College Experience | ces01 | 99 | 175 | Approved |
| 6 | Janis-Field Feelings of Inadequacy Scale | jfis01 | 200 | 214 | Approved |
| 7 | Medical History Health Rating | medhxrat01 | 200 | 111 | Approved |
| 8 | Imaging (Structural, fMRI, DTI) | image03 | 144 | 296 | Approved |

---

## Detailed Variable Inventory

### 1. College Experience (`ces01`) — N=175

Demographics and academic information.

| Variable | Type | Description |
|---|---|---|
| `college_gpa` | Integer (0–6) | Cumulative GPA, categorical: 0=≤1.5, 1=1.6-2.0, 2=2.1-2.5, 3=2.6-3.0, 4=3.1-3.5, 5=3.6-4.0, 6=first semester |
| **`academ_gpa`** | **Float** | **Continuous cumulative GPA — directly usable** |
| `academ_1` | Likert 1–5 | I attend all my courses |
| `academ_2` | Likert 1–5 | I pay attention and participate in class |
| `academ_3` | Likert 1–5 | I complete homework on time |
| `academ_4` | Likert 1–5 | I prepare for examinations |
| `academ_5` | Likert 1–5 | My GPA reflects my effort |
| `academ_mhealth` | Likert 1–5 | My mental health impacts my GPA |
| `academ_total` | Integer 5–25 | Academic Competence Total (sum of academ_1 to academ_5) |
| `college_yr` | Integer 1–5 | Year in college |
| `college_hous` | Integer 1–4 | Housing type (on-campus, off-campus, family, other) |
| `college_extra1–11` | Binary 0/1 | Extracurricular activities (fraternity, athletics, clubs, etc.) |
| `cey1_8` | Likert 0–4 | Self-rated physical wellbeing |
| `cey1_9` | Likert 0–4 | Self-rated mental wellbeing |
| `cey1_10` | Likert 0–4 | Physical activity level |
| `cey1_11` | Likert 0–4 | Sleep pattern healthiness |
| `cey1_12` | Likert 0–4 | Eating pattern healthiness |
| `cey1_13` | Likert 0–4 | Social life satisfaction |
| `cey1_14` | Likert 0–4 | Lonely or homesick (frequency) |
| `cey1_15` | Likert 0–4 | Isolated from school life (frequency) |
| `cey1_24` | Likert 0–4 | Summer stress level |
| `cey1_42` | Integer 0–3 | Most academically demanding term |
| `piq_9–27` | Various | Personal information (Greek life, sports, social behavior) |
| Demographics | Various | Race, ethnicity, age, sex, SES, parental education, political orientation |

### 2. Patient Health Questionnaire (`phq01`) — N=214

Depression screening. Multiple versions available.

| Variable | Type | Description |
|---|---|---|
| **PHQ-9 (2-week)** | | |
| `anhedonia` | Likert 0–3 | Little interest or pleasure |
| `down` | Likert 0–3 | Feeling down, depressed, hopeless |
| `sleep_trouble` | Likert 0–3 | Trouble sleeping |
| `phq_tired` | Likert 0–3 | Tired or low energy |
| `appetite` | Likert 0–3 | Poor appetite or overeating |
| `failure` | Likert 0–3 | Feeling bad about yourself |
| `concentration_problems` | Likert 0–3 | Trouble concentrating |
| `psychomotor_retardation` | Likert 0–3 | Moving/speaking slowly or restlessly |
| `si_hi` | Likert 0–3 | Thoughts of self-harm |
| `phq_9_total` | Integer 0–27 | **PHQ-9 total score** |
| `phq8_score` | Integer 0–24 | PHQ-8 total (excludes item 9) |
| **PHQ-9 (weekly)** | | |
| `phq_v8_1` to `phq_v8_8` | Likert 0–3 | Weekly PHQ items |
| **PHQ-9 (monthly)** | | |
| `phq_month_1` to `phq_month_9` | Likert 0–3 | Monthly PHQ items |
| **PHQ-4 (ultra-brief)** | | |
| `phq2totalscore` | Integer 0–6 | PHQ-2 depression screen |
| `gad7_01`, `gad7_02` | Likert 0–3 | GAD-2 anxiety screen (embedded) |
| **Functional impairment** | | |
| `difficulty_functioning` | Likert 0–3 | Difficulty at work/home/social |
| **Diagnostic flags** | | |
| `phqmdddx_t1` | Binary 0/1 | PHQ MDD diagnosis |
| `phqpanicdx_t1` | Binary 0/1 | PHQ Panic diagnosis |
| `phqgaddx_t1` | Binary 0/1 | PHQ GAD diagnosis |

### 3. Perceived Stress Scale (`pss01`) — N=214

Stress measurement. Multiple time windows.

| Variable | Type | Description |
|---|---|---|
| **PSS-10 (weekly)** | | |
| `pss_1` to `pss_10` | Likert 0–4 | Standard PSS-10 items (past week) |
| `pss_totalscore` | Integer 0–40 | **PSS total score** |
| **PSS-10 (24-hour)** | | |
| `pss1` to `pss10` | Likert 0–4 | Daily PSS items |
| **PSS-10 (2-week)** | | |
| `pss_2wk_1` to `pss_2wk_10` | Likert 0–4 | Biweekly PSS items |
| **PSS-10 (monthly)** | | |
| `pssp1_1` to `pssp3_4` | Likert 0–4 | Monthly PSS items |
| `pss10_total_rs` | Integer 0–40 | Monthly PSS total |
| **PSS-10 (90-day)** | | |
| `pss_quarter_1` to `pss_quarter_10` | Likert 0–4 | Quarterly PSS items |
| **Subscales** | | |
| `pss_distress_rs` | Integer | General Distress subscale |
| `pss_cope_rs` | Integer | Ability to Cope subscale |

### 4. Generalized Anxiety Disorder Screener (`gad01`) — N=215

Anxiety measurement.

| Variable | Type | Description |
|---|---|---|
| **GAD-7 (2-week)** | | |
| `gad7_01` to `gad7_07` | Likert 0–3 | Standard GAD-7 items |
| `gad7_total` | Float 0–21 | **GAD-7 total score** (5=mild, 10=moderate, 15=severe) |
| **GAD-7 (4-week)** | | |
| `gad_7_1` to `gad7` | Likert 0–3 | Monthly GAD items |
| **GAD-7 (24-hour)** | | |
| `gad7_001` to `gad7_007` | Likert 0–3 | Daily GAD items |
| **GAD-7 (lifetime)** | | |
| `gad7_q1_lifetime` to `gad7_q7_lifetime` | Likert 0–3 | Lifetime GAD items |
| **GAD-2 (ultra-brief)** | | |
| `gad2_total` | Integer 0–6 | GAD-2 screen (≥3 = probable anxiety) |
| **Functional impairment** | | |
| `gad_subj` | Likert 0–3 | Difficulty at work/home/social |

### 5. State Self-Esteem Scale (`sse01`) — N=213

Self-esteem with three subscales + social connectedness items.

| Variable | Type | Description |
|---|---|---|
| **Self-Esteem (20 items)** | | |
| `sse_1` to `sse_20` | Likert 1–5 | State Self-Esteem items |
| `sse_per` | Integer 7–35 | **Performance Self-Esteem subscale** |
| `sse_soc` | Integer 7–35 | **Social Self-Esteem subscale** |
| `sse_app` | Integer 6–30 | **Appearance Self-Esteem subscale** |
| `sse_total` | Integer 20–100 | **Overall Self-Esteem total** |
| **Social Connectedness (12 items)** | | |
| `sse_21` | Likert 0–4 | Feel like being around people |
| `sse_22` | Likert 0–4 | Feel like being alone |
| `sse_23` | Likert 0–4 | Feel outgoing and friendly |
| `sse_24` | Likert 0–4 | Feel shy |
| `sse_25` | Likert 0–4 | Feel alone |
| `sse_26` | Likert 0–4 | Feel liked |
| `sse_27` | Likert 0–4 | Feel connected to others |
| `sse_28` | Likert 0–4 | Feel disconnected from others |
| `sse_29` | Likert 0–4 | Feel isolated from others |
| `sse_30` | Likert 0–4 | Feel overly sensitive |
| `sse_31` | Likert 0–4 | Feel lonely |
| `sse_32` | Likert 0–4 | Feel in tune with people |

### 6. Janis-Field Feelings of Inadequacy Scale (`jfis01`) — N=214

Trait self-esteem / feelings of inadequacy.

| Variable | Type | Description |
|---|---|---|
| `janis_1` to `janis_36` | Likert 1–5 | 36 items covering inferiority, social anxiety, academic confidence, physical self-image |
| `janis_total` | Integer | **Total Self-Esteem score** |
| `janis_score2` | Integer | **Social Confidence subscale** |
| `janis_score3` | Integer | **Self-Regard subscale** |

### 7. Medical History (`medhxrat01`) — N=111

Not relevant to our analysis. Standard medical/psychiatric history.

### 8. Imaging (`image03`) — N=296

Brain imaging (fMRI, structural). Not relevant to our behavioral analysis.

---

## Comparison with StudentLife (Current Dataset)

| Measure | StudentLife (N=28) | CES (N=175–215) | Compatible |
|---|---|---|---|
| **GPA** | Continuous (registrar) | Continuous (`academ_gpa`) + categorical | Yes |
| **PHQ-9** | Pre/post survey | Multi-timepoint (daily/weekly/monthly) | Yes |
| **PSS** | Pre/post survey | Multi-timepoint (daily/weekly/monthly/quarterly) | Yes |
| **GAD-7** | Not collected | Full GAD-7 + daily/lifetime | New variable |
| **Big Five (BFI)** | BFI-44 | **Not collected** | **No** |
| **UCLA Loneliness** | 20-item scale | Not collected (partial proxy via sse_25,29,31) | Partial |
| **Flourishing Scale** | 8-item scale | Not collected | No |
| **PANAS** | 20-item scale | Not collected | No |
| **Self-Esteem** | Not collected | SSE (20 items) + Janis-Field (36 items) | New variable |
| **Academic Engagement** | Not collected | 5-item scale (`academ_1–5`) | New variable |
| **Sensor Data** | StudentLife App (Android) | StudentLife App (Android + iOS) | Yes |
| **Duration** | 10 weeks | 4 years (per cohort) | CES much longer |
| **Temporal Resolution** | Semester aggregate | Daily/weekly surveys + continuous sensing | CES much richer |

---

## What CES Can Validate from Our Study

### Directly Validable (high value)

| Our Finding | CES Variables | Validation Approach |
|---|---|---|
| Behavior → PHQ-9 (R²=0.468) | Sensor data + `phq_9_total` | Replicate with N=214 |
| Behavior → PSS | Sensor data + `pss_totalscore` | Replicate with N=214 |
| SVR non-linear Behavior → GPA | Sensor data + `academ_gpa` | Test linear vs non-linear with N=175 |
| LPA behavioral profiles → stress/loneliness | Sensor data + PSS + `sse_31` | Replicate clustering with N=200+ |
| ML framework (SHAP, Optuna, cross-model) | All variables | Full replication |

### Partially Validable

| Our Finding | Limitation |
|---|---|
| Behavior → Wellbeing (PLS-SEM) | No Flourishing/PANAS, but can use GAD-7 + Self-Esteem instead |
| LPA → loneliness | No UCLA scale, but SSE has loneliness items (sse_25, sse_29, sse_31) |

### Not Validable (missing measures)

| Our Finding | Missing in CES |
|---|---|
| **Personality → GPA** (core finding) | No Big Five |
| **Personality moderates behavior** (E×Activity→GPA) | No Big Five |
| **SHAP: Conscientiousness = #1 predictor** | No Big Five |
| Behavior → Flourishing / PANAS-NA | No Flourishing, no PANAS |

---

## Unique Opportunities with CES (Beyond Validation)

1. **Longitudinal analysis**: 4-year tracking enables within-person behavioral change studies (our StudentLife is cross-sectional aggregate)
2. **COVID-19 natural experiment**: Behavior shifts pre/during/post pandemic
3. **GAD-7 as new outcome**: Anxiety prediction via behavior (not available in StudentLife)
4. **Self-Esteem pathways**: SSE Performance subscale may relate to GPA differently than personality
5. **Academic engagement as mediator**: `academ_total` could mediate Behavior → GPA
6. **Larger sample for non-linear detection**: N=175+ gives SVR much more power to detect non-linear relationships
7. **Daily PSS/PHQ**: Within-person temporal dynamics of stress and depression

---

## Access Requirements

- **NDA Data Use Certification (DUC)** required
- Institutional agreement needed
- Application: https://nda.nih.gov/nda/data-use-agreements
- Typical processing time: 2–4 weeks after submission

---

## Bottom Line

CES is a **partial validation dataset**. It significantly strengthens the Behavior → Mental Health findings (N=28 → N=200+) and enables non-linear GPA analysis with adequate power. However, it cannot validate the Personality → GPA core narrative due to the absence of Big Five measures. A paper combining both datasets would position as "discovery (StudentLife) + partial validation (CES)" — stronger than StudentLife alone, but not a complete replication.
