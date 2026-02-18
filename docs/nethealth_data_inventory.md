# NetHealth (University of Notre Dame) — Data Inventory

> Institution: University of Notre Dame, Center for Network Science and Data
> Funding: National Heart, Lung and Blood Institute (NIH)
> Period: Fall 2015 – Spring 2019 (4 years)
> N: 722 students (698 enrolled; 300 continued full 4 years)
> Website: https://sites.nd.edu/nethealth/
> Data: https://sites.nd.edu/nethealth/data-2/
> Access: **Free download** — fill a short registration form

---

## Overview

NetHealth is a multi-year longitudinal study of social networks and health among University of Notre Dame undergraduates. The study combines Fitbit wearable sensing, communication logs, social network surveys, academic records, and comprehensive psychological assessments across 8 survey waves over 4 years.

**This is the strongest candidate dataset for validating our StudentLife findings.** It is the only publicly available dataset we have found that simultaneously includes Big Five personality, GPA, mental health measures, and behavioral sensing data in a college student sample.

---

## Available Data Files

| # | Dataset | Format | Contents |
|---|---|---|---|
| 1 | **Basic Survey Data** | .dta, .csv, codebook | 3,000+ variables across 8 waves |
| 2 | **Grades Data** | .dta, .csv, codebook | Course enrollment + grades (registrar) |
| 3 | **Fitbit Activity Data** | .dta, .csv, codebook | Daily steps, active minutes, heart rate |
| 4 | **Fitbit Sleep Data** | .dta, .csv, codebook | Sleep duration, interruptions, quality |
| 5 | **Communication Events** | .dta, .csv, codebook | Calls, texts, messages (timestamped) |
| 6 | **Network Survey Data** | .dta, .csv, codebook | Up to 20 social contacts per wave |
| 7 | **Network Edge List** | .dta, .csv, codebook | Social network structure |
| 8 | **Calendar** | .csv | Academic calendar/scheduling |

---

## Detailed Variable Inventory

### Personality

| Measure | Items | Scales | Notes |
|---|---|---|---|
| **Big Five Inventory (BFI)** | 44 items | Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness | **Same instrument as StudentLife (BFI-44)** — directly comparable |

### Mental Health

| Measure | Acronym | What It Measures | Notes |
|---|---|---|---|
| Beck Depression Inventory | **BDI** | Depression severity | Different from PHQ-9 but same construct |
| CES-D | **CES-D** | Depression (epidemiological) | 20-item, widely validated |
| State-Trait Anxiety Inventory | **STAI** | Anxiety (state + trait) | Gold standard anxiety measure |
| Beck Anxiety Inventory | **BAI** | Anxiety severity | Somatic focus |
| Stress measures | — | Perceived stress | Specific instrument TBD from codebook |
| SELSA | **SELSA** | Social & Emotional Loneliness | Replaces UCLA Loneliness conceptually |

### Academic Performance

| Measure | Source | Details |
|---|---|---|
| **Semester GPA** | Registrar records | Objective, per-semester |
| Course enrollment | Registrar records | Courses taken each semester |
| Major | Registrar records | Anonymized coding |

### Wearable Sensing (Fitbit)

| Measure | Frequency | Details |
|---|---|---|
| Daily steps | Daily | Total step count |
| Active minutes | Daily | Light, moderate, vigorous activity |
| Heart rate | Continuous | Resting + active heart rate |
| Sleep duration | Nightly | Total sleep time |
| Sleep interruptions | Nightly | Wake episodes |
| Sleep quality | Nightly | Efficiency metrics |

### Communication & Social

| Measure | Details |
|---|---|
| Call logs | Timestamped, duration, direction |
| Text messages | Timestamped, direction |
| Social network surveys | Up to 20 contacts, 8 waves |
| Network edge list | Full social graph |

### Other Psychological Measures

| Measure | Details |
|---|---|
| Self-Esteem | Scale TBD from codebook |
| Trust | Interpersonal trust |
| Self-Regulation | Two versions |
| Self-Efficacy | Two versions |
| Body Image Inventory | Body satisfaction |
| PSQI | Pittsburgh Sleep Quality Index |
| MEQ | Morningness-Eveningness (chronotype) |

### Demographics & Background

- Family background, prior schooling
- Activities, clubs, cultural activities
- Physical activities, music preferences
- Political attitudes, technology use
- Health-related behaviors

---

## Comparison with StudentLife (Our Dataset)

| Measure | StudentLife (N=28) | NetHealth (N=722) | Compatible |
|---|---|---|---|
| **Big Five** | BFI-44 | **BFI-44 (same!)** | ✅ Directly comparable |
| **GPA** | Registrar | **Registrar** | ✅ Directly comparable |
| **Depression** | PHQ-9 | BDI + CES-D | ⚠️ Different instruments, same construct |
| **Anxiety** | Not collected | STAI + BAI | New variable |
| **Stress** | PSS-10 | Stress measures | ⚠️ Need to check instrument |
| **Loneliness** | UCLA-20 | SELSA | ⚠️ Different instruments, same construct |
| **Flourishing** | Flourishing Scale | Not collected | Cannot validate |
| **PANAS** | PANAS-20 | Not collected | Cannot validate |
| **Phone sensors** | 13 modalities | Communication logs only | ⚠️ Limited overlap |
| **Wearable** | Not collected | Fitbit (activity/sleep/HR) | New modality |
| **Social network** | Bluetooth proximity | Full social network graph | Different but richer |
| **Duration** | 10 weeks | **4 years** | NetHealth much longer |
| **Survey waves** | Pre/post | **8 waves** | NetHealth much richer |

---

## What NetHealth Can Validate from Our Study

### Fully Validable (high value)

| Our Finding | NetHealth Variables | Validation Approach |
|---|---|---|
| **Personality → GPA** (core finding) | BFI-44 + Registrar GPA | **Direct replication with N=722** |
| **Conscientiousness = #1 GPA predictor** | Same BFI-44 | SHAP analysis on much larger sample |
| **Personality moderates behavior effects** | BFI + Fitbit activity + GPA | Test E×Activity→GPA with adequate power |
| **Cross-model consistency** (Kendall's τ) | All variables | 4 models × multiple outcomes |
| **ML framework** (SHAP, Optuna) | All variables | Full methodological replication |

### Partially Validable

| Our Finding | Limitation | Workaround |
|---|---|---|
| Behavior → Depression | PHQ-9 vs BDI/CES-D | Different scales but same construct; report both |
| Behavior → Loneliness | UCLA vs SELSA | Same construct, different operationalization |
| LPA behavioral profiles | Fitbit ≠ phone sensors | Different behavioral features, same clustering approach |
| SVR non-linear effects | Fitbit ≠ phone sensors | Test with Fitbit activity features |

### Not Validable

| Our Finding | Reason |
|---|---|
| PANAS-NA prediction | No PANAS in NetHealth |
| Flourishing prediction | No Flourishing Scale in NetHealth |
| PLS-SEM structural model | Different sensor modalities; partial replication only |

---

## Key Advantages of NetHealth

1. **Same personality instrument (BFI-44)** — no cross-instrument comparability issues
2. **Registrar GPA** — objective, not self-reported
3. **N=722** — 25× our sample; adequate power for all analyses
4. **4-year longitudinal** — can test temporal stability and within-person change
5. **Free and public** — no lengthy data access application
6. **Social network data** — unique addition not available in other datasets
7. **8 survey waves** — can track personality-outcome relationships over time

## Limitations

1. **Fitbit ≠ smartphone sensors** — different behavioral features than our 13 modalities
2. **No smartphone passive sensing** — no GPS, app usage, screen time, audio, WiFi
3. **Different depression measure** — BDI/CES-D vs PHQ-9 (conceptually same, scores not directly comparable)
4. **Single institution** — both datasets are from elite US universities (Dartmouth, Notre Dame)
5. **Different cohort years** — 2013 vs 2015-2019

---

## Access Instructions

1. Go to https://sites.nd.edu/nethealth/data-2/
2. Click download links (Google Drive) for each dataset
3. Fill out the short registration form
4. Email nethealth@nd.edu when publishing results
5. Data available in Stata (.dta) and CSV formats with codebooks

---

## Impact on Publication Strategy

| Scenario | Q1 Probability |
|---|---|
| StudentLife alone (N=28) | 10-20% |
| StudentLife + CES partial validation | 25-40% |
| **StudentLife + NetHealth full validation** | **50-60%** |
| StudentLife + NetHealth + CES | 60-70% |

NetHealth enables a **"discovery + validation"** paper structure:
- **Study 1** (StudentLife, N=28): Full 8-method exploratory analysis
- **Study 2** (NetHealth, N=722): Validation of Personality→GPA, moderation, SHAP, ML framework

This is the strongest validation dataset currently available for our research question.
