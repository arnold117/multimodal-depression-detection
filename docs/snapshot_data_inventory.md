# SNAPSHOT Study — MIT Media Lab — Data Inventory

> Institution: MIT Media Lab, Affective Computing Group
> PIs: Akane Sano, Rosalind Picard (+ Harvard Medical School / Brigham & Women's Hospital)
> Funding: NIH
> Period: Fall 2013 onwards (~7 cohorts, ~30 days per cohort)
> N: ~321 total across 7 cohorts (~50 undergrads per semester)
> Name: **S**leep, **N**etworks, **A**ffect, **P**erformance, **S**tress, and **H**ealth using **O**bjective **T**echniques
> Website: https://www.media.mit.edu/projects/snapshot-study/overview/
> Access: **NOT publicly available** — requires data sharing agreement with MIT

---

## Overview

The SNAPSHOT study is an NIH-funded longitudinal research project that tracks MIT undergraduates over ~30-day periods each semester. It combines wrist-worn physiological sensors, smartphone passive sensing, daily surveys, and comprehensive psychological assessments. The study has run for 7+ cohorts since Fall 2013, collecting over 100,000 hours of sensor data.

**Value for our project**: SNAPSHOT has the **perfect variable combination** — BFI personality + GPA + PSS + STAI + rich multimodal sensing. If data access could be obtained, it would be an excellent validation dataset. However, data is NOT publicly available.

---

## Available Data

### Wearable Sensors (~172 features)

| Sensor | Device | Details |
|---|---|---|
| **Electrodermal Activity (EDA)** | Affectiva Q-sensor (wrist) | Skin conductance, SCR amplitude/shape/rate |
| **Skin Temperature** | Affectiva Q-sensor | Continuous skin temperature |
| **3-axis Accelerometer** | Affectiva Q-sensor | Sampled at 8 Hz; activity level, step count, stillness |
| **Activity + Light** | MotionLogger actigraphy | Activity patterns and ambient light exposure |

### Smartphone Data (~75 features)

| Category | Features | Details |
|---|---|---|
| **Calls** | ~20 features | Timing, duration, unique contacts, entropy, across 8 time windows |
| **SMS** | ~30 features | Frequency, contact patterns, timing |
| **Screen** | ~25 features | On/off events, duration, across 8 time windows |
| **GPS/Location** | ~15 features | Distance traveled, campus time, outdoor time, location routineness (GMM) |

### Environmental Data (~40 features)

| Source | Details |
|---|---|
| **Weather** | Sunlight, temperature, wind, barometric pressure |
| **Seasonal** | Day length, seasonal differences |

### Survey Instruments

| Measure | Acronym | When | Description |
|---|---|---|---|
| **Big Five Inventory** | **BFI** | Pre-study | Full Big Five personality (standard version) |
| **Myers-Briggs Type Indicator** | **MBTI** | Pre-study | Personality type classification |
| **Morningness-Eveningness** | **MEQ** | Pre-study | Chronotype (Horne-Ostberg) |
| **GPA** | — | Post-study | Self-reported semester GPA (2.0-5.0 scale) |
| **Perceived Stress Scale** | **PSS** | Pre/post | Perceived stress |
| **State-Trait Anxiety Inventory** | **STAI** | Post-study | State and trait anxiety scores |
| **SF-12 Mental Health** | **SF-12 MCS** | Post-study | General mental health composite (0-100) |
| **Pittsburgh Sleep Quality** | **PSQI** | Pre/post | Subjective sleep quality |
| **Daily mood** | VAS | Every evening (after 8pm) | 0-100 visual analog scales: mood (sad/happy), stress (stressed/calm), health (sick/healthy) |

### Lab Measures

| Measure | Details |
|---|---|
| **Dim Light Melatonin Onset (DLMO)** | Circadian phase measurement via saliva samples |

---

## Comparison with StudentLife (Our Dataset)

| Measure | StudentLife (N=28) | SNAPSHOT (N=~321) | Compatible |
|---|---|---|---|
| **Big Five** | BFI-44 | **BFI (standard)** | ✅ Likely same instrument |
| **GPA** | Registrar | Self-reported | ⚠️ Self-report vs registrar |
| **Depression** | PHQ-9 | **Not collected** (SF-12 MCS only) | ❌ No depression-specific measure |
| **Stress** | PSS-10 | PSS | ✅ Same instrument |
| **Anxiety** | Not collected | STAI | New variable |
| **Affect** | PANAS-20 | Daily VAS mood (0-100) | ⚠️ Different operationalization |
| **Loneliness** | UCLA-20 | Not collected | No |
| **Flourishing** | Flourishing Scale | Not collected | No |
| **Phone sensors** | StudentLife app (13 modalities) | Calls/SMS/Screen/GPS | ⚠️ Partial overlap |
| **Wearable** | Not collected | Q-sensor (EDA/temp/accel) + MotionLogger | New modality |
| **Duration** | 10 weeks | ~30 days | StudentLife longer |
| **Sleep** | PSQI + inferred | PSQI + MotionLogger + DLMO | SNAPSHOT richer |

---

## What SNAPSHOT Can Validate from Our Study

### Directly Validable (if data obtained)

| Our Finding | SNAPSHOT Variables | Validation Approach |
|---|---|---|
| **Personality → GPA** (core finding) | BFI + self-reported GPA | Direct replication with N=~321 |
| **Conscientiousness = #1 GPA predictor** | BFI + GPA | SHAP analysis on larger sample |
| **Behavior → Stress** | Phone/wrist sensors + PSS | Replicate with richer sensor data |
| **Personality moderates behavior** | BFI + sensors + GPA | Test moderation with adequate power |
| **ML framework** (SHAP, Optuna, cross-model) | All variables | Full methodological replication |

### Partially Validable

| Our Finding | Limitation |
|---|---|
| **Behavior → Depression** | No PHQ-9/BDI; only SF-12 MCS (general mental health) |
| **LPA behavioral profiles** | Different sensor features (wrist EDA vs phone-only) |
| **Behavior → Affect** | Daily VAS ≠ PANAS (different scale and construct) |

### Not Validable

| Our Finding | Reason |
|---|---|
| Behavior → Loneliness | No loneliness measure |
| Behavior → Flourishing | No flourishing measure |
| PANAS-NA prediction | No PANAS |

---

## Key Advantages of SNAPSHOT

1. **BFI + GPA + PSS + sensors** — the closest variable match to our study design
2. **N=~321** — 11× our sample, adequate power for most analyses
3. **7 cohorts** — can test cross-cohort replication (internal validation)
4. **Physiological sensing** — EDA, skin temperature, heart rate (not available in StudentLife)
5. **DLMO circadian measurement** — unique objective circadian phase data
6. **MIT undergraduates** — elite university population similar to Dartmouth
7. **Daily VAS surveys** — temporal dynamics of mood, stress, health

## Limitations

1. **Data NOT publicly available** — requires formal collaboration with MIT Media Lab
2. **No depression measure** — only SF-12 MCS general mental health (not PHQ-9/BDI)
3. **Self-reported GPA** — less reliable than registrar data
4. **~30 days only** — shorter than our 10-week study
5. **Different sensing modalities** — wrist EDA/temperature vs phone audio/WiFi/Bluetooth
6. **MIT-specific population** — highly selective, may not generalize

---

## Access Information

- **Status**: NOT publicly available
- **Website**: https://www.media.mit.edu/projects/snapshot-study/overview/
- **Contact**: snapshot@media.mit.edu
- **Code available**: https://github.com/mitmedialab/personalizedmultitasklearning (Personalized Multitask Learning code only)
- **Process**: Likely requires formal data sharing agreement with MIT IRB approval
- **Realistic assessment**: Obtaining data would require establishing a research collaboration with the Affective Computing group. This is a long-shot unless there is a pre-existing relationship.

---

## Key Publications

1. Sano et al. (2015). "Recognizing Academic Performance, Sleep Quality, Stress Level, and Mental Health using Personality Traits, Wearable Sensors and Mobile Phones." IEEE BSN 2015. [PMC5431072](https://pmc.ncbi.nlm.nih.gov/articles/PMC5431072/)
2. Taylor, Jaques et al. (2017/2020). "Personalized Multitask Learning for Predicting Tomorrow's Mood, Stress, and Health." IEEE TAFFC. [PMC7266106](https://pmc.ncbi.nlm.nih.gov/articles/PMC7266106/)
3. Sano (2016). PhD Thesis: "Measuring College Students' Sleep, Stress, Mental Health and Wellbeing with Wearable Sensors and Mobile Phones." MIT DSpace.

---

## Bottom Line

SNAPSHOT is a **tantalizing but inaccessible** dataset. It has nearly the perfect variable combination for validating our findings (BFI + GPA + stress + multimodal sensing in college students). However, the data is not public, and obtaining access would require a formal research collaboration with MIT Media Lab.

**Recommendation**: Worth sending a cold email to snapshot@media.mit.edu or Akane Sano (now at Rice University) describing your research and asking about data access. Success probability is low (~10-20%) but the payoff would be high.

**In the meantime, prioritize NetHealth (free, immediate) and GLOBEM (PhysioNet DUA).**
