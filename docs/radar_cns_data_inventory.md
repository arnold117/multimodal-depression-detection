# RADAR-CNS (RADAR-MDD) — Data Inventory

> Remote Assessment of Disease and Relapse – Central Nervous System
> Institution: King's College London, VU Amsterdam, CIBER Barcelona
> Period: 2016–2022 | N≈600 (MDD patients)
> Protocol: https://pmc.ncbi.nlm.nih.gov/articles/PMC6379954/
> Data Access: Not open — request via King's College London

## Overview

RADAR-MDD is a multi-centre, prospective observational cohort study of adults with Major Depressive Disorder. Participants were tracked for 11–24 months using smartphone sensors, wearable devices (Fitbit), and self-report questionnaires. It is the largest multimodal remote measurement study in mental health.

**Important**: This is a **clinical psychiatric sample** (adults with MDD diagnosis), NOT a student/general population sample.

---

## Self-Report Questionnaires

| # | Instrument | Short Name | Items | Frequency | Description |
|---|---|---|---|---|---|
| 1 | Inventory of Depressive Symptomatology | IDS-SR | 30 | Biweekly | **Primary outcome** — depressive symptoms |
| 2 | CIDI Short Form | CIDI-SF | — | Biweekly | Diagnostic criteria for MDD relapse/remission |
| 3 | Patient Health Questionnaire | PHQ-8 | 8 | Biweekly | Depression severity (no item 9 suicidality) |
| 4 | Generalized Anxiety Disorder | GAD-7 | 7 | Periodic | Anxiety symptoms |
| 5 | Rosenberg Self-Esteem Scale | RSES | 10 | Biweekly | Self-esteem |
| 6 | Work and Social Adjustment Scale | WSAS | 5 | Periodic | Functional impairment |
| 7 | Brief Illness Perceptions | BIPQ | — | Periodic | Illness perception |
| 8 | AUDIT | AUDIT | 10 | Periodic | Alcohol use |
| 9 | List of Threatening Experiences | LTE-Q | — | Periodic | Life events |
| 10 | Client Service Receipt Inventory | CSRI | — | Periodic | Health economics |
| 11 | THINC-IT | — | — | Periodic | Cognitive screening (memory, concentration) |
| 12 | Experience Sampling (ESM) | — | 44 | Multiple/day | Mood, stress, sociability, activity, sleep |

## Passive Smartphone Sensors

| Sensor | Data Collected |
|---|---|
| GPS | Relative location (obfuscated for privacy) |
| Accelerometer | Movement patterns |
| Gyroscope | Orientation changes |
| Call logs | Duration, frequency (no content) |
| SMS logs | Frequency (no content) |
| Email logs | Frequency |
| Ambient noise | Microphone-based noise levels |
| Ambient light | Light sensor |
| Screen interactions | Unlock patterns, screen time |
| Bluetooth | Nearby devices (social proximity) |
| Battery | Charging patterns |
| Weather | External API linked to location |
| Keystroke dynamics | Typing patterns (London site only) |

## Wearable Data (Fitbit Charge 2)

| Measure | Description |
|---|---|
| Heart rate | Continuous monitoring |
| Steps | Daily step count |
| Sleep efficiency | % time asleep while in bed |
| Sleep latency | Time to fall asleep |
| Sleep fragmentation | Wake episodes during sleep |
| Sedentary time | Inactive periods |
| Physical activity | Exercise classification |

## Active Tasks

| Task | Description |
|---|---|
| Speech task 1 | Read "The North Wind and the Sun" excerpt |
| Speech task 2 | Open-ended: "describe something you look forward to" |
| Cognitive tests | THINC-IT app-based assessment |

---

## What RADAR-MDD Does NOT Have

| Variable | Status | Impact |
|---|---|---|
| **Big Five Personality** | ❌ Not collected | Cannot validate Personality→outcome findings |
| **GPA / Academic Performance** | ❌ Not applicable | Adult psychiatric patients, not students |
| **UCLA Loneliness Scale** | ❌ Not collected | — |
| **Flourishing Scale** | ❌ Not collected | — |
| **PANAS** | ❌ Not collected | — |
| **PSS** | ❌ Not collected (ESM stress items only) | Not directly comparable to PSS-10 |

---

## Comparison with Our Study Needs

| Our Finding | RADAR-MDD Can Validate? | Notes |
|---|---|---|
| Personality → GPA | ❌ No | No personality, no GPA |
| Personality moderates behavior | ❌ No | No personality |
| Behavior → PHQ depression | ⚠️ Partially | Has PHQ-8, but clinical MDD sample ≠ healthy students |
| Behavior → PSS stress | ❌ No | No PSS (only ESM stress items) |
| SVR non-linear Behavior → GPA | ❌ No | No GPA |
| LPA behavioral profiles | ⚠️ Partially | Different population, different baseline |
| ML framework (SHAP, Optuna) | ✅ Methodologically | Can apply same methods to different outcomes |

---

## Data Access Process

1. **Not openly available** — requires formal data sharing agreement
2. Contact: RADAR-CNS consortium via King's College London
3. Website: https://www.kcl.ac.uk/research/radarcns
4. GitHub (data dictionaries only): https://github.com/RADAR-base/RADAR-REDCap-Data-Dictionary
5. Expect significant processing time due to EU data protection (GDPR)

---

## When RADAR-MDD Would Be Useful

Despite being a poor fit for our current paper, RADAR-MDD could be valuable if:

1. **Pivoting to a clinical depression focus** — if the paper shifts from "GPA prediction" to "depression monitoring via passive sensing", RADAR-MDD's N≈600 clinical sample becomes highly relevant
2. **Methodological paper** — demonstrating that the same ML pipeline (SHAP + Optuna + multi-model) works across different populations and datasets
3. **Future work / second paper** — "From healthy students to clinical populations: generalisability of smartphone-based behavioral markers for depression"
4. **Wearable data analysis** — Fitbit heart rate + sleep data not available in StudentLife or CES
5. **ESM temporal dynamics** — 44-item ESM multiple times daily, much richer than weekly surveys

---

## Bottom Line

**Not recommended for the current paper.** The population mismatch (adult MDD patients vs. healthy college students) means any "validation" would be questioned by reviewers. No personality and no GPA makes it fundamentally incompatible with our core narrative.

**Keep on radar (pun intended) for future work** — especially if pivoting toward clinical depression prediction or writing a methods-focused follow-up paper.

### Priority Ranking for External Validation

| Dataset | Priority | Reason |
|---|---|---|
| **CES (NDA #2494)** | ⭐⭐⭐⭐ | Same school, same app, students, GPA + PHQ + PSS |
| **RADAR-MDD** | ⭐ | Wrong population, no personality, no GPA |
| **Original StudentLife 2016** | ⭐⭐ | N=83, has Big Five?, needs investigation |
| **SABLE (Stanford/UTA/Dartmouth)** | ⭐⭐⭐ | NSF personality+sensing project, needs investigation |
