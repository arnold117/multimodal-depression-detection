# Copenhagen Networks Study (CNS / SensibleDTU) — Data Inventory

> Institution: Technical University of Denmark (DTU) + University of Copenhagen
> PI: Sune Lehmann
> Period: 2012–2016 (two deployments)
> N: ~250 (2012) + ~1,000 (2013), total ~1,250
> Paper: Stopczynski et al. "Measuring Large-Scale Social Networks with High Resolution" (PLOS ONE, 2014)
> Data paper: Sapiezynski et al. "Interaction data from the Copenhagen Networks Study" (Scientific Data, 2019)
> Public data: https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433
> Access: **Interaction data only** is public; **questionnaire + grades data is RESTRICTED**

---

## Overview

The Copenhagen Networks Study is one of the largest and most comprehensive social sensing experiments ever conducted. It tracked ~1,000 DTU undergraduates over multiple years using custom Android smartphones, collecting Bluetooth proximity, communication logs, GPS, WiFi, and an extensive battery of psychological questionnaires including Big Five personality, depression (MDI), stress (PSS), loneliness (UCLA), PANAS, and institutional GPA.

**Why this matters for us**: CNS has **ALL** the variables we need — BFI + institutional GPA + depression + stress + loneliness + PANAS + rich sensing — in N=1,000 students. It is arguably the **most complete** dataset in the world for our research question. The catch: questionnaire and grades data are **not publicly released**.

---

## Deployments

| Deployment | Year | N | Duration | Questionnaire |
|---|---|---|---|---|
| **Pilot (2012)** | Oct 2012 – Sep 2013 | ~250 | ~12 months | 95 questions |
| **Main (2013)** | Aug 2013 – Feb 2016 | ~1,000 | ~2.5 years | **310 questions** |

---

## Available Data — What CNS Collected

### Personality

| Measure | Items | Notes |
|---|---|---|
| **Big Five Inventory (BFI)** | Standard BFI | Collected in both deployments; included in the 310-question battery |

### Mental Health (310-question battery, 2013 deployment)

| Measure | Acronym | What It Measures |
|---|---|---|
| **Major Depression Inventory** | **MDI** | Depression symptoms (ICD-10/DSM-IV aligned) |
| **Cohen's Perceived Stress Scale** | **PSS** | Perceived stress |
| **UCLA Loneliness Scale** | **UCLA** | Loneliness |
| **PANAS** | **PANAS** | Positive and Negative Affect |
| **Rosenberg Self-Esteem Scale** | **RSES** | Self-esteem |
| **Satisfaction With Life Scale** | **SWLS** | Life satisfaction |
| **Rotter's Locus of Control** | — | Internal vs external locus |
| **Narcissism (NAR-Q)** | **NAR-Q** | Narcissistic tendencies |
| **Self-Efficacy Scale** | — | Self-efficacy |
| **Copenhagen Social Relations** | — | Social relationships quality |

Questionnaires were **repeated each semester**.

### Academic Performance

| Measure | Source | Details |
|---|---|---|
| **Course grades** | DTU Registrar (institutional) | Denmark 7-point scale (-3 to 12, mapping F to A) |
| **Mean term grade** | Computed from registrar | Cumulative semester GPA equivalent |

Published in: Kassarnig et al. "Class attendance, peer similarity, and academic performance" (PLOS ONE, 2017)

### Behavioral Sensing (Custom Android Smartphones)

| Sensor | Frequency | Details |
|---|---|---|
| **Bluetooth proximity** | Every 5 minutes | Detects nearby CNS participants within 5-10m |
| **GPS/Location** | Periodic | Position estimates |
| **WiFi scans** | Regular intervals | 42.6M observations (2012 deployment) |
| **Phone calls** | All calls | Caller, callee, duration, timestamp |
| **SMS/Text** | All messages | Sender, recipient, timestamp |
| **Facebook** | Friendship graph + interactions | Social network structure |
| **App usage** | Continuous | Application usage tracking |
| **Screen status** | Continuous | Screen on/off patterns |
| **Battery** | Continuous | Charging patterns |
| **Sleep (inferred)** | Nightly | SensibleSleep model from phone activity |

Note: **No accelerometer** data was collected by the app.

---

## Comparison with StudentLife (Our Dataset)

| Measure | StudentLife (N=28) | CNS (N=~1,000) | Compatible |
|---|---|---|---|
| **Big Five** | BFI-44 | BFI | ✅ Likely same/similar instrument |
| **GPA** | Registrar (Dartmouth) | Registrar (DTU) | ✅ Both institutional |
| **Depression** | PHQ-9 | MDI | ⚠️ Different instruments, same construct |
| **Stress** | PSS-10 | PSS | ✅ Same instrument |
| **Loneliness** | UCLA-20 | UCLA | ✅ Same instrument |
| **Affect** | PANAS-20 | PANAS | ✅ Same instrument |
| **Flourishing** | Flourishing Scale | Not collected | No |
| **Phone sensors** | 13 modalities | BT/GPS/WiFi/calls/SMS/apps/screen | ⚠️ Partial overlap |
| **Duration** | 10 weeks | 2.5 years | CNS much longer |
| **Population** | US (Dartmouth) | Denmark (DTU) | ⚠️ Different culture/system |

---

## What CNS Can Validate (If Data Obtained)

### Directly Validable

| Our Finding | CNS Variables | Validation Approach |
|---|---|---|
| **Personality → GPA** (core!) | BFI + Registrar grades | Direct replication, N=~1,000 |
| **Conscientiousness = #1 GPA predictor** | BFI + grades | SHAP analysis on massive sample |
| **Behavior → Depression** | Sensors + MDI | Replicate with N=~1,000 |
| **Behavior → Stress** | Sensors + PSS | Same instrument, direct replication |
| **Behavior → Loneliness** | Sensors + UCLA | Same instrument, direct replication |
| **Behavior → Affect** | Sensors + PANAS | Same instrument, direct replication |
| **Personality moderates behavior** | BFI + sensors + grades | Test moderation with massive power |
| **LPA behavioral profiles** | Sensor features | Replicate clustering, N=~1,000 |
| **ML framework** | All variables | Full methodological replication |

### Not Validable

| Our Finding | Reason |
|---|---|
| Behavior → Flourishing | No Flourishing Scale |
| Exact PHQ-9 comparison | MDI ≠ PHQ-9 (different depression instrument) |

---

## Data Availability — The Critical Issue

### What IS publicly available

The **interaction data only** (4 weeks) is on Figshare:
- URL: https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433
- DOI: 10.6084/m9.figshare.7267433

| File | Records | Users |
|---|---|---|
| `bt.csv` | 5,474,289 | 706 |
| `calls.csv` | 3,600 | 540 |
| `sms.csv` | 24,333 | 577 |
| `fb_friends.csv` | 6,429 edges | 800 |
| `genders.csv` | 788 | 788 |

Also on Netzschleuder: https://networks.skewed.de/net/copenhagen

### What is NOT publicly available

- **All questionnaire data** (Big Five, MDI, PSS, UCLA, PANAS, etc.)
- **GPS/location traces**
- **Academic grades**
- **Full longitudinal data** (only 4 weeks released)
- **Absolute timestamps** (only relative)

### Why restricted

The researchers explicitly state privacy concerns: "In a complex dataset, such as ours, it is virtually impossible to provide guarantees regarding re-identification of users while preserving its value for research purposes." Location data withheld because it could be cross-correlated with Denmark's public address registries.

---

## How to Obtain Full Data — Detailed Investigation

### Evidence: Has the data been shared before?

**YES — with collaborators, under controlled conditions.**

| Researcher | Affiliation | Paper | Data Used |
|---|---|---|---|
| **Valentin Kassarnig** | Graz U. of Technology (Austria) | PLOS ONE 2017, EPJ Data Sci 2018 | **Personality (conscientiousness, self-esteem) + grades** |
| **Enys Mones** | External collaborator | EPJ Data Sci 2018 | Same as above |
| **Andreas Bjerre-Nielsen** | UCPH Economics/SODAS | PNAS 2021 | Full behavioral + questionnaire + Statistics Denmark registry |
| **Cal Poly student** | Cal Poly SLO (US) | Master's thesis | Questionnaire data ("socioeconomic, psychological, well-being") |

**Pattern**: Access is granted through **formal research collaboration** (co-authorship), not through an open application. The Austrian (Kassarnig) and American (Cal Poly) examples confirm that non-Danish researchers CAN get access.

### Primary contact

| Person | Role | Email | Affiliation |
|---|---|---|---|
| **Sune Lehmann** | PI, Professor | `sljo@dtu.dk` / `sune@sodas.ku.dk` | DTU Compute + UCPH SODAS |
| **David Dreyer Lassen** | Co-PI, Economics | `david.dreyer.lassen@econ.ku.dk` | UCPH Economics |
| **Andreas Bjerre-Nielsen** | Assoc. Professor | bjerre-nielsen.me | UCPH SODAS |
| **DTU Data Protection Officer** | GDPR compliance | `dpo@dtu.dk` | DTU |

### What to expect (based on precedent)

The evidence suggests you will need to:

1. **Propose a specific co-authored research project** — not just "give me the data"
2. **Sign a confidentiality/data sharing agreement** with DTU
3. **Possibly work on Danish infrastructure** — one source states researchers must "agree to work under supervision in Copenhagen"
4. **Obtain your own institution's ethics approval** for secondary data analysis
5. **Execute GDPR-compliant Standard Contractual Clauses (SCCs)** if data leaves the EU

### GDPR specifics for non-EU researcher

- Data was collected 2013-2015 (pre-GDPR), but team states consent was "implemented in a manner similar to GDPR rules"
- GDPR applies retroactively to continued processing
- **For transfers to non-EU**: Denmark allows transfers using EU-approved SCCs without additional Danish DPA approval
- **US-specific**: EU-US Data Privacy Framework may apply if receiving institution is certified
- **Anonymization workaround**: If data is sufficiently anonymized, GDPR does not apply. Could request only aggregated summary statistics (correlation matrix, means/SDs) instead of raw data
- **Study registered with Datatilsynet** (Danish Data Supervision Authority) — any new use may require additional ethics review

### Realistic access timeline

| Phase | Time | Action |
|---|---|---|
| 1 | Week 1 | Email Sune Lehmann with research proposal |
| 2 | Weeks 2-4 | Negotiate collaboration scope, co-authorship terms |
| 3 | Weeks 4-8 | Institutional agreements, GDPR paperwork, DTU DPO approval |
| 4 | Weeks 8-12 | Data access granted (possibly on Danish infrastructure) |
| **Total** | **2-3 months** | Optimistic; could be longer |

### Alternative: Request summary statistics only

The published CNS papers contain some useful but incomplete statistics:

- **Conscientiousness→GPA**: r ≈ 0.2-0.3 (Kassarnig et al. EPJ Data Sci 2018)
- **Self-esteem→GPA**: r ≈ 0.2-0.3 (same paper)
- **Neuroticism→GPA**: negative but weaker (same paper)
- **Attendance→GPA**: Spearman r = 0.255, p < 0.001 (Kassarnig et al. PLOS ONE 2017)

**Missing**: Full BFI correlation matrix, MDI/PSS/UCLA/PANAS correlations with GPA, behavioral feature importance rankings.

**Fallback option**: Request just the correlation matrix or summary statistics (not raw data) — this has far fewer GDPR implications and could be shared via email. This alone could support a meta-analytic comparison with our StudentLife findings.

### Indirect connection: Alex Pentland (MIT)

Alex "Sandy" Pentland (MIT Media Lab / Connection Science) is a co-author on the foundational CNS paper (Stopczynski et al., 2014). He also has connections to the SNAPSHOT study ecosystem at MIT. If you have any MIT connections, Pentland could potentially facilitate a warm introduction to Lehmann's group.

---

## Key Publications Using CNS Data

1. Stopczynski et al. (2014). "Measuring Large-Scale Social Networks with High Resolution." PLOS ONE. [DOI:10.1371/journal.pone.0095978](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0095978)
2. Sapiezynski et al. (2019). "Interaction data from the Copenhagen Networks Study." Scientific Data. [PMC6906316](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906316/)
3. Kassarnig et al. (2017). "Class attendance, peer similarity, and academic performance in a large field study." PLOS ONE. [PMC5678706](https://pmc.ncbi.nlm.nih.gov/articles/PMC5678706/)
4. Kassarnig et al. (2018). "Academic performance and behavioral patterns." EPJ Data Science. [DOI:10.1140/epjds/s13688-018-0138-8](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-018-0138-8)
5. Andersen et al. (2022). "Predicting stress and depressive symptoms using high-resolution smartphone data and sleep behavior." SLEEP. [PMID:35298650](https://pubmed.ncbi.nlm.nih.gov/35298650/)
6. Bjerre-Nielsen et al. (2021). "Task-specific information outperforms surveillance-style big data in predictive analytics." PNAS. [DOI:10.1073/pnas.2020258118](https://www.pnas.org/doi/10.1073/pnas.2020258118)

---

## Bottom Line

CNS is the **holy grail dataset** for our research question — it has everything we need (BFI + institutional GPA + depression + stress + loneliness + PANAS + rich sensing) in N=~1,000 students.

**Good news**: External researchers (including non-Danish) have successfully accessed the full data before, through collaboration.

**Realistic assessment**: Getting access requires proposing a co-authored project to Sune Lehmann, navigating GDPR, and ~2-3 months of admin. This is a **high-reward, medium-effort** path with precedent for success.

**Recommended approach**:
1. **Immediate**: Proceed with NetHealth (free) and GLOBEM (PhysioNet DUA) — don't wait
2. **This week**: Email Sune Lehmann (`sljo@dtu.dk`) with a brief, specific research proposal
3. **Fallback**: Request only the correlation matrix / summary statistics (minimal GDPR friction)
4. **Long-term**: If collaboration established, CNS becomes the crown jewel of a multi-dataset validation paper
