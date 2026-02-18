# StudentLife 2nd Study (2016) & SABLE/PNAS Personality Study — Data Inventory

## 1. StudentLife 2nd Study (2016, N=83)

> Institution: Dartmouth College
> Period: Winter + Spring 2016 (two 9-week terms)
> Paper: Rui Wang et al. "Tracking Depression Dynamics in College Students Using Mobile Phone and Wearable Sensing" (IMWUT 2018)
> PubMed: https://pubmed.ncbi.nlm.nih.gov/39449996/
> PDF: https://studentlife.cs.dartmouth.edu/a43-Wang.pdf

### Overview

Follow-up to the original StudentLife study (2013, N=48). Tracked 83 Dartmouth undergraduates across two consecutive terms using smartphones and wearables. Focused on depression dynamics rather than personality or academic performance.

### Data Collected

| Category | Details |
|---|---|
| **Smartphone sensors** | Android + iOS (same StudentLife app) |
| **Wearable** | Microsoft Band 2 (heart rate, steps, sleep) |
| **PHQ-8** | Pre/post each term |
| **PHQ-4** | Weekly throughout each term |
| **Big Five** | ❌ Not collected |
| **GPA** | ❌ Not mentioned |
| **PSS** | ❌ Not mentioned |
| **Loneliness** | ❌ Not mentioned |
| **PANAS** | ❌ Not mentioned |

### Key Findings

- Depression prediction: 81.5% recall, 69.1% precision on week-by-week basis
- Symptom features derived from phone + wearable sensors mapped to DSM-5 criteria
- Behavioral patterns differ between depressed and non-depressed students

### Data Availability

**❌ NOT publicly available.** The StudentLife website (studentlife.cs.dartmouth.edu/dataset.html) only hosts the original 2013 dataset (N=48). The 2016 dataset has never been released publicly. Contact Andrew Campbell's lab for access inquiries.

---

## 2. SABLE / PNAS Personality Study (N=624)

> Institution: Ludwig-Maximilians-Universität München (LMU), Germany
> Co-authors from: Stanford, UT Austin (but data collected at LMU)
> Period: 30 consecutive days per participant
> Paper: Stachl et al. "Predicting personality from patterns of behavior collected with smartphones" (PNAS 2020)
> DOI: 10.1073/pnas.1920484117
> PubMed: https://pubmed.ncbi.nlm.nih.gov/32665436/
> PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7395458/
> NSF Award: #1520288 "An Automated Technology-Based Personality Classifier"

### Overview

Large-scale smartphone sensing study specifically designed to predict Big Five personality from behavioral patterns. Part of the NSF-funded SABLE project, though data was collected in Germany, not at US universities.

### Data Collected

| Category | Details |
|---|---|
| **Smartphone sensors** | Calls, SMS, GPS, app usage, screen events, music |
| **Big Five** | ✅ BFSI (Big Five Structure Inventory) — 5 domains + 30 facets, 4-point Likert |
| **N logging events** | 25,347,089 total |
| **PHQ / PSS** | ❌ Not collected |
| **GPA** | ❌ Not collected |
| **Wearable** | ❌ Not used |

### Key Findings

- Big Five domain prediction: median r = 0.37
- Big Five facet prediction: median r = 0.40
- Communication/social behavior, app usage, and mobility most predictive
- Accuracy comparable to social media-based personality prediction

### Data Availability

**⚠️ Partially available.** Code and processed features on OSF: https://osf.io/kqjhr/. However, **raw data cannot be provided** due to privacy implications (German data protection law). Only aggregated/processed features are accessible.

### Note on BFSI vs BFI-44

The BFSI (used here) and BFI-44 (used in StudentLife) both measure the Big Five but differ in structure:
- BFSI: 300 items, 30 facets, 4-point scale (German)
- BFI-44: 44 items, 5 domains, 5-point scale (English)
- Domain-level scores are conceptually comparable but not directly equatable

---

## 3. Weichen Wang Personality Sensing (N=646)

> Institution: Dartmouth College
> Paper: "Sensing Behavioral Change over Time: Using Within-Person Variability Features from Mobile Sensing to Predict Personality Traits" (IMWUT/UbiComp 2018)
> DOI: 10.1145/3264951
> PDF: https://weichen.wang/paper/ubicomp18_personality.pdf

### Overview

Used within-person behavioral variability (not just averages) from smartphone sensing to predict personality traits in 646 college students.

### Data Availability

**❌ NOT publicly available.** No dataset download link found on author website or publication. Likely requires direct contact with Weichen Wang (now at CMU/industry) or Andrew Campbell at Dartmouth.

---

## Why These Don't Help Our Current Paper

| Dataset | Missing for Our Needs |
|---|---|
| StudentLife 2016 (N=83) | No Big Five, no GPA, not public |
| SABLE/PNAS (N=624) | No GPA, no mental health, raw data restricted, German sample |
| Weichen Wang (N=646) | Not public, details unclear |
