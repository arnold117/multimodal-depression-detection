# External Datasets Summary — Potential Validation Sources

> Last updated: 2026-02-17
> Purpose: Track all candidate datasets for validating our StudentLife personality-behavior-GPA findings

---

## Quick Comparison

| Dataset | N | Year | Big Five | GPA | Depression | Stress | Sensors | Public | Priority |
|---|---|---|---|---|---|---|---|---|---|
| **StudentLife (ours)** | 28 | 2013 | ✅ BFI-44 | ✅ Registrar | ✅ PHQ-9 | ✅ PSS | ✅ 13 phone modalities | ✅ | Current |
| **NetHealth** | **722** | 2015-2019 | **✅ BFI-44** | **✅ Registrar** | ✅ BDI+CES-D | ✅ | ✅ Fitbit+Comms | **✅ Free** | ⭐⭐⭐⭐⭐ |
| **GLOBEM** | **497** | 2018-2021 | **✅ BFI-10** | ⚠️ Academic Fit | **✅ BDI-II** | ✅ PSS-4 | **✅ Phone+Fitbit** | **✅ PhysioNet** | ⭐⭐⭐⭐ |
| **Copenhagen (CNS)** | **~1,000** | 2012-2016 | **✅ BFI** | **✅ Registrar** | **✅ MDI** | **✅ PSS** | **✅ BT+GPS+Comms** | **❌ Restricted** | ⭐⭐⭐ |
| **SNAPSHOT (MIT)** | ~321 | 2013-2017 | **✅ BFI** | **✅ Self-report** | ⚠️ SF-12 MCS | ✅ PSS | ✅ Wrist+Phone | **❌ Not public** | ⭐⭐⭐ |
| **CES (NDA #2494)** | 175-215 | 2017-2022 | ❌ | ✅ Self-report | ✅ PHQ-9 | ✅ PSS | ✅ Same app | NDA申请 | ⭐⭐⭐ |
| **StudentLife 2016** | 83 | 2016 | ❌ | ❌ | ✅ PHQ-8/4 | ❌ | ✅ Same app+Band | ❌ | ⭐ |
| **SABLE/PNAS** | 624 | ~2018 | ✅ BFSI | ❌ | ❌ | ❌ | ✅ Phone logs | ⚠️ No raw | ⭐ |
| **Weichen Wang** | 646 | 2018 | ✅ | ❓ | ❓ | ❓ | ✅ | ❌ | ⭐⭐ |
| **RADAR-MDD** | ~600 | 2016-2022 | ❌ | ❌ | ✅ PHQ-8 | ❌ | ✅ Phone+Fitbit | 需申请KCL | ⭐ |
| **NSSE** | varies | ongoing | ❌ | ❌ | ❌ | ❌ | ❌ | 院校登录 | — |
| **OpenICPSR** | varies | varies | — | — | — | — | — | 平台搜索 | — |

---

## Detailed Inventory (by priority)

### ⭐⭐⭐⭐⭐ NetHealth — University of Notre Dame (BEST FIT)

- **Source**: Notre Dame, Center for Network Science and Data
- **N**: 722 students, 8 survey waves, 4 years longitudinal
- **Why best fit**: **Same BFI-44 instrument**, registrar GPA, depression/anxiety/loneliness measures, Fitbit sensing, N=722, **free download**
- **Has**: Big Five (BFI-44), GPA (registrar), BDI, CES-D, STAI, BAI, SELSA loneliness, self-esteem, PSQI sleep, Fitbit (activity/sleep/HR), communication logs, social network
- **Missing**: No smartphone passive sensing (GPS, app usage, screen time), no PHQ-9 (has BDI/CES-D instead), no PANAS, no Flourishing
- **Can validate**: Personality→GPA (core!), Conscientiousness=#1 (SHAP), personality moderates behavior, cross-model consistency, LPA, ML framework
- **Access**: **Free** — fill short form, download from Google Drive
- **Full inventory**: [nethealth_data_inventory.md](nethealth_data_inventory.md)

### ⭐⭐⭐⭐ GLOBEM — University of Washington (STRONG COMPLEMENT)

- **Source**: University of Washington, AWARE framework + Fitbit
- **N**: 497 unique participants (~705 person-years, 4 annual cohorts)
- **Period**: 2018-2021, ~10 weeks per year
- **Why strong**: **BFI-10 personality**, BDI-II depression, PSS-4, PANAS, **both smartphone + Fitbit sensing**, N=497, publicly available on PhysioNet
- **Has**: BFI-10 (Big Five short form), BDI-II, PHQ-4, PSS-4, PANAS, ERQ, CHIPS, Academic Fit, GPS, screen, Bluetooth, calls, Fitbit (activity/sleep/HR)
- **Missing**: No raw GPA (has "Sense of Academic Fit" instead), BFI-10 is 10-item short form (vs our BFI-44)
- **Can validate**: Personality→mental health, behavior→depression (BDI-II), LPA, ML framework, cross-model consistency
- **Cannot validate**: Personality→GPA (no GPA), direct BFI-44 comparison (different instrument length)
- **Access**: **PhysioNet credentialed access** — sign DUA, free
- **Key paper**: Xu et al. "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling" (NeurIPS 2022)
- **PhysioNet**: https://physionet.org/content/globem/1.1/

### ⭐⭐⭐ Copenhagen Networks Study — DTU (HOLY GRAIL, DATA RESTRICTED)

- **Source**: Technical University of Denmark (DTU) + University of Copenhagen, PI: Sune Lehmann
- **N**: ~1,000 students (2013 deployment), ~250 (2012 pilot)
- **Period**: 2012-2016, up to 2.5 years per cohort
- **Why remarkable**: Has **everything** — BFI + institutional GPA + MDI depression + PSS stress + UCLA loneliness + PANAS + rich sensing, N=~1,000
- **Has**: BFI, registrar grades (DTU 7-point scale), MDI, PSS, UCLA Loneliness, PANAS, RSES self-esteem, SWLS life satisfaction, NAR-Q, self-efficacy, locus of control, Bluetooth proximity, GPS, WiFi, calls, SMS, Facebook, app usage, screen, sleep inference
- **Missing**: No Flourishing Scale, no PHQ-9 (has MDI instead), Danish grading system ≠ US GPA
- **Can validate**: Personality→GPA, Conscientiousness=#1, behavior→depression/stress/loneliness/affect, moderation, LPA, ML framework — **virtually everything**
- **Access**: **Restricted** — interaction data only on Figshare; questionnaire + grades require collaboration with Sune Lehmann (slj@dtu.dk). GDPR applies.
- **Full inventory**: [copenhagen_networks_data_inventory.md](copenhagen_networks_data_inventory.md)

### ⭐⭐⭐ SNAPSHOT — MIT Media Lab (PERFECT VARIABLES, DATA NOT PUBLIC)

- **Source**: MIT Media Lab, Affective Computing Group (Akane Sano, Rosalind Picard)
- **N**: ~321 across 7 cohorts (~50 undergrads per semester)
- **Period**: Fall 2013 onwards, ~30 days per participant
- **Why interesting**: Has **BFI + GPA + PSS + STAI + rich multimodal sensing** — variable combination matches perfectly
- **Has**: BFI, self-reported GPA, PSS, STAI, SF-12 MCS, MBTI, MEQ, daily mood/stress VAS, Q-sensor (EDA/temperature/accelerometer), phone (calls/SMS/screen/GPS), weather
- **Missing**: No depression-specific instrument (PHQ/BDI), only SF-12 mental health composite
- **Can validate**: Personality→GPA (core!), behavior→stress, behavior→mental health, ML framework
- **Access**: **NOT publicly available** — requires data sharing agreement with MIT Media Lab
- **Action needed**: Contact snapshot@media.mit.edu for data access inquiry
- **Key papers**: Sano et al. 2015 (IEEE BSN), Taylor & Jaques 2020 (IEEE TAFFC)

### ⭐⭐⭐ CES — College Experience Study

- **Source**: NDA Collection #2494, Dartmouth
- **N**: 175-215, 5 years longitudinal
- **Why useful**: Same school, same StudentLife app, has GPA + PHQ + PSS
- **Missing**: Big Five, UCLA Loneliness, Flourishing, PANAS
- **Can validate**: Behavior→PHQ, Behavior→PSS, SVR non-linear→GPA, LPA, ML framework
- **Cannot validate**: Personality→GPA, moderation effects, SHAP Conscientiousness
- **Access**: NDA Data Use Certification required (2-4 weeks)
- **Full inventory**: [ces_data_inventory.md](ces_data_inventory.md)

### ⭐⭐ Weichen Wang Personality Sensing (N=646)

- **Source**: Dartmouth, UbiComp 2018
- **Why interesting**: Large N, college students, personality + phone sensors
- **Missing**: Unclear if GPA/mental health included; data not public
- **Action needed**: Contact Weichen Wang directly for data access and variable list
- **Full inventory**: [studentlife_2016_sable_inventory.md](studentlife_2016_sable_inventory.md)

### ⭐ StudentLife 2nd Study (N=83)

- **Source**: Dartmouth, IMWUT 2018
- **Why considered**: Same school, same app, slightly larger N
- **Missing**: No Big Five, no GPA, data not released
- **Action needed**: Contact Andrew Campbell's lab; unlikely to be released
- **Full inventory**: [studentlife_2016_sable_inventory.md](studentlife_2016_sable_inventory.md)

### ⭐ SABLE/PNAS (N=624)

- **Source**: LMU Munich (Germany), PNAS 2020
- **Why considered**: Large N, Big Five (BFSI), smartphone sensing
- **Missing**: No GPA, no mental health, raw data restricted, German sample
- **Action needed**: Only processed features on OSF; raw data inaccessible
- **Full inventory**: [studentlife_2016_sable_inventory.md](studentlife_2016_sable_inventory.md)

### ⭐ RADAR-MDD (N≈600)

- **Source**: King's College London / VU Amsterdam / Barcelona
- **Why considered**: Large N, rich sensing + wearable, longitudinal
- **Missing**: No Big Five, no GPA, clinical MDD patients (not students)
- **Fundamental problem**: Population mismatch — adult psychiatric vs healthy students
- **Full inventory**: [radar_cns_data_inventory.md](radar_cns_data_inventory.md)

---

## Evaluated But Not Useful

### LifeSnaps (Aristotle University of Thessaloniki, N=71)

- **Source**: Scientific Data (Nature) 2022, EU Horizon 2020 RAIS project
- **Has**: IPIP Big Five (50-item), STAI anxiety, PANAS, rich Fitbit Sense data (32+ modalities including HRV, EDA, SpO2)
- **Missing**: **No GPA**, **no depression measure** (no PHQ/BDI/CES-D)
- **Population**: European university-affiliated adults (not pure student sample), 68% Master's degree holders
- **Public**: Yes (Kaggle + Zenodo, CC BY 4.0)
- **Why not useful**: No academic outcome and no depression measure. Only useful for personality→anxiety/affect with wearable data.

### StudentMEH (Worcester Polytechnic Institute, N=88)

- **Source**: IEEE Big Data 2024 Workshop, WPI Data Science
- **Has**: CES-D-10 depression, STAI anxiety, PSS-4 stress, Fitbit (HR/sleep/steps/distance/calories)
- **Missing**: **No Big Five**, **no GPA**
- **Population**: US undergraduates during COVID-19 (2020-2021)
- **Public**: Code only (GitHub); actual data not released
- **Why not useful**: Missing both personality and academic performance. COVID context limits generalizability.

### K-EmoPhone (KAIST, Korea, N=77)

- **Source**: Scientific Data (Nature) 2023
- **Has**: K-BFI-15 (Korean Big Five), emotion/stress/attention labels (5,582 EMAs), smartphone + Microsoft Band 2
- **Missing**: **No GPA**, only 7-day study period
- **Population**: Korean students
- **Public**: Yes (Zenodo)
- **Why not useful**: Too short (7 days), no academic outcome, Korean instrument not directly comparable to BFI-44.

### TILES-2018 (USC, N=212)

- **Source**: Scientific Data (Nature) 2020, USC SAIL Lab
- **Has**: Personality traits, well-being, job performance, Fitbit + biometric garment + smartphone + audio
- **Missing**: **No GPA** (workplace study, not students)
- **Population**: Hospital workers
- **Public**: Yes (tiles-data.isi.edu)
- **Why not useful**: Wrong population (hospital workers, not students). No academic performance variable. Methodologically excellent but population mismatch.

### Tesserae (Notre Dame + Multi-University, N=757)

- **Source**: CHI 2019, $7.9M NSF-funded
- **Has**: Personality, mood, anxiety, Garmin wearable + smartphone + Bluetooth beacons
- **Missing**: **No GPA** (workplace study)
- **Population**: US information workers
- **Public**: Restricted (DUA required, non-profit only)
- **Why not useful**: Workers not students, no GPA, restricted access. Large N but wrong population.

### NSSE (National Survey of Student Engagement)

- **Source**: Indiana University, https://nsse.indiana.edu/nsse/reports-data/raw.html
- **What it is**: Large-scale survey of student engagement at US colleges
- **Data**: Core survey items, population file, topical modules; CSV format
- **Access**: Institutional login required (per-school basis)
- **Why not useful**: No Big Five personality, no mental health measures, no sensor/behavioral data, no individual-level GPA. Purely engagement survey. Data access restricted to participating institutions.

### OpenICPSR

- **Source**: ICPSR, University of Michigan, https://www.openicpsr.org/openicpsr/
- **What it is**: Open data repository for social/behavioral science research
- **Why not useful**: General repository — searched for datasets with Big Five + GPA + smartphone sensing; no matching dataset found. Individual datasets may be useful for specific subquestions but none combine our core variables. Worth revisiting if new datasets are deposited.

---

## The Gap in the Literature (Updated)

~~No publicly available dataset currently combines all three: Big Five personality + GPA + smartphone passive sensing.~~

**UPDATE**: NetHealth comes very close — it has Big Five (BFI-44) + Registrar GPA + Fitbit behavioral sensing + depression/anxiety/loneliness. The only gap is that it uses **Fitbit wearable** rather than **smartphone passive sensing** (no GPS, app usage, screen time). This is a meaningful difference but not a dealbreaker for validation.

**NEW**: GLOBEM adds another strong option — BFI-10 + BDI-II + smartphone + Fitbit, but no GPA. SNAPSHOT (MIT) has the perfect variable combination but data is not publicly available.

| What exists | What's missing | Dataset |
|---|---|---|
| **Big Five + GPA + Wearable + Mental Health** | No smartphone passive sensing | **NetHealth** ✅ |
| **Big Five + Depression + Phone + Fitbit** | No GPA | **GLOBEM** ✅ |
| **Big Five + GPA + Depression + Stress + Loneliness + PANAS + Sensors** | Restricted (GDPR) | **Copenhagen (CNS)** |
| Big Five + GPA + Phone + Wrist sensors | Not public | SNAPSHOT (MIT) |
| Big Five + Sensors (phone) | No GPA, no mental health | SABLE |
| GPA + Sensors (phone) + Mental Health | No Big Five | CES |
| Big Five + GPA + Phone Sensors + Mental Health | N=28 only | StudentLife (ours) |
| Large N + Phone Sensors + Mental Health | No personality, no GPA, wrong population | RADAR |

---

## Recommended Strategy (Updated)

### Primary path: NetHealth validation (RECOMMENDED)

1. **Download NetHealth data** (free, immediate)
2. Replicate **Personality→GPA** with N=722 (core finding, same BFI-44)
3. Run SHAP analysis → confirm Conscientiousness #1
4. Test moderation: Big Five × Fitbit activity → GPA
5. Run LPA on Fitbit behavioral features → mental health outcomes
6. Apply full ML framework (4 models, Optuna, SHAP, cross-model τ)
7. Frame paper as **Study 1 (StudentLife, discovery) + Study 2 (NetHealth, validation)**

### Secondary path: Add GLOBEM for behavioral-mental health validation

1. Apply for PhysioNet credentialed access (sign DUA)
2. Replicate Personality→Depression with BFI-10 + BDI-II (N=497)
3. Replicate Behavior→Depression with phone + Fitbit (N=497)
4. Three-dataset paper: StudentLife (discovery) → NetHealth (personality-GPA) → GLOBEM (personality-depression)

### Tertiary path: Add CES for sensor validation

1. Apply for NDA access (2-4 weeks)
2. Replicate Behavior→PHQ/PSS with same phone sensors (N=200+)
3. Four-dataset paper: StudentLife → NetHealth → GLOBEM → CES

### Long shot: SNAPSHOT (MIT)

- Contact snapshot@media.mit.edu for data sharing
- If available: has BFI + GPA + sensors, would be excellent validation

### Impact on Q1 targeting

| Strategy | Q1 Probability | Effort |
|---|---|---|
| StudentLife alone | 10-20% | Done |
| + NetHealth validation | **50-60%** | 2-3 weeks analysis |
| + NetHealth + GLOBEM | **60-70%** | + PhysioNet DUA |
| + NetHealth + GLOBEM + CES | 70-80% | + NDA application |

---

## Comprehensive Search Log

Datasets systematically searched and evaluated (19 total):

| # | Dataset | Source | Evaluated | Result |
|---|---|---|---|---|
| 1 | StudentLife (ours) | Dartmouth | ✅ | Current dataset |
| 2 | NetHealth | Notre Dame | ✅ | ⭐⭐⭐⭐⭐ Best fit |
| 3 | GLOBEM | UW | ✅ | ⭐⭐⭐⭐ Strong complement |
| 4 | SNAPSHOT | MIT Media Lab | ✅ | ⭐⭐⭐ Perfect vars, not public |
| 5 | CES | Dartmouth/NDA | ✅ | ⭐⭐⭐ No personality |
| 6 | Weichen Wang | Dartmouth | ✅ | ⭐⭐ Not public |
| 7 | StudentLife 2016 | Dartmouth | ✅ | ⭐ Not public, no BFI/GPA |
| 8 | SABLE/PNAS | LMU Munich | ✅ | ⭐ No GPA, restricted |
| 9 | RADAR-MDD | KCL/VU/Barcelona | ✅ | ⭐ Wrong population |
| 10 | LifeSnaps | Aristotle U | ✅ | ✗ No GPA, no depression |
| 11 | StudentMEH | WPI | ✅ | ✗ No BFI, no GPA |
| 12 | K-EmoPhone | KAIST | ✅ | ✗ No GPA, 7 days only |
| 13 | TILES-2018 | USC | ✅ | ✗ Workers, no GPA |
| 14 | Tesserae | Notre Dame | ✅ | ✗ Workers, no GPA |
| 15 | NSSE | Indiana U | ✅ | ✗ No BFI/MH/sensors |
| 16 | OpenICPSR | Michigan | ✅ | ✗ No matching dataset |
| 17 | CrossCheck | Dartmouth/Cornell | ✅ | ✗ Schizophrenia patients |
| 18 | Copenhagen (CNS) | DTU/UCPH | ✅ | ⭐⭐⭐ Holy grail vars, restricted |
| 19 | NetSense | Notre Dame | ✅ | ⭐ NetHealth predecessor, no GPA/MH confirmed |

---

## Document Index

| File | Contents |
|---|---|
| [nethealth_data_inventory.md](nethealth_data_inventory.md) | **NetHealth full variable inventory — BEST FIT** |
| [globem_data_inventory.md](globem_data_inventory.md) | **GLOBEM full variable inventory — STRONG COMPLEMENT** |
| [snapshot_data_inventory.md](snapshot_data_inventory.md) | SNAPSHOT (MIT) full inventory — perfect vars, not public |
| [copenhagen_networks_data_inventory.md](copenhagen_networks_data_inventory.md) | Copenhagen Networks Study — holy grail, restricted |
| [ces_data_inventory.md](ces_data_inventory.md) | CES full variable inventory (8 data structures) |
| [radar_cns_data_inventory.md](radar_cns_data_inventory.md) | RADAR-MDD full inventory + assessment |
| [studentlife_2016_sable_inventory.md](studentlife_2016_sable_inventory.md) | StudentLife 2016 + SABLE/PNAS + Weichen Wang |
| **this file** | Master summary and strategy |
