# Digital Biomarkers for Suicide Risk Prediction

Explainable predictive modeling to identify digital biomarkers of suicidal risk using the StudentLife Dataset.

## Project Overview

This research project develops machine learning models to predict suicidal ideation (PHQ-9 Item #9) from smartphone-based digital biomarkers. The analysis uses multimodal behavioral data from 46 college students collected through the StudentLife study.

**Research Goals:**
- Identify digital biomarkers that predict suicidal risk
- Build explainable models suitable for clinical interpretation
- Enable early detection through passive smartphone sensing

## Dataset

**StudentLife Dataset** (Dartmouth College, 2013)
- **Sample**: 46 participants with pre-assessment PHQ-9 data
- **Duration**: ~10 months of continuous data collection
- **Modalities**: 9 data types
  - App usage patterns (~2.9M records)
  - GPS mobility (~500KB per user)
  - Call/SMS communication (~4K records per user)
  - Physical activity sensors (~45.6M records)
  - Phone lock/screen state
  - Additional: Bluetooth, WiFi, audio, calendar, dining

**Outcome Measures:**
- **Primary**: PHQ-9 Item #9 (suicidal ideation) - Binary classification
  - Class 0: No ideation (42 users, 91.3%)
  - Class 1: Any ideation (4 users, 8.7%)
  - Users with ideation: u18, u19, u31, u50
  - Imbalance ratio: 10.50:1
- **Secondary**: PHQ-9 total score (depression severity)

## Implementation Roadmap

### Phase 1: Data Preprocessing (Weeks 1-2) ✓
- [x] Load and encode PHQ-9 survey data
- [x] Create outcome labels (Item #9 binary + PHQ-9 total)
- [x] Extract temporal alignment for behavioral data
- [x] Data quality assessment

**Status**: Complete
**Run**: `python scripts/01_preprocess_phq9.py`

### Phase 2: Feature Engineering (Weeks 3-5) 🔄
- [ ] Extract GPS mobility features (location variance, entropy)
- [ ] Extract communication features (call/SMS patterns)
- [ ] Extract app usage features (screen time, circadian patterns)
- [ ] Extract activity features (movement, sedentary behavior)
- [ ] Feature selection and dimensionality reduction (target: 20-40 features)

**Next**: Start with GPS features (strongest literature support)

### Phase 3: Baseline Modeling (Week 6)
- [ ] Logistic regression with L2 regularization
- [ ] 5-fold stratified cross-validation
- [ ] Evaluation metrics (AUC-ROC, PR-AUC, sensitivity, specificity)
- [ ] Feature importance analysis

### Phase 4: Model Optimization (Weeks 7-8)
- [ ] Random Forest and XGBoost models
- [ ] Hyperparameter tuning with nested CV
- [ ] Model comparison and selection
- [ ] Sensitivity analyses

### Phase 5: Interpretability (Week 9)
- [ ] SHAP value analysis
- [ ] Clinical interpretation of biomarkers
- [ ] Individual prediction explanations

### Phase 6: Manuscript Preparation (Weeks 10-12)
- [ ] Publication-quality figures
- [ ] Results tables
- [ ] Methods and results sections
- [ ] Discussion and limitations

## Project Structure

```
multimodal-depression-detection/
├── data/
│   ├── raw/dataset/               # Original StudentLife data (27GB)
│   ├── processed/
│   │   ├── labels/                # PHQ-9 outcome labels ✓
│   │   └── features/              # Extracted features
│   └── interim/
│       └── user_timelines/        # Temporal alignment ✓
├── src/
│   ├── preprocessing/             # PHQ-9 and temporal modules ✓
│   ├── features/                  # Feature extractors
│   ├── models/                    # ML models
│   └── visualization/             # Plotting utilities
├── scripts/
│   └── 01_preprocess_phq9.py     # Phase 1 script ✓
├── notebooks/
│   └── 01_eda_phq9.ipynb         # EDA notebook ✓
├── results/
│   ├── models/                    # Trained models
│   ├── figures/                   # Publication figures
│   ├── tables/                    # Results tables
│   └── metrics/                   # Performance metrics
└── literature_references.md       # Reference papers ✓
```

## Setup

### Requirements
- Python 3.8+
- 16GB RAM (for chunk-based feature processing)
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Clone repository
git clone <repository-url>
cd multimodal-depression-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Jupyter kernel
python -m ipykernel install --user --name=multimodal-depression
```

## Usage

### Phase 1: Preprocess PHQ-9 Data

```bash
# Run preprocessing script
python scripts/01_preprocess_phq9.py

# Or explore with notebook
jupyter notebook notebooks/01_eda_phq9.ipynb
```

**Outputs:**
- `data/processed/labels/item9_labels_pre.csv` - Suicidal ideation labels
- `data/processed/labels/phq9_labels_pre.csv` - Depression severity labels
- `data/interim/user_timelines/user_timelines.json` - Temporal alignment
- `data/interim/user_timelines/valid_users.csv` - Users with sufficient data

### Phase 2: Feature Engineering (In Progress)

```bash
# Coming soon
python scripts/02_extract_features.py
```

## Key Findings (Phase 1)

### Primary Outcome (Suicidal Ideation - PHQ-9 Item #9):
- **4 users (8.7%)** reported any suicidal ideation
- **Class imbalance**: 10.50:1 (severe imbalance)
- **Users with ideation**: u18, u19, u31, u50
- **Distribution**:
  - "Not at all": 42 users (91.3%)
  - "Several days": 2 users (4.3%) - u19, u50
  - "More than half the days": 2 users (4.3%) - u18, u31
  - "Nearly every day": 0 users (0%)

### Data Quality:
- **46 users** with pre-assessment data (100% of PHQ-9 pre)
- **38 users** with post-assessment (8 dropouts)
- **Mean data collection**: ~9 months per user
- **Sufficient data for modeling**: 46 users meet inclusion criteria (≥14 days)

## Literature Support

Key digital biomarkers from prior research:
- **GPS location variance** ↓ → Social withdrawal, anhedonia (Saeb et al., 2015)
- **Call/SMS frequency** ↓ → Social isolation (Farhan et al., 2016)
- **Physical activity** ↓ → Psychomotor retardation (Canzian & Musolesi, 2015)
- **Screen time** ↑ → Avoidance, rumination (Wang et al., 2014)

See [literature_references.md](literature_references.md) for complete references.

## Expected Performance

Based on similar studies predicting depression from smartphone data:
- **Baseline (Logistic Regression)**: AUC 0.60-0.70
- **Optimized (Ensemble)**: AUC 0.65-0.75
- **Stretch Goal**: AUC > 0.75

With n=46 and 6 positive cases, wide confidence intervals expected. Permutation tests critical for statistical validation.

## Methodology Highlights

- **Memory-efficient**: Chunk-based processing for 27GB dataset
- **Class imbalance**: Balanced class weights + stratified CV (no SMOTE with small n)
- **Interpretability**: SHAP values for clinical translation
- **Validation**: 5-fold stratified CV with permutation tests
- **Reproducibility**: Random seeds, version-locked dependencies, documented preprocessing

## Current Status

**Branch**: `feature/digital-biomarkers`
**Phase**: 1 (Preprocessing) ✓ Complete
**Next**: Phase 2 (Feature Engineering) - GPS features

**Last Updated**: December 5, 2025

## Contributing

This is a research project. For questions or collaborations, contact the project team.

## Ethical Considerations

- PHQ-9 Item #9 contains sensitive mental health data
- All data is de-identified (StudentLife dataset)
- Research use only - not for clinical deployment without validation
- Model fairness and bias testing required before any application

## License

Research use only. See dataset license for StudentLife data usage terms.

## Acknowledgments

- **StudentLife Dataset**: Dartmouth College (Wang et al., 2014)
- **Methodology**: Based on Saeb et al. (2015), Farhan et al. (2016), Canzian & Musolesi (2015)

## References

See [literature_references.md](literature_references.md) for complete bibliography.
