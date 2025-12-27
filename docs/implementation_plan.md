# Implementation Plan: Digital Biomarkers for Suicide Risk Prediction

## Project Overview
Develop explainable predictive models to identify digital biomarkers that impact suicidal risk using the StudentLife Dataset (n=46 participants, 9 data modalities, 27GB raw data).

## Key Decisions & Clarifications (Confirmed)

### 1. Temporal Window Strategy ✓
- **Decision**: Use ALL available data before pre-assessment (full collection period)
- **Rationale**: Maximum data for feature extraction, captures long-term behavioral patterns
- **Future work**: Test sensitivity with shorter windows (1-4 weeks) in Phase 4

### 2. PHQ-9 Assessment Timing ✓
- **Challenge**: PHQ-9.csv has no timestamp column
- **Solution**: Infer assessment dates from behavioral data boundaries
  - **Pre-assessment**: Use earliest timestamp across all modalities per user (data collection start)
  - **Post-assessment**: Use latest timestamp (data collection end)
  - **Source**: EMA response timestamps (`resp_time` field in Unix format) + sensor data
- **Implementation**: Create `temporal_alignment.py` to extract min/max timestamps per user

### 3. Dual Outcomes Priority ✓
- **Primary outcome**: PHQ-9 Item #9 (suicidal ideation) - binary classification
  - Column: "Thoughts that you would be better off dead, or of hurting yourself"
  - Encoding: Class 0 = "Not at all", Class 1 = Any ideation (score >0)
  - **Expected distribution**: ~40-42 Class 0 (91%), ~4-6 Class 1 (9%) based on PHQ-9.csv review
- **Secondary outcome**: PHQ-9 total score (clinical depression threshold ≥10)
- **Paper focus**: Item #9 as main results, total score as supplementary

### 4. Computing Resources & Framework ✓
- **GPU**: GTX 1070 available but NOT used for Phase 1-4 (traditional ML is CPU-based)
- **ML Framework**: PyTorch available, but Phase 1-4 uses scikit-learn for traditional ML
- **Future PyTorch use**: Optional in Phase 5 if exploring neural networks
- **Memory requirement**: ≥16GB RAM for chunk-based feature processing
- **Processing time**: Feature extraction ~2-4 hours

### 5. Code Management ✓
- **Strategy**: Create NEW clean code structure in separate git branch
- **Branch name**: `feature/digital-biomarkers` (protects main branch)
- **Existing code**: Keep exploration notebook as reference, don't modify
- **Structure**: Fresh modular implementation (src/, scripts/, notebooks/)

## Research Goals
1. **Dual outcome approach**: Predict both PHQ-9 Item #9 (suicidal ideation) and PHQ-9 total score (depression severity)
2. **Explainable modeling**: Interpretable models suitable for research publication
3. **Resource-efficient**: Traditional ML methods (no deep learning due to GTX 1070 GPU constraints and small sample)
4. **Priority modalities**: Phone usage, GPS mobility, social interactions (calls/SMS), activity levels

## Implementation Phases

### Phase 1: Data Preprocessing & Ground Truth (Week 1-2)

**Objective**: Create labeled dataset with PHQ-9 outcomes and establish temporal alignment

**Key Files**:
- Input: [data/raw/dataset/survey/PHQ-9.csv](data/raw/dataset/survey/PHQ-9.csv)
- Output: `data/processed/labels/phq9_labels.csv`, `data/processed/labels/item9_labels.csv`

**CRITICAL FINDING - Class Distribution Analysis**:
From PHQ-9.csv review (Item #9: "Thoughts that you would be better off dead, or of hurting yourself"):

**Pre-assessment (n=46) - PRIMARY OUTCOME**:
- "Not at all": 42 users (91.3%)
- "Several days": 2 users (4.3%): u19, u50
- "More than half the days": 2 users (4.3%): u18, u31
- "Nearly every day": 0 users (0%)
- **Binary classification**: Class 0 = 42 users (91.3%), Class 1 = 4 users (8.7%)
- **Imbalance ratio**: 10.50:1 (severe imbalance)
- **Statistical power**: With n=46 and only 4 positive cases, expect very wide confidence intervals; permutation tests critical, may need Leave-One-Out CV

**Post-assessment (n=38) - LONGITUDINAL ANALYSIS**:
- "Not at all": 33 users (86.8%)
- "Several days": 3 users (7.9%): u19, u23, u52
- "More than half the days": 2 users (5.3%): u18, u33
- "Nearly every day": 0 users (0%)
- **Binary classification**: Class 0 = 33 users (86.8%), Class 1 = 5 users (13.2%)
- **Imbalance ratio**: 6.60:1

**Longitudinal Patterns (Pre → Post)**:
- **Persistent**: 2 users (u18, u19) - ideation at both timepoints
- **Resolved**: 2 users (u31, u50) - had ideation at pre, resolved at post
- **New onset**: 3 users (u23, u33, u52) - developed ideation by post
- **Dropouts**: 8 users did not complete post-assessment

**Tasks**:
1. **Load PHQ-9 survey data** (84 responses: 46 pre-assessment, 38 post-assessment)
   - Encode ordinal responses: "Not at all"=0, "Several days"=1, "More than half the days"=2, "Nearly every day"=3

2. **Create Primary Outcome - PHQ-9 Item #9 (Suicidal Ideation)**
   - Extract column: "Thoughts that you would be better off dead, or of hurting yourself"
   - **Binary classification** (RECOMMENDED): Class 0 = "Not at all", Class 1 = Any ideation (score >0)
   - Clinical rationale: ANY suicidal ideation warrants attention

3. **Create Secondary Outcome - PHQ-9 Total Score**
   - Sum all 9 items (range 0-27)
   - **Binary classification**: Clinical depression (score ≥10) vs. not (<10)
   - Alternative: Keep continuous for correlation analysis

4. **Analyze class distribution** to understand imbalance (expect ~10-20% positive class for Item #9)

5. **Define temporal strategy**:
   - Use **pre-assessment** as primary outcome (n=46, maximum sample size)
   - Extract all behavioral data BEFORE pre-assessment date
   - Future work: Post-assessment prediction (n=38) and change detection

6. **User inclusion criteria**:
   - Must have PHQ-9 data
   - Minimum 14 days of data in priority modalities (app usage, GPS, calls/SMS, activity)
   - Document any excluded users

**Deliverables**:
- Processed label files with user IDs and outcomes
- EDA notebook: `notebooks/01_eda_phq9.ipynb` with class distribution, missing data analysis
- Data quality report

---

### Phase 2: Feature Engineering (Week 3-5)

**Objective**: Extract digital biomarkers from 4 priority modalities using memory-efficient chunk-based processing

**Strategy**:
- Process each user individually (never load all 27GB into memory)
- Aggregate raw data → daily summaries → user-level statistics
- Target 20-40 features total (avoid overfitting with n=46)

#### 2.1 Phone Usage Features (Week 3)

**Key Files**: [data/raw/dataset/app_usage/running_app_u*.csv](data/raw/dataset/app_usage/) (~2.9M records)

**Features to Extract**:
- **Usage volume**: Mean daily app switches, unique apps per day
- **Circadian disruption**: Night usage ratio (0-5am), circadian regularity score
- **Variability**: Coefficient of variation in daily usage (isolation/instability marker)
- **Weekend patterns**: Weekend vs weekday usage ratio

**Output**: `data/processed/features/app_usage_features.parquet` (~8-10 features per user)

#### 2.2 GPS Mobility Features (Week 3)

**Key Files**: [data/raw/dataset/sensing/gps/gps_u*.csv](data/raw/dataset/sensing/gps/) (49 users)

**Features to Extract** (Based on Saeb et al., 2015 - strongest depression predictors):
- **Location variance**: Mean and std of daily location variance (KEY BIOMARKER)
- **Location diversity**: Mean number of significant locations visited (DBSCAN clustering, eps=0.001)
- **Distance traveled**: Mean daily distance (haversine formula)
- **Home stay time**: Proportion of time at home location (social withdrawal proxy)
- **Movement entropy**: Predictability of hourly movement patterns
- **Regularity**: Coefficient of variation in location variance

**Data quality**:
- Filter GPS points with accuracy >100m
- Skip days with <5 valid GPS points

**Output**: `data/processed/features/gps_mobility_features.parquet` (~10-12 features per user)

#### 2.3 Social Interaction Features (Week 4)

**Key Files**:
- [data/raw/dataset/call_log/call_log_u*.csv](data/raw/dataset/call_log/) (~2.2K records/user)
- [data/raw/dataset/sms/sms_u*.csv](data/raw/dataset/sms/) (~1.8K records/user)

**Features to Extract**:
- **Call patterns**: Mean daily call count/duration, outgoing/incoming ratio, unique contacts
- **SMS patterns**: Mean daily SMS count, sent/received ratio, unique SMS contacts
- **Social diversity**: Combined unique contacts across call and SMS
- **Communication variability**: CV of daily communication (social withdrawal marker)
- **Temporal trend**: Linear regression slope over time (increasing isolation?)

**Output**: `data/processed/features/communication_features.parquet` (~12-15 features per user)

#### 2.4 Physical Activity Features (Week 4)

**Key Files**: [data/raw/dataset/sensing/activity/activity_u*.csv](data/raw/dataset/sensing/activity/) (~45.6M records - LARGE!)

**Processing Strategy**: CHUNK-BASED (100K rows per chunk) due to large file size

**Features to Extract**:
- **Activity levels**: Mean daily still ratio, moving ratio (activity codes: 0=vehicle, 1=bicycle, 2=foot, 3=still)
- **Sedentary behavior**: Proportion of days with >70% still ratio (depression marker)
- **Activity variability**: Std of daily moving ratio
- **Activity fragmentation**: Mean daily activity state transitions
- **Temporal trend**: Activity level change over time

**Output**: `data/processed/features/activity_features.parquet` (~8-10 features per user)

#### 2.5 Feature Integration & Selection (Week 5)

**Tasks**:
1. **Merge modality features** into combined matrix: `data/processed/features/combined_features.parquet`
2. **Handle missing data**:
   - Impute with median for secondary features
   - Add missingness indicators (compliance signal)
   - Feature: days of data coverage per modality
3. **Dimensionality reduction**:
   - Remove highly correlated features (Pearson r > 0.9)
   - Domain-driven selection (prioritize literature-validated features)
   - Univariate feature selection (mutual information, f-statistics)
   - Target: 20-40 final features
4. **Feature scaling**: Standardize for logistic regression/SVM

**Deliverables**:
- Final feature matrix: (46 users × 20-40 features)
- Feature engineering notebook: `notebooks/02_feature_engineering.ipynb`
- Feature documentation (definitions, clinical interpretation)

---

### Phase 3: Baseline Modeling (Week 6)

**Objective**: Establish interpretable baseline with logistic regression

**Model**: Logistic Regression with L2 regularization
```python
LogisticRegressionCV(
    penalty='l2',
    solver='liblinear',
    cv=StratifiedKFold(5),
    scoring='roc_auc',
    class_weight='balanced'
)
```

**Cross-Validation Strategy**:
- **5-fold Stratified CV** (ensures balanced class distribution per fold)
- If severe imbalance (e.g., only 4-6 positive cases), fall back to Leave-One-Out CV

**Class Imbalance Handling**:
- `class_weight='balanced'` (preferred over SMOTE for small sample)
- Stratified sampling in CV folds

**Evaluation Metrics** (for imbalanced data):
1. **AUC-ROC**: Overall discrimination
2. **Precision-Recall AUC**: More sensitive to imbalance
3. **Sensitivity (Recall)**: Clinical priority (don't miss positive cases) - TARGET ≥0.80
4. **Specificity**: True negative rate
5. **Brier Score**: Probability calibration

**Statistical Validation**:
- Permutation test (p-value): Is model better than chance?
- Bootstrap confidence intervals (1000 iterations)
- Report CV mean and std (stability)

**Interpretation**:
- Feature coefficients → odds ratios
- Identify top 5-10 predictive features
- Map to clinical constructs (e.g., location variance → social withdrawal)

**Success Criteria**:
- AUC > 0.60 (better than chance)
- Permutation test p < 0.05
- Identify significant digital biomarkers

**Deliverables**:
- Baseline model: `results/models/logistic_baseline.pkl`
- Performance report: `results/metrics/baseline_metrics.json`
- Feature importance plot: `results/figures/logistic_coefficients.png`
- Notebook: `notebooks/03_baseline_modeling.ipynb`

---

### Phase 4: Model Optimization (Week 7-8)

**Objective**: Improve performance with ensemble methods while maintaining interpretability

**Models to Compare**:

1. **Random Forest** (balance of performance and interpretability)
   ```python
   RandomForestClassifier(
       n_estimators=500,
       max_depth=3,  # Shallow to prevent overfitting
       min_samples_split=5,
       class_weight='balanced'
   )
   ```
   - Feature importance via Gini impurity
   - Hyperparameter tuning: max_depth (2-5), n_estimators (100-500), min_samples_split (3-7)

2. **XGBoost** (maximum performance)
   ```python
   XGBClassifier(
       n_estimators=100,
       max_depth=3,
       learning_rate=0.1,
       scale_pos_weight=ratio  # Class imbalance
   )
   ```
   - Built-in regularization
   - SHAP values for interpretation

**Hyperparameter Tuning**:
- **Nested Cross-Validation**: Outer 5-fold for evaluation, inner 3-fold for tuning
- Prevents optimistic bias in performance estimates

**Model Comparison**:
- McNemar's test for statistical significance of differences
- Compare: AUC-ROC, PR-AUC, sensitivity, CV stability
- Select best model based on performance + interpretability tradeoff

**Sensitivity Analyses**:
1. Different feature subsets (modality ablation: what if we remove GPS? Communication?)
2. Different temporal windows (last 1 week, 2 weeks, 4 weeks vs. full period)
3. Different outcome definitions (Item #9 binary vs. multi-class, PHQ-9 ≥10 vs. ≥15)

**Deliverables**:
- Trained models: `results/models/{rf, xgb}_model.pkl`
- Model comparison table: `results/tables/model_comparison.csv`
- Notebook: `notebooks/04_model_comparison.ipynb`

---

### Phase 5: Interpretability & Clinical Translation (Week 9)

**Objective**: Generate publication-ready explanations linking digital biomarkers to clinical constructs

**SHAP Analysis** (Model-agnostic explanation):
```python
import shap
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)
```

**Visualizations to Create**:
1. **SHAP Summary Plot**: Global feature importance with directionality
2. **SHAP Dependence Plots**: Feature-outcome relationships (top 5 features)
3. **SHAP Force Plots**: Individual prediction explanations (case examples)

**Clinical Interpretation Framework**:

Map digital biomarkers → clinical constructs (with literature support):

| Digital Biomarker | Clinical Construct | Reference |
|-------------------|-------------------|-----------|
| Location variance ↓ | Social withdrawal, anhedonia | Saeb et al., 2015 |
| Call/SMS frequency ↓ | Social isolation | Farhan et al., 2016 |
| Activity (moving) ↓ | Psychomotor retardation | Canzian & Musolesi, 2015 |
| Screen time ↑ | Avoidance, rumination | Wang et al., 2014 |
| Circadian disruption | Sleep disturbance | - |
| Communication variability ↑ | Mood instability | - |

**Feature Importance Report**:
- Top 10 predictive features with SHAP values
- Direction of effect (protective vs. risk factor)
- Clinical interpretation for each
- Example cases (high vs. low risk profiles)

**Deliverables**:
- SHAP plots: `results/figures/shap_*.png`
- Clinical interpretation report: `results/interpretation_report.md`
- Feature importance table: `results/tables/feature_importance.csv`

---

### Phase 6: Results & Manuscript Preparation (Week 10-12)

**Objective**: Complete results package for research paper submission

**Publication Figures** (high-resolution, 300 DPI):

1. **Figure 1: Model Performance** (4-panel)
   - Panel A: ROC curves (all models)
   - Panel B: Precision-Recall curves
   - Panel C: Calibration plot
   - Panel D: Cross-validation stability (boxplots)

2. **Figure 2: Feature Importance**
   - SHAP summary plot (beeswarm)
   - Top 10 features with confidence intervals

3. **Figure 3: Clinical Insights**
   - Feature distributions by outcome (violin plots)
   - Example temporal patterns (behavioral changes)

**Results Tables**:

1. **Table 1: Sample Characteristics**
   - Demographics (if available)
   - PHQ-9 score distribution
   - Data availability by modality

2. **Table 2: Model Performance**
   - All models with metrics (AUC-ROC, PR-AUC, sensitivity, specificity)
   - 95% confidence intervals
   - Statistical significance tests

3. **Table 3: Top Predictive Features**
   - Feature name, importance score, direction, clinical interpretation

**Manuscript Sections to Draft**:

**Methods**:
- Dataset description (StudentLife, n=46, modalities)
- Feature engineering (formulas, temporal aggregation)
- Missing data handling
- Model selection and hyperparameter tuning
- Cross-validation strategy
- Evaluation metrics

**Results**:
- Sample characteristics and data quality
- Model performance (primary: Item #9, secondary: PHQ-9 total)
- Feature importance and clinical interpretation
- Sensitivity analyses

**Discussion**:
- Key findings: Which digital biomarkers predict suicide risk?
- Clinical implications: Early warning signs, passive monitoring
- Comparison to literature
- Limitations: Small sample, single cohort, self-report, temporal ambiguity
- Future directions: Validation in larger samples, real-time monitoring, intervention studies

**Deliverables**:
- All publication figures: `results/figures/figure*.png`
- All tables: `results/tables/table*.csv`
- Manuscript draft: `manuscript_draft.md`
- Supplementary materials: Feature definitions, model details, additional analyses

---

## Technical Implementation Details

### Code Organization
```
multimodal-depression-detection/
├── src/
│   ├── preprocessing/
│   │   ├── phq9_processor.py          # PHQ-9 encoding and labeling
│   │   └── temporal_alignment.py      # Date extraction and windowing
│   ├── features/
│   │   ├── base_extractor.py          # Abstract base class
│   │   ├── app_usage.py               # Phone usage features
│   │   ├── gps_mobility.py            # GPS features with DBSCAN
│   │   ├── communication.py           # Call/SMS features
│   │   ├── activity.py                # Activity sensor (chunk processing)
│   │   └── feature_selector.py        # Dimensionality reduction
│   ├── models/
│   │   ├── baseline.py                # Logistic regression
│   │   ├── ensemble.py                # RF, XGBoost
│   │   └── evaluation.py              # Metrics, CV, statistical tests
│   └── visualization/
│       ├── performance_plots.py       # ROC, PR, calibration
│       └── shap_plots.py              # SHAP visualizations
├── scripts/
│   ├── 01_preprocess_phq9.py
│   ├── 02_extract_features.py
│   ├── 03_train_models.py
│   └── 04_generate_results.py
├── notebooks/
│   ├── 01_eda_phq9.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_modeling.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_interpretation.ipynb
└── requirements.txt
```

### Key Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
shap>=0.41.0
matplotlib>=3.4.0
seaborn>=0.11.0
geopy>=2.2.0
pyarrow>=6.0.0
tqdm>=4.62.0
```

### Memory Management
- **Never load all data**: Process per-user files individually
- **Chunk processing**: Use `pd.read_csv(chunksize=100000)` for large activity files
- **Immediate aggregation**: Raw data → daily summaries → user statistics
- **Compressed storage**: Use Parquet format for intermediate features
- **GPU not required**: Traditional ML (scikit-learn) runs on CPU

---

## Critical Files to Modify/Create

### Input Files (Read-only)
1. [data/raw/dataset/survey/PHQ-9.csv](data/raw/dataset/survey/PHQ-9.csv) - Ground truth labels
2. [data/raw/dataset/sensing/gps/gps_u*.csv](data/raw/dataset/sensing/gps/) - GPS data (49 files)
3. [data/raw/dataset/app_usage/running_app_u*.csv](data/raw/dataset/app_usage/) - App usage (49 files)
4. [data/raw/dataset/call_log/call_log_u*.csv](data/raw/dataset/call_log/) - Call logs (49 files)
5. [data/raw/dataset/sms/sms_u*.csv](data/raw/dataset/sms/) - SMS data (49 files)
6. [data/raw/dataset/sensing/activity/activity_u*.csv](data/raw/dataset/sensing/activity/) - Activity data (49 files)

### Output Files (To Create)
1. `data/processed/labels/phq9_labels.csv` - Encoded outcomes
2. `data/processed/features/combined_features.parquet` - Final feature matrix
3. `results/models/best_model.pkl` - Final trained model
4. `results/figures/` - All publication figures
5. `results/tables/` - All results tables
6. `literature_references.md` - Reference list (CREATED)

---

## Expected Outcomes & Success Criteria

### Performance Benchmarks
Based on literature (Saeb et al., 2015: AUC=0.69; Farhan et al., 2016: AUC=0.76):
- **Baseline (Logistic Regression)**: AUC 0.60-0.70
- **Optimized (RF/XGBoost)**: AUC 0.65-0.75
- **Stretch goal**: AUC > 0.75

### Publication Success Criteria
- ✓ AUC significantly > 0.5 (p < 0.05)
- ✓ Identify 3-5 significant digital biomarkers with clinical interpretation
- ✓ Rigorous methodology (nested CV, permutation tests, confidence intervals)
- ✓ Explainable models (SHAP, feature importance)
- ✓ Reproducible code and documentation

---

## Risk Mitigation

**Risk 1: Severe class imbalance** (e.g., only 4-6 positive cases)
- **Mitigation**: Use Leave-One-Out CV, focus on PR-AUC and sensitivity, consider expanded outcome (Item #9>0 OR PHQ-9≥10)

**Risk 2: Overfitting with n=46**
- **Mitigation**: Strong L2 regularization, nested CV, limit features to 20-40, permutation tests, report CV variance

**Risk 3: Low performance** (AUC < 0.60)
- **Mitigation**: Frame as exploratory/proof-of-concept, emphasize feature discovery over prediction, discuss need for larger samples

**Risk 4: Missing data reduces sample**
- **Mitigation**: Relax inclusion criteria (≥7 days instead of 14), impute secondary features, include missingness as features

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1. Data Preprocessing | Weeks 1-2 | PHQ-9 labels, EDA report |
| 2. Feature Engineering | Weeks 3-5 | Combined feature matrix (46×20-40) |
| 3. Baseline Modeling | Week 6 | Logistic regression baseline, feature importance |
| 4. Model Optimization | Weeks 7-8 | Best model selection, sensitivity analyses |
| 5. Interpretability | Week 9 | SHAP explanations, clinical interpretation |
| 6. Manuscript | Weeks 10-12 | Complete results package, manuscript draft |

**Total**: 12 weeks (3 months)

---

## Next Steps - Implementation Kickoff

### Step 0: Git Branch Setup (Protect Main Branch Integrity)
```bash
cd /home/arnold/Documents/multimodal-depression-detection
git checkout -b feature/digital-biomarkers
git push -u origin feature/digital-biomarkers
```

### Step 1: Environment Setup
```bash
# Create requirements.txt with dependencies
# Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Project Structure Creation
```bash
# Create directory structure as outlined in "Code Organization" section
mkdir -p src/{preprocessing,features,models,visualization,utils}
mkdir -p scripts notebooks results/{models,figures,tables,metrics}
mkdir -p data/processed/{labels,features,splits}
mkdir -p data/interim/user_timelines
```

### Step 3: Phase 1 - Data Preprocessing
1. **Create EDA notebook**: `notebooks/01_eda_phq9.ipynb`
   - Load and explore PHQ-9.csv
   - Verify class distribution (expect 40 vs 6 for Item #9)
   - Analyze PHQ-9 total score distribution
   - Document missing data patterns
2. **Implement `src/preprocessing/phq9_processor.py`**: Encode PHQ-9 responses and create labels
3. **Implement `src/preprocessing/temporal_alignment.py`**: Extract assessment dates from EMA/sensor timestamps

### Step 4: Phase 2 - Feature Engineering
1. **Start with GPS features** (strongest literature support): `src/features/gps_mobility.py`
2. **Then communication features**: `src/features/communication.py`
3. **Then app usage**: `src/features/app_usage.py`
4. **Then activity**: `src/features/activity.py` (chunk-based processing for large files)

### Step 5: Phase 3 - Baseline Modeling
1. **Create baseline notebook**: `notebooks/03_baseline_modeling.ipynb`
2. **Implement logistic regression** with 5-fold stratified CV
3. **Generate first results**: Feature importance, performance metrics

### Ready to Begin!
All planning complete. Proceed with git branch creation and environment setup when ready to implement.
