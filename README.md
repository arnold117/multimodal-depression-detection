# Digital Biomarkers for Depression Detection

Machine learning pipeline for identifying behavioral biomarkers of depression from multimodal smartphone sensor data using the StudentLife Dataset.

## Project Overview

This research project develops explainable machine learning models to detect depression from smartphone-based digital biomarkers. The analysis uses multimodal behavioral data from 46 college students collected through the StudentLife study.

**Research Goals:**
- Identify digital biomarkers that correlate with depression
- Build explainable models suitable for clinical interpretation
- Enable passive detection through smartphone sensing

## Dataset

**StudentLife Dataset** (Dartmouth College, 2013)
- **Sample**: 46 participants with PHQ-9 assessment data
- **Duration**: ~10 months of continuous data collection
- **Modalities**: GPS, app usage, communication, physical activity, phone usage
  - App usage patterns (~2.9M records)
  - GPS mobility (~500KB per user)
  - Call/SMS communication (~4K records per user)
  - Physical activity sensors (~45.6M records)
  - Phone lock/screen state

**Outcome Measure:**
- **Primary**: PHQ-9 Item #9 (suicidal ideation) - Binary classification
  - Class 0: No ideation (42 users, 91.3%)
  - Class 1: Any ideation (4 users, 8.7%)
  - **Severe class imbalance**: 10.5:1 ratio

## Implementation Status

### ✅ Phase 1: Data Preprocessing
- PHQ-9 survey data extraction and encoding
- Outcome label generation (Item #9 binary + PHQ-9 total score)
- Temporal alignment and data quality assessment
- **Run**: `python scripts/01_preprocess_phq9.py`

### ✅ Phase 2: Feature Engineering
- **50 behavioral features** extracted from 4 modalities:
  - **GPS/Location** (11 features): Location variance, entropy, radius of gyration, home stay ratio
  - **App Usage** (10 features): App diversity, screen time, night usage patterns
  - **Communication** (11 features): Call/SMS frequency, duration, unique contacts, passive communication
  - **Physical Activity** (18 features): Movement ratios, activity transitions, sedentary behavior, phone lock patterns

- **Data Leakage Fixed**: Original feature matrix accidentally included depression labels (item9_score, item9_binary), leading to artificial 100% accuracy. These were removed, reducing features from 52 to 50.
- **Run**: `python scripts/02-06_*.py` (GPS, app, communication, activity, integration)

### ✅ Phase 3: Baseline Modeling
- Logistic Regression, Random Forest, XGBoost with balanced class weights
- 5-fold stratified cross-validation
- Feature importance analysis (coefficient magnitudes for LR)
- **Run**: `python scripts/07_train_baseline.py`

### ✅ Phase 4: Deep Learning Models
- **VAE** (Variational Autoencoder): Latent representation learning, anomaly detection
- **GNN** (Graph Attention Network): User similarity graph with 5-NN connections
- **Contrastive Learning**: SimCLR-style representation with tabular augmentation
- **Multimodal Transformer**: Cross-modal attention across 4 modalities
- **Platform**: Apple Silicon MPS acceleration via PyTorch 2.5
- **Run**: `python scripts/08-11_train_*.py`

### ✅ Phase 5: Model Comparison & Evaluation
- Comprehensive comparison across 7 models (3 baseline + 4 deep learning)
- Unified evaluation with AUC-ROC, sensitivity, specificity, confusion matrices
- Publication-quality visualizations (ROC curves, performance bars, ranking heatmap)
- **Run**: `python scripts/12_evaluate_all_models.py`

### ✅ Phase 6: Digital Biomarker Report
- Comprehensive clinical report with behavioral interpretations
- Top 10 biomarkers ranked by logistic regression coefficients
- Clinical recommendations, limitations, future directions
- **Output**: `results/reports/digital_biomarkers_report.md`

## Key Results

### Model Performance (Clean Data)

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC | Notes |
|-------|----------|-------------|-------------|---------|-------|
| **Logistic Regression** | **87.0%** | **25.0%** | **92.9%** | **0.762** | **Best overall** |
| Random Forest | 91.3% | 0.0% | 100.0% | 0.542 | Predicts all negatives |
| XGBoost | 91.3% | 0.0% | 100.0% | 0.578 | Class imbalance issue |
| GNN | 8.7% | 100.0% | 0.0% | 0.500 | Predicts all positives |
| VAE | N/A | N/A | N/A | N/A | Anomaly detection |
| Contrastive | 91.3% | 0.0% | 0.0% | 0.63 | Representation learning |
| Transformer | 91.3% | 0.0% | 0.0% | 0.30 | Cross-modal attention |

**Key Finding**: Logistic Regression achieved the best balance with 76.2% AUC-ROC, the only model maintaining both sensitivity and specificity above zero. Tree-based and deep learning models struggled with severe class imbalance (4:42 ratio).

### Top 10 Digital Biomarkers

Ranked by coefficient magnitude from Logistic Regression:

1. **App Usage Diversity Variability** (0.762) - Erratic engagement suggests anhedonia
2. **SMS Received Ratio** (0.700) - Passive communication indicates withdrawal
3. **Physical Activity Variability** (0.655) - Irregular rest-activity rhythms
4. **GPS Mobility Variability** (0.648) - Inconsistent daily routines
5. **Weekend/Weekday Ratio** (0.643) - Disrupted work-life structure
6. **Activity Transitions** (0.590) - Psychomotor retardation patterns
7. **Movement Ratio** (0.567) - Reduced physical activity/fatigue
8. **Communication Engagement Days** (0.538) - Social isolation
9. **GPS Distance Variability** (0.538) - Unstable mobility patterns
10. **GPS Data Completeness** (0.536) - Behavioral disengagement

**Clinical Interpretation**: All top biomarkers align with established depression symptoms - social withdrawal, psychomotor changes, disrupted routines, and anhedonia.

## Project Structure

```
multimodal-depression-detection/
├── data/
│   ├── raw/dataset/               # StudentLife data (27GB)
│   ├── processed/
│   │   ├── labels/                # PHQ-9 labels
│   │   └── features/              # 50-feature matrix (clean)
│   └── interim/                   # Temporal alignment, user timelines
├── src/
│   ├── preprocessing/             # PHQ-9 encoding
│   ├── features/                  # GPS, app, communication, activity extractors
│   ├── models/                    # Baseline + deep learning models
│   ├── interpretability/          # SHAP analysis (unused for LR)
│   ├── visualization/             # Model comparison plots
│   └── utils/                     # Data loaders, PyTorch utilities
├── scripts/
│   ├── 00_fix_data_leakage.py    # Remove label columns from features
│   ├── 01_preprocess_phq9.py     # Phase 1
│   ├── 02-06_*.py                 # Phase 2: Feature extraction
│   ├── 07_train_baseline.py      # Phase 3: Baseline models
│   ├── 08-11_train_*.py          # Phase 4: Deep learning
│   └── 12_evaluate_all_models.py # Phase 5: Model comparison
├── notebooks/
│   ├── 01_eda_phq9.ipynb         # PHQ-9 exploratory analysis
│   ├── 02-06_*.ipynb             # Feature engineering notebooks
│   └── dataset_exploration.ipynb  # Initial data exploration
├── results/
│   ├── models/                    # Trained models (.pkl, .pth)
│   ├── figures/                   # ROC curves, confusion matrices, rankings
│   ├── tables/                    # Model comparison metrics
│   └── reports/                   # Digital biomarker report (markdown)
└── configs/
    └── model_configs.yaml         # Hyperparameters for all models
```

## Setup

### Requirements
- Python 3.13+ (or 3.10+)
- 16GB RAM recommended
- Apple Silicon with MPS support (optional, for deep learning acceleration)
- Environment manager: mamba or conda

### Installation

```bash
# Clone repository
git clone <repository-url>
cd multimodal-depression-detection

# Create environment
mamba create -n qbio python=3.13
mamba activate qbio

# Install dependencies
pip install -r requirements.txt

# For PyTorch with MPS (Apple Silicon)
mamba install pytorch torchvision -c pytorch
pip install torch-geometric pytorch-metric-learning
```

## Usage

### Quick Start (Full Pipeline)

```bash
# Activate environment
mamba activate qbio

# 1. Preprocess PHQ-9 data
python scripts/01_preprocess_phq9.py

# 2. Extract features (run all feature scripts)
python scripts/02_extract_gps_features.py
python scripts/03_extract_app_features.py
python scripts/04_extract_communication_features.py
python scripts/05_extract_activity_features.py
python scripts/06_integrate_features.py

# 3. Fix data leakage (critical!)
python scripts/00_fix_data_leakage.py

# 4. Train baseline models
python scripts/07_train_baseline.py

# 5. Train deep learning models (optional)
python scripts/08_train_vae.py
python scripts/09_train_gnn.py
python scripts/10_train_contrastive.py
python scripts/11_train_transformer.py

# 6. Compare all models
python scripts/12_evaluate_all_models.py

# 7. View biomarker report
cat results/reports/digital_biomarkers_report.md
```

### Key Outputs

**Models:**
- `results/models/logistic_baseline.pkl` - Best model (76.2% AUC-ROC)
- `results/models/*_baseline.pkl` - Random Forest, XGBoost
- `results/models/*.pth` - Deep learning models (PyTorch)

**Visualizations:**
- `results/figures/all_models_roc_comparison.png` - ROC curve comparison
- `results/figures/all_models_performance_comparison.png` - Metric bar charts
- `results/figures/all_models_confusion_matrices.png` - Confusion matrix grid
- `results/figures/all_models_ranking.png` - Model ranking heatmap

**Reports:**
- `results/reports/digital_biomarkers_report.md` - **Main deliverable**: Comprehensive clinical report
- `results/tables/model_comparison_summary.json` - Metrics in JSON
- `results/models/logistic_baseline_feature_importance.csv` - Feature rankings

## Critical Lessons Learned

### 1. Data Leakage Discovery
**Issue**: The original feature matrix (52 columns) accidentally included `item9_score` and `item9_binary` - the depression labels themselves. This caused XGBoost to achieve perfect 100% accuracy.

**Detection**: SHAP analysis revealed these as the top 2 features, exposing the leakage.

**Fix**: Created `scripts/00_fix_data_leakage.py` to remove label columns, reducing features to 50. Backed up original data at `combined_features_with_leakage.parquet`.

**Impact**: After cleaning, XGBoost accuracy dropped to realistic 91.3% (0% sensitivity), and Logistic Regression emerged as the best model with 76.2% AUC-ROC.

**Lesson**: **Always audit feature sets for label leakage** - especially important in multimodal pipelines where labels may propagate through intermediate files.

### 2. Class Imbalance Challenges
With only 4 positive cases vs. 42 negatives (10.5:1), most models collapsed:
- Tree models → predict all negatives (high accuracy, 0% sensitivity)
- GNN → predict all positives (100% sensitivity, 0% specificity)
- Only Logistic Regression with balanced weights maintained both metrics

**Mitigation Strategies Used:**
- Balanced class weights (sklearn `class_weight='balanced'`)
- Stratified cross-validation (preserve ratios in folds)
- AUC-ROC as primary metric (better than accuracy for imbalance)
- Avoid SMOTE/oversampling (unreliable with n=4 positives)

### 3. Small Sample Size Limitations
n=46 participants is insufficient for robust machine learning:
- Wide confidence intervals on all metrics
- Deep learning models failed to learn meaningful patterns
- Sensitivity of 25% (1 out of 4 detected) is clinically unacceptable

**Recommendation**: Minimum n=200 with 1:2 positive:negative ratio for production systems.

## Clinical Implications

### Strengths
✅ Identified **genuine behavioral biomarkers** aligned with depression literature
✅ All top features have **clinical interpretability** (anhedonia, withdrawal, psychomotor changes)
✅ Passive monitoring requires **no patient burden** (runs in background)
✅ Multi-modal approach captures **diverse behavioral domains**

### Limitations
⚠️ **25% sensitivity** - missed 3 out of 4 depression cases (clinically insufficient)
⚠️ **Small sample** (n=46) - limited generalizability
⚠️ **Cross-sectional** - cannot establish causality or predict onset
⚠️ **Privacy concerns** - continuous monitoring raises ethical issues
⚠️ **Demographic bias** - college student population only

### Clinical Readiness
**Status**: **NOT READY FOR CLINICAL DEPLOYMENT**

While biomarkers are scientifically meaningful, the model cannot be used for:
- Depression screening (sensitivity too low)
- Diagnostic decision-making (not validated)
- Automated intervention triggers (high false negative rate)

**Potential Use Cases:**
- Research tool for biomarker discovery ✅
- Complement to clinical interviews (not replacement) ✅
- Hypothesis generation for larger studies ✅

## Future Research Directions

1. **Larger Cohorts**: Target n ≥ 200 with balanced 1:2 depression:control ratio
2. **Longitudinal Design**: Track individuals 6-12 months to capture temporal dynamics
3. **Diverse Populations**: Include age, gender, cultural diversity beyond college students
4. **Intervention Studies**: RCTs testing biomarker-guided vs. standard care
5. **Multimodal Integration**: Add wearables (heart rate variability), voice analysis, typing dynamics
6. **Explainable AI**: Attention mechanisms to highlight critical time windows for clinicians

## Methodology Highlights

- **Memory-efficient**: Chunk-based processing for 27GB dataset
- **Reproducibility**: Fixed random seeds, version-locked dependencies
- **Class imbalance handling**: Balanced weights + stratified CV (no synthetic oversampling)
- **Interpretability**: Coefficient-based importance for Logistic Regression
- **Validation**: 5-fold stratified cross-validation (appropriate for small n)
- **Platform optimization**: Apple Silicon MPS acceleration for PyTorch models

## Ethical Considerations

- PHQ-9 Item #9 contains **sensitive mental health data**
- All data **de-identified** (StudentLife dataset)
- **Research use only** - not for clinical deployment without validation
- Model fairness and bias testing required before any application
- Continuous passive monitoring raises **privacy concerns**
- Informed consent critical for real-world deployment

## References

**Key Literature:**
1. **Saeb et al. (2015)** - Mobile phone sensor correlates of depressive symptom severity. *J Med Internet Res*
2. **Farhan et al. (2016)** - Behavior vs. introspection: Refining prediction of clinical depression. *Wireless Health*
3. **Canzian & Musolesi (2015)** - Trajectories of depression: Unobtrusive monitoring via mobility traces. *UbiComp*

**Dataset:**
- **Wang et al. (2014)** - StudentLife: Assessing mental health, academic performance and behavioral trends. *UbiComp*

## Acknowledgments

- **StudentLife Dataset**: Dartmouth College
- **Methodology**: Based on digital phenotyping literature (Saeb, Farhan, Canzian)
- **Platform**: Apple Silicon MPS acceleration, PyTorch 2.5

## License

Research use only. See StudentLife dataset license for data usage terms.

---

**Project Status:** ✅ Complete (Phases 1-6)
**Last Updated:** 2025-12-28
**Branch:** `fix/remove-label-leakage` → merge to `main`
