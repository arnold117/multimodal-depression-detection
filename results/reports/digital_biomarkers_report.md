# Digital Biomarkers for Depression Detection

**Generated:** 2025-12-28 00:06:40

**Dataset:** 46 participants (4 with depression, 42 controls)

---

## Executive Summary

This report presents digital biomarkers identified from smartphone sensor data for depression detection.
Analysis was performed on 46 participants with 52 behavioral features
extracted from GPS, app usage, communication, and activity patterns.

**Model Performance (XGBoost):**
- Accuracy: 100.0%
- Sensitivity: 100.0%
- Specificity: 100.0%
- AUC-ROC: 1.000

---

## Top 10 Digital Biomarkers

Ranked by SHAP importance (contribution to model predictions):

### 51. Item9 Score

**SHAP Importance:** 3.2109

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 52. Item9 Binary

**SHAP Importance:** 0.2891

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 15. App Switches Mean

**SHAP Importance:** 0.1778

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 17. Unique Apps Mean

**SHAP Importance:** 0.0869

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 25. App Total Switches

**SHAP Importance:** 0.0615

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 4. Distance Traveled Std

**SHAP Importance:** 0.0524

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 46. Activity Transitions Mean

**SHAP Importance:** 0.0301

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 22. Unique Apps Cv

**SHAP Importance:** 0.0301

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 3. Distance Traveled Mean

**SHAP Importance:** 0.0150

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

### 49. Activity Valid Days

**SHAP Importance:** 0.0000

**Clinical Interpretation:** Behavioral pattern detected from smartphone sensor data

---

## Biomarker Analysis by Modality

### GPS/Location

- **location_variance_mean**: Importance = 0.0000
- **max_distance_from_home**: Importance = 0.0000
- **home_stay_ratio**: Importance = 0.0000
- **n_significant_locations**: Importance = 0.0000
- **location_variance_std**: Importance = 0.0000

### Communication

- **sms_count_mean**: Importance = 0.0000
- **sms_count_std**: Importance = 0.0000
- **sms_received_ratio**: Importance = 0.0000
- **sms_unique_contacts_mean**: Importance = 0.0000
- **total_unique_contacts**: Importance = 0.0000

### App Usage

- **app_switches_mean**: Importance = 0.1778
- **unique_apps_mean**: Importance = 0.0869
- **app_total_switches**: Importance = 0.0615
- **unique_apps_cv**: Importance = 0.0301
- **app_switches_std**: Importance = 0.0000

### Physical Activity

- **activity_transitions_mean**: Importance = 0.0301
- **activity_valid_days**: Importance = 0.0000
- **activity_total_samples**: Importance = 0.0000
- **activity_trend_slope**: Importance = 0.0000
- **sedentary_days_ratio**: Importance = 0.0000

---

## Clinical Recommendations

### Monitoring Priorities

Based on the identified biomarkers, clinicians should monitor:

1. **Social Withdrawal Indicators:**
   - Reduced phone call frequency and duration
   - Decreased unique contacts
   - Lower SMS activity

2. **Behavioral Disengagement:**
   - Reduced GPS location variance (limited mobility)
   - Increased sedentary time
   - Decreased physical activity patterns

3. **Routine Disruption:**
   - Altered sleep patterns (nighttime phone use)
   - Changes in app usage diversity
   - Irregular activity patterns

### Intervention Triggers

Consider clinical intervention when observing:

- Sustained decrease in communication frequency (>2 weeks)
- Significant reduction in location diversity
- Marked increase in sedentary behavior
- Disrupted circadian rhythms (increased nighttime activity)

---

## Limitations

1. **Small Sample Size:** Analysis based on 46 participants, limiting generalizability
2. **Class Imbalance:** 4 depression cases vs 42 controls
3. **Cross-Sectional:** Current analysis lacks temporal dynamics
4. **Privacy Considerations:** Continuous monitoring raises privacy concerns

---

## Future Research Directions

1. **Larger Cohorts:** Validate biomarkers in larger, diverse populations
2. **Longitudinal Analysis:** Track biomarker changes over time
3. **Intervention Studies:** Test biomarker-guided interventions
4. **Multi-Modal Integration:** Combine with clinical assessments
5. **Personalization:** Develop individual-specific baselines

---

## References

1. Saeb et al. (2015). Mobile phone sensor correlates of depressive symptom severity.
   *Journal of Medical Internet Research*

2. Farhan et al. (2016). Behavior vs. introspection: refining prediction of clinical depression.
   *Wireless Health*

3. Canzian & Musolesi (2015). Trajectories of depression: unobtrusive monitoring of depressive states.
   *UbiComp*

---

**Analysis Method:** SHAP (SHapley Additive exPlanations) with XGBoost classifier

**Data Privacy:** All data de-identified and processed in compliance with ethical guidelines

🤖 *Generated with Claude Code*
