# Literature References: Digital Biomarkers for Suicide Risk Detection

## Key Papers on Digital Phenotyping for Mental Health

### Foundational Studies

1. **StudentLife Dataset Original Paper**
   - Wang, R., Chen, F., Chen, Z., et al. (2014). "StudentLife: assessing mental health, academic performance and behavioral trends of college students using smartphones." Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing.
   - Established baseline for smartphone-based passive sensing for mental health

2. **Phone Usage Patterns & Depression**
   - Saeb, S., Zhang, M., Karr, C. J., et al. (2015). "Mobile phone sensor correlates of depressive symptom severity in daily-life behavior: an exploratory study." Journal of Medical Internet Research, 17(7), e175.
   - Found GPS mobility patterns (location variance, circadian movement) strongly correlated with PHQ-9 scores
   - Screen time duration associated with depression severity

3. **Social Interaction Digital Biomarkers**
   - Farhan, A. A., Yue, C., Morillo, R., et al. (2016). "Behavior vs. introspection: refining prediction of clinical depression via smartphone sensing data." 2016 IEEE Wireless Health (WH).
   - Call/SMS frequency and social interaction patterns as predictors
   - Bluetooth proximity as social contact proxy

4. **Activity & Movement Patterns**
   - Canzian, L., & Musolesi, M. (2015). "Trajectories of depression: unobtrusive monitoring of depressive states by means of smartphone mobility traces analysis." Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing.
   - Mobility entropy and location diversity decreased with depression
   - Regularity of movement patterns changed with mood states

### Suicide Risk Specific Research

5. **Digital Markers of Suicidal Ideation**
   - Coppersmith, G., Leary, R., Crutchley, P., & Fine, A. (2018). "Natural language processing of social media as screening for suicide risk." Biomedical informatics insights, 10.
   - Language patterns and behavioral changes preceding suicidal ideation
   - Social withdrawal as digital marker

6. **Smartphone Sensors for Suicide Prevention**
   - Kleiman, E. M., Turner, B. J., Fedor, S., et al. (2017). "Examination of real-time fluctuations in suicidal ideation and its risk factors: Results from two ecological momentary assessment studies." Journal of abnormal psychology, 126(6), 726.
   - EMA (ecological momentary assessment) for real-time suicide risk monitoring
   - Within-person variability of mood and stress as risk factors

### Explainable ML for Healthcare

7. **Interpretable Models in Healthcare**
   - Caruana, R., Lou, Y., Gehrke, J., et al. (2015). "Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission." Proceedings of the 21th ACM SIGKDD.
   - Importance of model interpretability for clinical adoption
   - GAMs (Generalized Additive Models) and rule-based models

8. **SHAP for Healthcare Applications**
   - Lundberg, S. M., Nair, B., Vavilala, M. S., et al. (2018). "Explainable machine-learning predictions for the prevention of hypoxaemia during surgery." Nature Biomedical Engineering, 2(10), 749-760.
   - SHAP (SHapley Additive exPlanations) for model interpretation
   - Feature importance and interaction effects

## Digital Biomarkers by Modality

### Phone Usage Patterns
- **Screen time**: Total daily usage, usage during nighttime hours
- **App categories**: Social media, communication, entertainment, productivity
- **Usage fragmentation**: Number of app-switching events
- **Diurnal patterns**: Circadian rhythm disruption indicators

### Mobility & GPS
- **Location variance**: Spatial dispersion of visited locations
- **Circadian movement**: Regularity of daily movement patterns
- **Entropy**: Predictability/unpredictability of location patterns
- **Home stay**: Time spent at home/residence location
- **Transition time**: Time spent in transit vs. stationary

### Social Interactions
- **Call/SMS frequency**: Total number per day/week
- **Interaction diversity**: Number of unique contacts
- **Response latency**: Time to respond to incoming messages
- **Initiated vs. received**: Ratio of outgoing to incoming communication
- **Social rhythms**: Regularity of social contact patterns

### Activity & Movement
- **Physical activity levels**: Time in different activity states
- **Sedentary periods**: Duration of stationary/inactive states
- **Activity fragmentation**: Breaks in sedentary behavior
- **Sleep-wake patterns**: Inferred from phone lock and activity data

### Additional Behavioral Markers
- **Sleep quality**: Phone lock patterns as sleep proxy
- **Conversation time**: Audio-based conversation detection
- **Social proximity**: Bluetooth co-location with others
- **Academic engagement**: Class attendance, assignment completion

## Feature Engineering Approaches

### Temporal Aggregations
- **Daily summaries**: Mean, variance, min, max per day
- **Weekly patterns**: Day-of-week variations, weekend vs. weekday
- **Longitudinal trends**: Changes over weeks/months
- **Variability metrics**: Standard deviation, coefficient of variation

### Derived Features
- **Entropy measures**: Shannon entropy of location, activity, app usage
- **Regularity scores**: Autocorrelation of daily patterns
- **Change detection**: Differences from baseline/typical behavior
- **Contextual features**: Time-of-day, day-of-week interactions

## Methodological Considerations

### Sample Size & Power
- StudentLife: n=49-46 participants (limited for deep learning)
- Suitable for traditional ML (logistic regression, random forest, gradient boosting)
- Cross-validation critical for small sample generalization

### Class Imbalance
- Suicidal ideation typically rare (often <20% prevalence)
- SMOTE, class weighting, or stratified sampling may be needed
- Precision-recall curves more informative than ROC for imbalanced data

### Missing Data
- Passive sensing has gaps (phone off, battery dead, sensor failures)
- Imputation strategies vs. exclusion criteria
- Missingness itself as a feature (data collection compliance)

### Temporal Validation
- Time-based cross-validation (not random splits)
- Pre-assessment data to predict post-assessment outcomes
- Avoid data leakage from future to past

## Ethics & Privacy

### Considerations
- PHQ-9 Item #9 is sensitive (suicidal ideation disclosure)
- De-identification verification critical
- Model fairness across demographic groups
- Clinical validation before deployment

### Reporting Standards
- TRIPOD guidelines for prediction model reporting
- Confusion matrix, calibration curves, clinical utility metrics
- Subgroup analysis and generalization discussion

---

**Last Updated**: December 5, 2025
**Purpose**: Reference material for multimodal suicide risk prediction research
