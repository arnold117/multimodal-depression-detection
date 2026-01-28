"""
DSM-5 Mental Health Knowledge Schema.

Defines the clinical framework for Major Depressive Disorder (MDD)
and mappings between digital behaviors and clinical symptoms.

References:
- DSM-5 (Diagnostic and Statistical Manual of Mental Disorders, 5th Edition)
- PHQ-9 (Patient Health Questionnaire-9)

IMPORTANT: This is a research tool, NOT a diagnostic instrument.
Clinical diagnosis requires professional evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MDDSymptom(Enum):
    """
    DSM-5 Major Depressive Disorder Diagnostic Criteria.

    At least 5 symptoms present for 2+ weeks, including at least one of:
    - Depressed mood
    - Anhedonia (loss of interest/pleasure)

    Reference: DSM-5, Section II, Depressive Disorders
    """

    DEPRESSED_MOOD = "depressed_mood"
    ANHEDONIA = "anhedonia"  # Loss of interest or pleasure
    WEIGHT_CHANGE = "weight_change"  # Significant weight loss/gain
    SLEEP_DISTURBANCE = "sleep_disturbance"  # Insomnia or hypersomnia
    PSYCHOMOTOR_CHANGE = "psychomotor_change"  # Agitation or retardation
    FATIGUE = "fatigue"  # Fatigue or loss of energy
    WORTHLESSNESS = "worthlessness"  # Feelings of worthlessness/guilt
    CONCENTRATION = "concentration_difficulty"  # Diminished concentration
    SUICIDAL_IDEATION = "suicidal_ideation"  # Thoughts of death/suicide


class PHQ9Item(Enum):
    """
    PHQ-9 items mapped to DSM-5 symptoms.

    The PHQ-9 is a validated screening tool that maps directly to
    DSM-5 MDD criteria.
    """

    Q1_ANHEDONIA = "little_interest_pleasure"  # Maps to ANHEDONIA
    Q2_DEPRESSED = "feeling_down_depressed"  # Maps to DEPRESSED_MOOD
    Q3_SLEEP = "trouble_sleeping"  # Maps to SLEEP_DISTURBANCE
    Q4_FATIGUE = "feeling_tired"  # Maps to FATIGUE
    Q5_APPETITE = "poor_appetite_overeating"  # Maps to WEIGHT_CHANGE
    Q6_WORTHLESS = "feeling_bad_about_self"  # Maps to WORTHLESSNESS
    Q7_CONCENTRATION = "trouble_concentrating"  # Maps to CONCENTRATION
    Q8_PSYCHOMOTOR = "moving_speaking_slowly"  # Maps to PSYCHOMOTOR_CHANGE
    Q9_SUICIDAL = "thoughts_of_death"  # Maps to SUICIDAL_IDEATION


# PHQ-9 to DSM-5 mapping
PHQ9_TO_DSM5: Dict[PHQ9Item, MDDSymptom] = {
    PHQ9Item.Q1_ANHEDONIA: MDDSymptom.ANHEDONIA,
    PHQ9Item.Q2_DEPRESSED: MDDSymptom.DEPRESSED_MOOD,
    PHQ9Item.Q3_SLEEP: MDDSymptom.SLEEP_DISTURBANCE,
    PHQ9Item.Q4_FATIGUE: MDDSymptom.FATIGUE,
    PHQ9Item.Q5_APPETITE: MDDSymptom.WEIGHT_CHANGE,
    PHQ9Item.Q6_WORTHLESS: MDDSymptom.WORTHLESSNESS,
    PHQ9Item.Q7_CONCENTRATION: MDDSymptom.CONCENTRATION,
    PHQ9Item.Q8_PSYCHOMOTOR: MDDSymptom.PSYCHOMOTOR_CHANGE,
    PHQ9Item.Q9_SUICIDAL: MDDSymptom.SUICIDAL_IDEATION,
}


class EvidenceLevel(Enum):
    """Evidence strength for symptom-behavior mappings."""

    STRONG = "strong"  # Multiple RCTs or meta-analyses
    MODERATE = "moderate"  # Observational studies with replication
    WEAK = "weak"  # Single study or theoretical basis
    HYPOTHETICAL = "hypothetical"  # Theoretically plausible, untested


@dataclass
class DigitalBehavior:
    """
    A measurable digital behavior from smartphone sensors.

    Attributes:
        name: Unique identifier
        description: Human-readable description
        sensor_source: Which sensor provides this data
        measurement_unit: Unit of measurement
        interpretation: How high/low values should be interpreted
    """

    name: str
    description: str
    sensor_source: str
    measurement_unit: str
    high_interpretation: str  # What high values indicate
    low_interpretation: str  # What low values indicate


@dataclass
class SymptomBehaviorMapping:
    """
    Mapping between a DSM-5 symptom and digital behaviors.

    This encodes the research hypothesis that specific digital behaviors
    can serve as proxies for clinical symptoms.

    IMPORTANT: These mappings are research hypotheses, not validated
    diagnostic criteria.
    """

    symptom: MDDSymptom
    digital_behaviors: List[str]
    evidence_level: EvidenceLevel
    mechanism: str  # Theoretical mechanism explaining the relationship
    references: List[str]  # Literature citations
    notes: str = ""


# =============================================================================
# DIGITAL BEHAVIOR DEFINITIONS
# =============================================================================

DIGITAL_BEHAVIORS: Dict[str, DigitalBehavior] = {
    # GPS/Mobility behaviors
    "home_stay_ratio": DigitalBehavior(
        name="home_stay_ratio",
        description="Proportion of time spent at home location",
        sensor_source="gps",
        measurement_unit="proportion (0-1)",
        high_interpretation="Social withdrawal, reduced activity",
        low_interpretation="Active social life, work engagement",
    ),
    "location_entropy": DigitalBehavior(
        name="location_entropy",
        description="Diversity/randomness of visited locations",
        sensor_source="gps",
        measurement_unit="bits",
        high_interpretation="Varied routine, exploration",
        low_interpretation="Restricted movement, routine disruption",
    ),
    "total_distance": DigitalBehavior(
        name="total_distance",
        description="Total distance traveled per day",
        sensor_source="gps",
        measurement_unit="kilometers",
        high_interpretation="Active lifestyle",
        low_interpretation="Sedentary behavior, possible avolition",
    ),
    "circadian_movement": DigitalBehavior(
        name="circadian_movement",
        description="Regularity of daily movement patterns",
        sensor_source="gps",
        measurement_unit="correlation coefficient",
        high_interpretation="Stable routine",
        low_interpretation="Disrupted circadian rhythm",
    ),
    # Activity behaviors
    "still_ratio": DigitalBehavior(
        name="still_ratio",
        description="Proportion of time in stationary state",
        sensor_source="activity",
        measurement_unit="proportion (0-1)",
        high_interpretation="Psychomotor retardation, fatigue",
        low_interpretation="Active, engaged",
    ),
    "activity_transitions": DigitalBehavior(
        name="activity_transitions",
        description="Number of activity state changes per day",
        sensor_source="activity",
        measurement_unit="count/day",
        high_interpretation="Dynamic activity pattern",
        low_interpretation="Low energy, possible psychomotor change",
    ),
    # Phone usage behaviors
    "unlock_count": DigitalBehavior(
        name="unlock_count",
        description="Number of phone unlocks per day",
        sensor_source="phonelock",
        measurement_unit="count/day",
        high_interpretation="Frequent phone checking (possible anxiety)",
        low_interpretation="Reduced phone engagement",
    ),
    "night_usage_ratio": DigitalBehavior(
        name="night_usage_ratio",
        description="Phone usage during night hours (0-5am)",
        sensor_source="phonelock",
        measurement_unit="proportion (0-1)",
        high_interpretation="Sleep disturbance, insomnia",
        low_interpretation="Normal sleep pattern",
    ),
    "session_duration_mean": DigitalBehavior(
        name="session_duration_mean",
        description="Average phone session length",
        sensor_source="phonelock",
        measurement_unit="minutes",
        high_interpretation="Extended escapism, rumination aid",
        low_interpretation="Brief, purposeful use",
    ),
    # Social behaviors
    "conversation_count": DigitalBehavior(
        name="conversation_count",
        description="Number of face-to-face conversations detected",
        sensor_source="conversation",
        measurement_unit="count/day",
        high_interpretation="Social engagement",
        low_interpretation="Social withdrawal",
    ),
    "call_count": DigitalBehavior(
        name="call_count",
        description="Number of phone calls per day",
        sensor_source="call_log",
        measurement_unit="count/day",
        high_interpretation="Social connection",
        low_interpretation="Social isolation",
    ),
    "unique_contacts": DigitalBehavior(
        name="unique_contacts",
        description="Number of unique people contacted per day",
        sensor_source="call_log",
        measurement_unit="count/day",
        high_interpretation="Diverse social network",
        low_interpretation="Narrow social circle, withdrawal",
    ),
    # Audio environment
    "silence_ratio": DigitalBehavior(
        name="silence_ratio",
        description="Proportion of time in silent environment",
        sensor_source="audio",
        measurement_unit="proportion (0-1)",
        high_interpretation="Isolation, quiet environment seeking",
        low_interpretation="Social environments",
    ),
}


# =============================================================================
# SYMPTOM-BEHAVIOR MAPPINGS (Literature-Based)
# =============================================================================

DSM5_BEHAVIOR_MAPPINGS: List[SymptomBehaviorMapping] = [
    # ANHEDONIA - Loss of interest or pleasure
    SymptomBehaviorMapping(
        symptom=MDDSymptom.ANHEDONIA,
        digital_behaviors=[
            "home_stay_ratio",
            "location_entropy",
            "conversation_count",
            "unique_contacts",
        ],
        evidence_level=EvidenceLevel.STRONG,
        mechanism=(
            "Anhedonia reduces motivation for social activities and exploration, "
            "leading to increased time at home and reduced location diversity."
        ),
        references=[
            "Saeb2015_MobilePhoneData",
            "Canzian2015_TrajectoryMining",
            "Wang2014_StudentLife",
        ],
    ),
    # DEPRESSED MOOD
    SymptomBehaviorMapping(
        symptom=MDDSymptom.DEPRESSED_MOOD,
        digital_behaviors=[
            "home_stay_ratio",
            "still_ratio",
            "silence_ratio",
        ],
        evidence_level=EvidenceLevel.MODERATE,
        mechanism=(
            "Depressed mood leads to social withdrawal and reduced activity, "
            "reflected in increased home time and sedentary behavior."
        ),
        references=[
            "Wahle2016_MobileSensing",
            "BenZeev2015_CrossCheck",
        ],
    ),
    # SLEEP DISTURBANCE
    SymptomBehaviorMapping(
        symptom=MDDSymptom.SLEEP_DISTURBANCE,
        digital_behaviors=[
            "night_usage_ratio",
            "circadian_movement",
            "unlock_count",
        ],
        evidence_level=EvidenceLevel.STRONG,
        mechanism=(
            "Sleep disturbances manifest as nighttime phone usage and irregular "
            "movement patterns. Frequent checking may indicate anxiety-driven insomnia."
        ),
        references=[
            "Abdullah2016_CircadianRhythm",
            "Sano2015_StressRecognition",
            "Murnane2016_SleepPhones",
        ],
    ),
    # FATIGUE
    SymptomBehaviorMapping(
        symptom=MDDSymptom.FATIGUE,
        digital_behaviors=[
            "still_ratio",
            "activity_transitions",
            "total_distance",
        ],
        evidence_level=EvidenceLevel.MODERATE,
        mechanism=(
            "Fatigue reduces physical activity, resulting in more sedentary time, "
            "fewer activity transitions, and less daily movement."
        ),
        references=[
            "Farhan2016_Mobility",
            "Saeb2016_Behavior",
        ],
    ),
    # PSYCHOMOTOR CHANGE
    SymptomBehaviorMapping(
        symptom=MDDSymptom.PSYCHOMOTOR_CHANGE,
        digital_behaviors=[
            "activity_transitions",
            "still_ratio",
            "session_duration_mean",
        ],
        evidence_level=EvidenceLevel.MODERATE,
        mechanism=(
            "Psychomotor retardation reduces activity transitions and increases "
            "sedentary time. Agitation may show as restless phone checking."
        ),
        references=[
            "Jacobson2020_PhoneSensors",
        ],
    ),
    # CONCENTRATION DIFFICULTY
    SymptomBehaviorMapping(
        symptom=MDDSymptom.CONCENTRATION,
        digital_behaviors=[
            "unlock_count",
            "session_duration_mean",
        ],
        evidence_level=EvidenceLevel.WEAK,
        mechanism=(
            "Concentration difficulties may lead to frequent task-switching "
            "and phone checking. Short sessions indicate inability to focus."
        ),
        references=[
            "Mark2014_Multitasking",
        ],
        notes="Limited direct evidence; primarily theoretical.",
    ),
    # SOCIAL WITHDRAWAL (not a DSM criterion but important behavioral marker)
    SymptomBehaviorMapping(
        symptom=MDDSymptom.ANHEDONIA,  # Most closely related
        digital_behaviors=[
            "call_count",
            "unique_contacts",
            "conversation_count",
        ],
        evidence_level=EvidenceLevel.STRONG,
        mechanism=(
            "Social withdrawal is a key behavioral manifestation of depression, "
            "reflected in reduced phone calls, fewer contacts, and less face-to-face interaction."
        ),
        references=[
            "Wang2018_CrossCheck",
            "Madan2010_SocialSensing",
        ],
    ),
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_behaviors_for_symptom(symptom: MDDSymptom) -> List[str]:
    """Get all digital behaviors mapped to a symptom."""
    behaviors = []
    for mapping in DSM5_BEHAVIOR_MAPPINGS:
        if mapping.symptom == symptom:
            behaviors.extend(mapping.digital_behaviors)
    return list(set(behaviors))


def get_symptoms_for_behavior(behavior: str) -> List[MDDSymptom]:
    """Get all symptoms a behavior is mapped to."""
    symptoms = []
    for mapping in DSM5_BEHAVIOR_MAPPINGS:
        if behavior in mapping.digital_behaviors:
            symptoms.append(mapping.symptom)
    return list(set(symptoms))


def get_mapping_evidence(symptom: MDDSymptom, behavior: str) -> Optional[EvidenceLevel]:
    """Get evidence level for a specific symptom-behavior mapping."""
    for mapping in DSM5_BEHAVIOR_MAPPINGS:
        if mapping.symptom == symptom and behavior in mapping.digital_behaviors:
            return mapping.evidence_level
    return None


def get_all_mappings_as_edges() -> List[Dict]:
    """
    Convert all mappings to graph edges format.

    Returns list of dicts with:
    - source: behavior name
    - target: symptom name
    - relation: 'reflects'
    - evidence: evidence level
    - references: list of citations
    """
    edges = []
    for mapping in DSM5_BEHAVIOR_MAPPINGS:
        for behavior in mapping.digital_behaviors:
            edges.append(
                {
                    "source": behavior,
                    "target": mapping.symptom.value,
                    "source_type": "behavior",
                    "target_type": "symptom",
                    "relation": "reflects",
                    "evidence": mapping.evidence_level.value,
                    "mechanism": mapping.mechanism,
                    "references": mapping.references,
                }
            )
    return edges
