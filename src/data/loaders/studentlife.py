"""
StudentLife dataset loader.

Loads and preprocesses the Dartmouth StudentLife dataset:
- 51 participants over 10 weeks
- Smartphone sensing data (GPS, activity, phone usage, etc.)
- Surveys (PHQ-9, Big Five, PANAS, PSQI, etc.)
- EMA responses
- Academic data (grades, class schedules)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from loguru import logger

from .base import BaseDatasetLoader, DatasetInfo


class StudentLifeLoader(BaseDatasetLoader):
    """
    Loader for the StudentLife dataset.

    Reference:
        Wang et al. "StudentLife: Assessing Mental Health, Academic Performance
        and Behavioral Trends of College Students using Smartphones" (UbiComp 2014)
    """

    # PHQ-9 response mapping
    PHQ9_SCALE = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3,
    }

    # Big Five response mapping
    BIGFIVE_SCALE = {
        "Disagree strongly": 1,
        "Disagree a little": 2,
        "Neither agree nor disagree": 3,
        "Agree a little": 4,
        "Agree strongly": 5,
    }

    def __init__(self, data_dir: Path, config: Optional[Dict] = None):
        """
        Initialize StudentLife loader.

        Args:
            data_dir: Path to data/raw/dataset directory
            config: Optional configuration from studentlife.yaml
        """
        super().__init__(data_dir, config)
        self._user_ids: Optional[List[str]] = None

    def _validate_data_dir(self) -> None:
        """Verify StudentLife directory structure."""
        super()._validate_data_dir()

        required_dirs = ["survey", "sensing", "EMA"]
        for d in required_dirs:
            if not (self.data_dir / d).exists():
                logger.warning(f"Expected directory not found: {d}")

    def get_user_ids(self) -> List[str]:
        """Get all user IDs from GPS directory (most complete)."""
        if self._user_ids is not None:
            return self._user_ids

        gps_dir = self.data_dir / "sensing" / "gps"
        if gps_dir.exists():
            self._user_ids = sorted(
                [f.stem.replace("gps_", "") for f in gps_dir.glob("gps_u*.csv")]
            )
        else:
            # Fallback: scan survey directory
            survey_dir = self.data_dir / "survey"
            if survey_dir.exists():
                phq9 = pl.read_csv(survey_dir / "PHQ-9.csv")
                self._user_ids = sorted(phq9["uid"].unique().to_list())
            else:
                self._user_ids = []

        logger.info(f"Found {len(self._user_ids)} users")
        return self._user_ids

    def get_time_range(self) -> Tuple[str, str]:
        """Return the study time range."""
        return ("2013-03-27", "2013-06-03")

    def load_surveys(self) -> Dict[str, pl.DataFrame]:
        """
        Load all survey data with proper scoring.

        Returns:
            Dictionary with keys: 'phq9', 'bigfive', 'panas', 'psqi', 'stress', etc.
        """
        survey_dir = self.data_dir / "survey"
        surveys = {}

        # PHQ-9
        phq9_path = survey_dir / "PHQ-9.csv"
        if phq9_path.exists():
            surveys["phq9"] = self._load_phq9(phq9_path)
            logger.info(f"Loaded PHQ-9: {len(surveys['phq9'])} records")

        # Big Five
        bf_path = survey_dir / "BigFive.csv"
        if bf_path.exists():
            surveys["bigfive"] = self._load_bigfive(bf_path)
            logger.info(f"Loaded BigFive: {len(surveys['bigfive'])} records")

        # PANAS
        panas_path = survey_dir / "panas.csv"
        if panas_path.exists():
            surveys["panas"] = self._load_panas(panas_path)
            logger.info(f"Loaded PANAS: {len(surveys['panas'])} records")

        # PSQI
        psqi_path = survey_dir / "psqi.csv"
        if psqi_path.exists():
            surveys["psqi"] = pl.read_csv(psqi_path)
            logger.info(f"Loaded PSQI: {len(surveys['psqi'])} records")

        # Stress
        stress_path = survey_dir / "PerceivedStressScale.csv"
        if stress_path.exists():
            surveys["stress"] = pl.read_csv(stress_path)
            logger.info(f"Loaded Stress: {len(surveys['stress'])} records")

        # Loneliness
        lonely_path = survey_dir / "LonelinessScale.csv"
        if lonely_path.exists():
            surveys["loneliness"] = pl.read_csv(lonely_path)
            logger.info(f"Loaded Loneliness: {len(surveys['loneliness'])} records")

        # Flourishing
        flour_path = survey_dir / "FlourishingScale.csv"
        if flour_path.exists():
            surveys["flourishing"] = pl.read_csv(flour_path)
            logger.info(f"Loaded Flourishing: {len(surveys['flourishing'])} records")

        return surveys

    # PHQ-9 question column names (actual format in StudentLife)
    PHQ9_QUESTIONS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, hopeless.",
        "Trouble falling or staying asleep, or sleeping too much.",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have noticed. Or the opposite being so figety or restless that you have been moving around a lot more than usual",
        "Thoughts that you would be better off dead, or of hurting yourself",
    ]

    def _load_phq9(self, path: Path) -> pl.DataFrame:
        """
        Load and score PHQ-9 data.

        PHQ-9 is a 9-item depression screening tool.
        Total score ranges from 0-27.
        Item 9 specifically asks about suicidal ideation.
        """
        df = pl.read_csv(path)

        # Find PHQ-9 question columns (may be full text or Q1-Q9)
        q_cols = []
        for i, q in enumerate(self.PHQ9_QUESTIONS):
            # Try exact match first
            if q in df.columns:
                q_cols.append(q)
            else:
                # Try Q1, Q2, etc. format
                q_format = f"Q{i+1}"
                if q_format in df.columns:
                    q_cols.append(q_format)

        if not q_cols:
            logger.warning("No PHQ-9 question columns found")
            return df

        logger.info(f"Found {len(q_cols)} PHQ-9 questions")

        # Convert responses to numeric scores
        score_cols = []
        for i, col in enumerate(q_cols):
            score_col = f"Q{i+1}_score"
            df = df.with_columns(
                pl.col(col)
                .replace(self.PHQ9_SCALE)
                .cast(pl.Int32)
                .alias(score_col)
            )
            score_cols.append(score_col)

        # Calculate total score
        if score_cols:
            df = df.with_columns(
                pl.sum_horizontal(score_cols).alias("total_score")
            )

        # Item 9 binary (suicidal ideation: any response > 0)
        if "Q9_score" in df.columns:
            df = df.with_columns(
                (pl.col("Q9_score") > 0).cast(pl.Int32).alias("item9_binary")
            )

        # Severity classification
        if "total_score" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("total_score") <= 4)
                .then(pl.lit("minimal"))
                .when(pl.col("total_score") <= 9)
                .then(pl.lit("mild"))
                .when(pl.col("total_score") <= 14)
                .then(pl.lit("moderate"))
                .when(pl.col("total_score") <= 19)
                .then(pl.lit("moderately_severe"))
                .otherwise(pl.lit("severe"))
                .alias("severity")
            )

        return df

    def _load_bigfive(self, path: Path) -> pl.DataFrame:
        """
        Load and score Big Five personality data.

        44 items measuring 5 personality dimensions:
        - Extraversion
        - Agreeableness
        - Conscientiousness
        - Neuroticism
        - Openness
        """
        df = pl.read_csv(path)

        # Get question columns
        q_cols = [c for c in df.columns if c.startswith("Q") and c[1:].isdigit()]

        # Convert responses to numeric
        for col in q_cols:
            df = df.with_columns(
                pl.col(col)
                .replace(self.BIGFIVE_SCALE)
                .cast(pl.Float32)
                .alias(f"{col}_score")
            )

        # Items that need reverse scoring (from BFI scoring guide)
        reverse_items = [2, 6, 8, 9, 12, 18, 21, 23, 24, 27, 31, 34, 35, 37, 41, 43]
        for item in reverse_items:
            col = f"Q{item}_score"
            if col in df.columns:
                df = df.with_columns((6 - pl.col(col)).alias(col))

        # Calculate dimension scores
        dimension_items = {
            "extraversion": [1, 6, 11, 16, 21, 26, 31, 36],
            "agreeableness": [2, 7, 12, 17, 22, 27, 32, 37, 42],
            "conscientiousness": [3, 8, 13, 18, 23, 28, 33, 38, 43],
            "neuroticism": [4, 9, 14, 19, 24, 29, 34, 39],
            "openness": [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
        }

        for dim, items in dimension_items.items():
            cols = [f"Q{i}_score" for i in items if f"Q{i}_score" in df.columns]
            if cols:
                df = df.with_columns(pl.mean_horizontal(cols).alias(f"{dim}_score"))

        return df

    def _load_panas(self, path: Path) -> pl.DataFrame:
        """
        Load PANAS (Positive and Negative Affect Schedule) data.

        20 items: 10 positive affect, 10 negative affect.
        Each rated 1-5.
        """
        df = pl.read_csv(path)

        positive_items = [
            "Interested",
            "Excited",
            "Strong",
            "Enthusiastic",
            "Proud",
            "Alert",
            "Inspired",
            "Determined",
            "Attentive",
            "Active",
        ]
        negative_items = [
            "Distressed",
            "Upset",
            "Guilty",
            "Scared",
            "Hostile",
            "Irritable",
            "Ashamed",
            "Nervous",
            "Jittery",
            "Afraid",
        ]

        # Calculate positive and negative affect scores
        pos_cols = [c for c in positive_items if c in df.columns]
        neg_cols = [c for c in negative_items if c in df.columns]

        if pos_cols:
            df = df.with_columns(pl.sum_horizontal(pos_cols).alias("positive_affect"))
        if neg_cols:
            df = df.with_columns(pl.sum_horizontal(neg_cols).alias("negative_affect"))

        return df

    def load_sensors(self) -> Dict[str, pl.DataFrame]:
        """
        Load all sensor data.

        Returns:
            Dictionary with keys: 'gps', 'activity', 'phonelock', etc.
        """
        sensing_dir = self.data_dir / "sensing"
        sensors = {}

        sensor_types = [
            "gps",
            "activity",
            "audio",
            "conversation",
            "phonelock",
            "phonecharge",
            "wifi",
            "bluetooth",
        ]

        for sensor_type in sensor_types:
            sensor_dir = sensing_dir / sensor_type
            if sensor_dir.exists():
                sensors[sensor_type] = self._load_sensor_type(sensor_dir, sensor_type)
                if sensors[sensor_type] is not None:
                    logger.info(
                        f"Loaded {sensor_type}: {len(sensors[sensor_type])} records"
                    )

        return sensors

    def _load_sensor_type(
        self, sensor_dir: Path, sensor_type: str
    ) -> Optional[pl.DataFrame]:
        """Load all files for a sensor type and concatenate."""
        all_data = []

        for filepath in sorted(sensor_dir.glob(f"{sensor_type}_u*.csv")):
            user_id = filepath.stem.replace(f"{sensor_type}_", "")
            try:
                df = pl.read_csv(
                    filepath,
                    infer_schema_length=10000,
                    truncate_ragged_lines=True,  # Handle irregular CSV rows
                )
                df = df.with_columns(pl.lit(user_id).alias("user_id"))
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        if not all_data:
            return None

        return pl.concat(all_data, how="diagonal")

    def load_ema(self) -> Dict[str, pl.DataFrame]:
        """
        Load EMA (Ecological Momentary Assessment) data.

        EMA includes frequent in-the-moment surveys about:
        - Mood
        - Sleep quality
        - Stress levels
        - Social interactions
        - Activities
        """
        ema_dir = self.data_dir / "EMA" / "response"
        ema_data = {}

        ema_categories = ["Mood", "Sleep", "Stress", "Social", "Activity"]

        for category in ema_categories:
            cat_dir = ema_dir / category
            if cat_dir.exists():
                ema_data[category.lower()] = self._load_ema_category(cat_dir)
                if ema_data[category.lower()] is not None:
                    logger.info(
                        f"Loaded EMA {category}: {len(ema_data[category.lower()])} records"
                    )

        return ema_data

    def _load_ema_category(self, category_dir: Path) -> Optional[pl.DataFrame]:
        """Load all EMA responses for a category."""
        all_data = []

        for filepath in sorted(category_dir.glob("*.json")):
            user_id = filepath.stem.replace("Mood_", "").replace("Sleep_", "")
            try:
                with open(filepath) as f:
                    records = json.load(f)
                if records:
                    df = pl.DataFrame(records)
                    df = df.with_columns(pl.lit(user_id).alias("user_id"))
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        if not all_data:
            return None

        return pl.concat(all_data, how="diagonal")

    def load_education(self) -> Dict[str, pl.DataFrame]:
        """
        Load education-related data.

        Includes:
        - grades: GPA and course grades
        - classes: Course enrollment
        - deadlines: Assignment due dates
        """
        edu_dir = self.data_dir / "education"
        education = {}

        # Grades
        grades_path = edu_dir / "grades.csv"
        if grades_path.exists():
            education["grades"] = pl.read_csv(grades_path)
            logger.info(f"Loaded grades: {len(education['grades'])} records")

        # Classes (variable number of courses per student)
        classes_path = edu_dir / "class.csv"
        if classes_path.exists():
            education["classes"] = pl.read_csv(
                classes_path, truncate_ragged_lines=True
            )

        # Class info (JSON)
        class_info_path = edu_dir / "class_info.json"
        if class_info_path.exists():
            with open(class_info_path) as f:
                education["class_info"] = json.load(f)

        return education

    def load_communication(self) -> Dict[str, pl.DataFrame]:
        """
        Load communication data (calls, SMS, app usage).

        Note: Contact information is hashed for privacy.
        """
        comm_data = {}

        # Call logs
        calls_dir = self.data_dir / "call_log"
        if calls_dir.exists():
            comm_data["calls"] = self._load_communication_type(calls_dir, "calls")

        # SMS
        sms_dir = self.data_dir / "sms"
        if sms_dir.exists():
            comm_data["sms"] = self._load_communication_type(sms_dir, "sms")

        # App usage
        app_dir = self.data_dir / "app_usage"
        if app_dir.exists():
            comm_data["app_usage"] = self._load_communication_type(app_dir, "app_usage")

        return comm_data

    def _load_communication_type(
        self, data_dir: Path, data_type: str
    ) -> Optional[pl.DataFrame]:
        """Load communication data files."""
        all_data = []

        for filepath in sorted(data_dir.glob("*.csv")):
            user_id = filepath.stem.replace(f"{data_type}_", "")
            try:
                df = pl.read_csv(filepath, infer_schema_length=10000)
                df = df.with_columns(pl.lit(user_id).alias("user_id"))
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        if not all_data:
            return None

        return pl.concat(all_data, how="diagonal")

    def get_dataset_info(self) -> DatasetInfo:
        """Get summary statistics about the dataset."""
        user_ids = self.get_user_ids()
        start, end = self.get_time_range()

        # Calculate days
        from datetime import datetime

        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        n_days = (end_dt - start_dt).days

        # Check available data
        survey_dir = self.data_dir / "survey"
        available_surveys = (
            [f.stem for f in survey_dir.glob("*.csv")] if survey_dir.exists() else []
        )

        sensing_dir = self.data_dir / "sensing"
        available_sensors = (
            [d.name for d in sensing_dir.iterdir() if d.is_dir()]
            if sensing_dir.exists()
            else []
        )

        return DatasetInfo(
            name="StudentLife",
            n_participants=len(user_ids),
            n_days=n_days,
            time_range=(start, end),
            available_surveys=available_surveys,
            available_sensors=available_sensors,
            missing_rate={},  # TODO: Calculate missing rates
        )
