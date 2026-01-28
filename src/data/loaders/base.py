"""
Base class for dataset loaders.

Provides a standardized interface for loading multimodal mental health datasets.
New datasets (e.g., GLOBEM) should inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl


@dataclass
class DatasetInfo:
    """Metadata about the loaded dataset."""

    name: str
    n_participants: int
    n_days: int
    time_range: Tuple[str, str]
    available_surveys: List[str]
    available_sensors: List[str]
    missing_rate: Dict[str, float]


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.

    Provides a standardized interface for loading:
    - Survey data (PHQ-9, Big Five, etc.)
    - Sensor data (GPS, activity, phone usage, etc.)
    - EMA data (ecological momentary assessment)
    - Auxiliary data (grades, communication logs, etc.)

    Subclasses must implement all abstract methods.
    """

    def __init__(self, data_dir: Path, config: Optional[Dict] = None):
        """
        Initialize the loader.

        Args:
            data_dir: Path to the raw dataset directory
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self._validate_data_dir()

    def _validate_data_dir(self) -> None:
        """Verify the data directory exists and has expected structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    @abstractmethod
    def load_surveys(self) -> Dict[str, pl.DataFrame]:
        """
        Load all survey/questionnaire data.

        Returns:
            Dictionary mapping survey name to DataFrame.
            Each DataFrame should have columns:
            - user_id: str
            - survey_type: str (e.g., 'pre', 'post')
            - item columns (Q1, Q2, ...) or score columns
            - total_score: float (if applicable)
        """
        pass

    @abstractmethod
    def load_sensors(self) -> Dict[str, pl.DataFrame]:
        """
        Load all sensor data.

        Returns:
            Dictionary mapping sensor name to DataFrame.
            Each DataFrame should have columns:
            - user_id: str
            - timestamp: datetime
            - sensor-specific columns
        """
        pass

    @abstractmethod
    def load_ema(self) -> Dict[str, pl.DataFrame]:
        """
        Load EMA (Ecological Momentary Assessment) data.

        Returns:
            Dictionary mapping EMA category to DataFrame.
            Each DataFrame should have columns:
            - user_id: str
            - timestamp: datetime
            - response columns
        """
        pass

    @abstractmethod
    def get_user_ids(self) -> List[str]:
        """
        Get list of all participant user IDs.

        Returns:
            List of user ID strings (e.g., ['u00', 'u01', ...])
        """
        pass

    @abstractmethod
    def get_time_range(self) -> Tuple[str, str]:
        """
        Get the time range of the study.

        Returns:
            Tuple of (start_date, end_date) as ISO format strings
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> DatasetInfo:
        """
        Get metadata about the dataset.

        Returns:
            DatasetInfo object with dataset statistics
        """
        pass

    def load_all(self) -> Dict[str, Dict[str, pl.DataFrame]]:
        """
        Load all available data.

        Returns:
            Nested dictionary with structure:
            {
                'surveys': {...},
                'sensors': {...},
                'ema': {...}
            }
        """
        return {
            "surveys": self.load_surveys(),
            "sensors": self.load_sensors(),
            "ema": self.load_ema(),
        }
