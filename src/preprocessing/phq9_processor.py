"""
PHQ-9 Data Preprocessing Module

This module handles loading, encoding, and processing of PHQ-9 survey data
for suicide risk prediction research.

Author: Digital Phenotyping Research
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class PHQ9Processor:
    """
    Processor for PHQ-9 (Patient Health Questionnaire-9) survey data.

    Handles encoding of ordinal responses and creation of outcome variables
    for suicidal ideation prediction.
    """

    # Ordinal encoding mapping for PHQ-9 responses
    ENCODING_MAP = {
        'Not at all': 0,
        'Several days': 1,
        'More than half the days': 2,
        'Nearly every day': 3
    }

    # PHQ-9 item column names (items 1-9)
    ITEM_COLUMNS = [
        'Little interest or pleasure in doing things',
        'Feeling down, depressed, hopeless.',
        'Trouble falling or staying asleep, or sleeping too much.',
        'Feeling tired or having little energy',
        'Poor appetite or overeating',
        'Feeling bad about yourself or that you are a failure or have let yourself or your family down',
        'Trouble concentrating on things, such as reading the newspaper or watching television',
        'Moving or speaking so slowly that other people could have noticed. Or the opposite being so figety or restless that you have been moving around a lot more than usual',
        'Thoughts that you would be better off dead, or of hurting yourself'
    ]

    # Item #9 specifically (suicidal ideation)
    ITEM9_COLUMN = 'Thoughts that you would be better off dead, or of hurting yourself'

    def __init__(self, data_path: str):
        """
        Initialize PHQ9Processor.

        Args:
            data_path: Path to PHQ-9.csv file
        """
        self.data_path = Path(data_path)
        self.phq9_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load PHQ-9 survey data from CSV.

        Returns:
            DataFrame with PHQ-9 responses
        """
        self.phq9_data = pd.read_csv(self.data_path)
        print(f"Loaded PHQ-9 data: {len(self.phq9_data)} responses")
        print(f"  - Pre-assessment: {(self.phq9_data['type'] == 'pre').sum()}")
        print(f"  - Post-assessment: {(self.phq9_data['type'] == 'post').sum()}")
        return self.phq9_data

    def encode_responses(self) -> pd.DataFrame:
        """
        Encode ordinal PHQ-9 responses to numeric values (0-3).

        Returns:
            DataFrame with encoded responses
        """
        if self.phq9_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.phq9_data.copy()

        # Encode each PHQ-9 item
        for col in self.ITEM_COLUMNS:
            encoded_col = f"{col}_encoded"
            df[encoded_col] = df[col].map(self.ENCODING_MAP)

        return df

    def create_item9_labels(self, assessment_type: str = 'pre') -> pd.DataFrame:
        """
        Create binary outcome labels for PHQ-9 Item #9 (suicidal ideation).

        Args:
            assessment_type: 'pre' or 'post' assessment

        Returns:
            DataFrame with user IDs and Item #9 labels
        """
        df = self.encode_responses()

        # Filter by assessment type
        df_subset = df[df['type'] == assessment_type].copy()

        # Encode Item #9
        item9_encoded_col = f"{self.ITEM9_COLUMN}_encoded"
        df_subset['item9_score'] = df_subset[self.ITEM9_COLUMN].map(self.ENCODING_MAP)

        # Binary classification: 0 = "Not at all", 1 = Any ideation
        df_subset['item9_binary'] = (df_subset['item9_score'] > 0).astype(int)

        # Create clean label DataFrame
        labels = df_subset[['uid', 'item9_score', 'item9_binary']].copy()

        # Print class distribution
        n_total = len(labels)
        n_positive = labels['item9_binary'].sum()
        n_negative = n_total - n_positive

        print(f"\n{assessment_type.upper()}-ASSESSMENT Item #9 Distribution:")
        print(f"  Total: {n_total}")
        print(f"  Class 0 (No ideation): {n_negative} ({n_negative/n_total*100:.1f}%)")
        print(f"  Class 1 (Any ideation): {n_positive} ({n_positive/n_total*100:.1f}%)")
        print(f"  Imbalance ratio: {n_negative/n_positive:.2f}:1")

        # Show users with suicidal ideation
        positive_users = labels[labels['item9_binary'] == 1]['uid'].tolist()
        print(f"  Users with ideation: {positive_users}")

        return labels

    def create_phq9_total_labels(self, assessment_type: str = 'pre',
                                 threshold: int = 10) -> pd.DataFrame:
        """
        Create labels based on PHQ-9 total score.

        Args:
            assessment_type: 'pre' or 'post' assessment
            threshold: Cutoff for clinical depression (default: 10)

        Returns:
            DataFrame with user IDs and PHQ-9 total score labels
        """
        df = self.encode_responses()

        # Filter by assessment type
        df_subset = df[df['type'] == assessment_type].copy()

        # Calculate total score (sum of 9 items)
        encoded_cols = [f"{col}_encoded" for col in self.ITEM_COLUMNS]
        df_subset['phq9_total'] = df_subset[encoded_cols].sum(axis=1)

        # Binary classification based on threshold
        df_subset['clinical_depression'] = (df_subset['phq9_total'] >= threshold).astype(int)

        # Categorical severity (Kroenke et al., 2001)
        def categorize_severity(score):
            if score < 5:
                return 0  # Minimal
            elif score < 10:
                return 1  # Mild
            elif score < 15:
                return 2  # Moderate
            elif score < 20:
                return 3  # Moderately severe
            else:
                return 4  # Severe

        df_subset['depression_severity'] = df_subset['phq9_total'].apply(categorize_severity)

        # Create clean label DataFrame
        labels = df_subset[['uid', 'phq9_total', 'clinical_depression', 'depression_severity']].copy()

        # Print statistics
        print(f"\n{assessment_type.upper()}-ASSESSMENT PHQ-9 Total Score Distribution:")
        print(f"  Mean: {labels['phq9_total'].mean():.2f} ± {labels['phq9_total'].std():.2f}")
        print(f"  Median: {labels['phq9_total'].median():.0f}")
        print(f"  Range: {labels['phq9_total'].min():.0f} - {labels['phq9_total'].max():.0f}")
        print(f"\n  Clinical depression (≥{threshold}): {labels['clinical_depression'].sum()} ({labels['clinical_depression'].mean()*100:.1f}%)")
        print(f"\n  Severity distribution:")
        for severity in range(5):
            count = (labels['depression_severity'] == severity).sum()
            severity_names = ['Minimal', 'Mild', 'Moderate', 'Moderately severe', 'Severe']
            print(f"    {severity_names[severity]}: {count} ({count/len(labels)*100:.1f}%)")

        return labels

    def get_raw_responses(self, assessment_type: str = 'pre') -> pd.DataFrame:
        """
        Get raw PHQ-9 responses for detailed analysis.

        Args:
            assessment_type: 'pre' or 'post' assessment

        Returns:
            DataFrame with user IDs and raw responses
        """
        if self.phq9_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df_subset = self.phq9_data[self.phq9_data['type'] == assessment_type].copy()
        return df_subset

    def save_labels(self, output_dir: str, assessment_type: str = 'pre'):
        """
        Save processed labels to CSV files.

        Args:
            output_dir: Directory to save label files
            assessment_type: 'pre' or 'post' assessment
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save Item #9 labels
        item9_labels = self.create_item9_labels(assessment_type)
        item9_path = output_path / f"item9_labels_{assessment_type}.csv"
        item9_labels.to_csv(item9_path, index=False)
        print(f"\nSaved Item #9 labels to: {item9_path}")

        # Save PHQ-9 total score labels
        phq9_labels = self.create_phq9_total_labels(assessment_type)
        phq9_path = output_path / f"phq9_labels_{assessment_type}.csv"
        phq9_labels.to_csv(phq9_path, index=False)
        print(f"Saved PHQ-9 total labels to: {phq9_path}")

        return item9_path, phq9_path


def main():
    """
    Main function for standalone execution.
    """
    # Example usage
    data_path = "data/raw/dataset/survey/PHQ-9.csv"
    output_dir = "data/processed/labels"

    processor = PHQ9Processor(data_path)
    processor.load_data()

    # Process pre-assessment labels (primary)
    processor.save_labels(output_dir, assessment_type='pre')

    # Process post-assessment labels (secondary)
    print("\n" + "="*60)
    processor.save_labels(output_dir, assessment_type='post')


if __name__ == "__main__":
    main()
