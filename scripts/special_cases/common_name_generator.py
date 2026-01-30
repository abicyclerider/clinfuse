"""
Common Name Generator - Replaces names with high-frequency names to increase collision probability.

This module replaces patient names with very common names (like "John Smith", "Maria Garcia")
to create realistic scenarios where multiple different people share the same name.
This tests the entity resolution algorithm's ability to distinguish between different people
with identical names using other demographic data.

The percentage of patients to receive common names is configurable via config.
"""

import random
from typing import Dict, List
import pandas as pd


class CommonNameGenerator:
    """Generates patients with high-collision common names."""

    # Most common US names by gender (creates high collision probability)
    COMMON_FIRST_NAMES = {
        'M': [
            'James', 'John', 'Robert', 'Michael', 'William',
            'David', 'Richard', 'Joseph', 'Thomas', 'Christopher',
            'Charles', 'Daniel', 'Matthew', 'Mark', 'Donald',
            'Paul', 'Steven', 'Andrew', 'Kenneth', 'Joshua'
        ],
        'F': [
            'Mary', 'Patricia', 'Jennifer', 'Linda', 'Barbara',
            'Elizabeth', 'Susan', 'Jessica', 'Sarah', 'Karen',
            'Lisa', 'Nancy', 'Betty', 'Margaret', 'Sandra',
            'Ashley', 'Kimberly', 'Emily', 'Donna', 'Michelle'
        ]
    }

    # Most common US last names
    COMMON_LAST_NAMES = [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
        'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
        'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
        'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
        'Lee', 'Thompson', 'White', 'Harris', 'Sanchez',
        'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker'
    ]

    def __init__(self, config: Dict):
        """
        Initialize common name generator with configuration.

        Args:
            config: Configuration dictionary with 'special_cases.common_names' section
        """
        self.config = config
        self.common_names_config = config.get('special_cases', {}).get('common_names', {})
        self.enabled = self.common_names_config.get('enabled', True)
        self.percentage = self.common_names_config.get('percentage', 0.15)

        # Set random seed if provided for reproducibility
        if 'random_seed' in config:
            random.seed(config['random_seed'])

    def apply_common_names(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace names of selected patients with common names.

        Args:
            patients_df: DataFrame of patients

        Returns:
            DataFrame with common names applied to selected patients
        """
        if not self.enabled:
            return patients_df

        # Add case_type column if it doesn't exist
        if 'case_type' not in patients_df.columns:
            patients_df['case_type'] = 'standard'

        # Calculate number of patients to receive common names
        n_common_names = int(len(patients_df) * self.percentage)

        if n_common_names == 0:
            return patients_df

        # Randomly select patients to receive common names
        selected_indices = patients_df.sample(n=n_common_names, random_state=self.config.get('random_seed')).index

        # Apply common names to selected patients
        for idx in selected_indices:
            gender = patients_df.loc[idx, 'GENDER']
            current_case_type = patients_df.loc[idx, 'case_type']

            # Update names
            patients_df.loc[idx, 'FIRST'] = random.choice(self.COMMON_FIRST_NAMES.get(gender, self.COMMON_FIRST_NAMES['M']))
            patients_df.loc[idx, 'LAST'] = random.choice(self.COMMON_LAST_NAMES)

            # Update case type (may already be twin or jr_sr, so append)
            if current_case_type == 'standard':
                patients_df.loc[idx, 'case_type'] = 'common_name'
            else:
                # Patient is already a special case (twin or jr_sr), append common_name
                patients_df.loc[idx, 'case_type'] = f"{current_case_type},common_name"

        return patients_df
