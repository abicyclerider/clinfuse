"""
Twin Generator - Creates identical twin pairs with same DOB, similar names, same address.

This module generates twin pairs for testing entity resolution algorithms.
Twins share:
- Same date of birth (identical)
- Same last name (identical)
- Similar but not identical first names (e.g., Mary/Marie, John/Jon)
- Same address
- Sequential IDs to suggest relationship

The percentage of patients to convert to twins is configurable via config.
"""

import random
from typing import Dict, List, Tuple
import pandas as pd


class TwinGenerator:
    """Generates twin pairs from base patient population."""

    # Common twin name pairs (similar names that twins might have)
    TWIN_NAME_PAIRS = [
        # Female pairs
        ('Mary', 'Marie'), ('Anna', 'Anne'), ('Sarah', 'Sara'),
        ('Catherine', 'Katherine'), ('Elizabeth', 'Elisabeth'),
        ('Christina', 'Christine'), ('Michelle', 'Michele'),
        ('Rebecca', 'Rebekah'), ('Jessica', 'Jessie'), ('Jennifer', 'Jenny'),
        ('Nicole', 'Nicola'), ('Danielle', 'Daniela'), ('Lauren', 'Laurie'),
        # Male pairs
        ('John', 'Jon'), ('Michael', 'Mike'), ('Daniel', 'Danny'),
        ('Christopher', 'Chris'), ('Matthew', 'Matt'), ('William', 'Will'),
        ('Joseph', 'Joe'), ('Robert', 'Rob'), ('James', 'Jim'),
        ('David', 'Dave'), ('Richard', 'Rick'), ('Thomas', 'Tom'),
        # Gender neutral
        ('Alex', 'Alexis'), ('Jordan', 'Jorden'), ('Taylor', 'Tyler'),
    ]

    def __init__(self, config: Dict):
        """
        Initialize twin generator with configuration.

        Args:
            config: Configuration dictionary with 'special_cases.twins' section
        """
        self.config = config
        self.twin_config = config.get('special_cases', {}).get('twins', {})
        self.enabled = self.twin_config.get('enabled', True)
        self.percentage = self.twin_config.get('percentage', 0.10)

        # Set random seed if provided for reproducibility
        if 'random_seed' in config:
            random.seed(config['random_seed'])

    def generate_twin_pairs(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate twin pairs from patient population.

        Args:
            patients_df: DataFrame of patients from Synthea

        Returns:
            DataFrame with twin pairs added and case_type column updated
        """
        if not self.enabled:
            return patients_df

        # Add case_type column if it doesn't exist
        if 'case_type' not in patients_df.columns:
            patients_df['case_type'] = 'standard'

        # Calculate number of patients to convert to twins
        # We need pairs, so divide by 2
        n_twin_pairs = int(len(patients_df) * self.percentage / 2)

        if n_twin_pairs == 0:
            return patients_df

        # Select living patients only (twins should both be alive)
        living_patients = patients_df[patients_df['DEATHDATE'].isna()].copy()

        if len(living_patients) < n_twin_pairs * 2:
            n_twin_pairs = len(living_patients) // 2

        # Randomly select patients to become twins
        selected_patients = living_patients.sample(n=n_twin_pairs, random_state=self.config.get('random_seed'))

        twin_pairs = []

        for idx, patient in selected_patients.iterrows():
            # Create twin pair
            twin1, twin2 = self._create_twin_pair(patient)
            twin_pairs.extend([twin1, twin2])

            # Mark original patient as twin
            patients_df.loc[idx, 'case_type'] = 'twin'

        # Create DataFrame with twin pairs
        twins_df = pd.DataFrame(twin_pairs)

        # Remove original selected patients and add twin pairs
        patients_df = patients_df.drop(selected_patients.index)
        patients_df = pd.concat([patients_df, twins_df], ignore_index=True)

        return patients_df

    def _create_twin_pair(self, patient: pd.Series) -> Tuple[Dict, Dict]:
        """
        Create a twin pair from a single patient.

        Args:
            patient: Patient record as pandas Series

        Returns:
            Tuple of two dictionaries representing twin siblings
        """
        # Select twin name pair based on gender
        name_pairs = [pair for pair in self.TWIN_NAME_PAIRS]
        name1, name2 = random.choice(name_pairs)

        # Create first twin (modify original patient)
        twin1 = patient.to_dict()
        twin1['FIRST'] = name1
        twin1['case_type'] = 'twin'
        twin1['twin_pair_id'] = patient['Id']  # Use original ID as pair identifier

        # Create second twin (duplicate with similar name)
        twin2 = patient.to_dict()
        twin2['Id'] = f"{patient['Id']}_twin"  # Generate related ID
        twin2['FIRST'] = name2
        twin2['SSN'] = self._generate_sequential_ssn(patient['SSN'])
        twin2['DRIVERS'] = self._generate_sequential_license(patient.get('DRIVERS', ''))
        twin2['PASSPORT'] = self._generate_sequential_passport(patient.get('PASSPORT', ''))
        twin2['case_type'] = 'twin'
        twin2['twin_pair_id'] = patient['Id']  # Same pair identifier

        return twin1, twin2

    def _generate_sequential_ssn(self, original_ssn: str) -> str:
        """Generate sequential SSN (increment last digit)."""
        if not original_ssn or pd.isna(original_ssn):
            return original_ssn

        # Parse SSN format: ###-##-####
        parts = original_ssn.split('-')
        if len(parts) != 3:
            return original_ssn

        # Increment last digit
        last_group = int(parts[2])
        last_group = (last_group + 1) % 10000

        return f"{parts[0]}-{parts[1]}-{last_group:04d}"

    def _generate_sequential_license(self, original_license: str) -> str:
        """Generate sequential driver's license."""
        if not original_license or pd.isna(original_license):
            return original_license

        # Increment last digit if it's numeric, otherwise just change it
        if original_license and len(original_license) > 0:
            last_char = original_license[-1]
            if last_char.isdigit():
                return original_license[:-1] + str((int(last_char) + 1) % 10)
            else:
                # If last char is a letter, just increment it
                new_char = chr((ord(last_char) - ord('A') + 1) % 26 + ord('A')) if last_char.isupper() else chr((ord(last_char) - ord('a') + 1) % 26 + ord('a'))
                return original_license[:-1] + new_char
        return original_license

    def _generate_sequential_passport(self, original_passport: str) -> str:
        """Generate sequential passport number."""
        if not original_passport or pd.isna(original_passport):
            return original_passport

        # Increment last digit if it's numeric, otherwise just change it
        if original_passport and len(original_passport) > 0:
            last_char = original_passport[-1]
            if last_char.isdigit():
                return original_passport[:-1] + str((int(last_char) + 1) % 10)
            else:
                # If last char is a letter, just change it to another letter
                new_char = 'Y' if last_char == 'X' else 'X'
                return original_passport[:-1] + new_char
        return original_passport
