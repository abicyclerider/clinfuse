"""
Jr/Sr Generator - Creates father/son pairs with same name but 25-35 year age gap.

This module generates Jr/Sr (Junior/Senior) pairs for testing entity resolution algorithms.
These pairs share:
- Identical full names (except suffix Jr/Sr)
- 25-35 year age gap (father and son)
- Similar addresses (often same household or nearby)
- Different SSNs and IDs

The percentage of patients to convert to Jr/Sr pairs is configurable via config.
"""

import random
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd


class JrSrGenerator:
    """Generates Jr/Sr father/son pairs from base patient population."""

    def __init__(self, config: Dict):
        """
        Initialize Jr/Sr generator with configuration.

        Args:
            config: Configuration dictionary with 'special_cases.jr_sr' section
        """
        self.config = config
        self.jr_sr_config = config.get('special_cases', {}).get('jr_sr', {})
        self.enabled = self.jr_sr_config.get('enabled', True)
        self.percentage = self.jr_sr_config.get('percentage', 0.10)
        self.age_gap_min = self.jr_sr_config.get('age_gap_min', 25)
        self.age_gap_max = self.jr_sr_config.get('age_gap_max', 35)

        # Set random seed if provided for reproducibility
        if 'random_seed' in config:
            random.seed(config['random_seed'])

    def generate_jr_sr_pairs(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Jr/Sr pairs from patient population.

        Args:
            patients_df: DataFrame of patients from Synthea

        Returns:
            DataFrame with Jr/Sr pairs added and case_type column updated
        """
        if not self.enabled:
            return patients_df

        # Add case_type column if it doesn't exist
        if 'case_type' not in patients_df.columns:
            patients_df['case_type'] = 'standard'

        # Calculate number of patients to convert to Jr/Sr pairs
        # We need pairs, so divide by 2
        n_jr_sr_pairs = int(len(patients_df) * self.percentage / 2)

        if n_jr_sr_pairs == 0:
            return patients_df

        # Select male patients only (Jr/Sr typically father/son)
        male_patients = patients_df[patients_df['GENDER'] == 'M'].copy()

        # Filter for living patients with appropriate age for being a father
        # Convert BIRTHDATE to datetime if it's not already
        male_patients['BIRTHDATE'] = pd.to_datetime(male_patients['BIRTHDATE'])
        current_year = datetime.now().year
        male_patients['age'] = current_year - male_patients['BIRTHDATE'].dt.year

        # Father should be old enough to have adult children (at least 25 + age_gap_min years old)
        suitable_fathers = male_patients[
            (male_patients['age'] >= self.age_gap_min + 25) &
            (male_patients['DEATHDATE'].isna())
        ].copy()

        if len(suitable_fathers) < n_jr_sr_pairs:
            n_jr_sr_pairs = len(suitable_fathers)

        if n_jr_sr_pairs == 0:
            return patients_df

        # Randomly select patients to become Sr (father)
        selected_fathers = suitable_fathers.sample(n=n_jr_sr_pairs, random_state=self.config.get('random_seed'))

        jr_sr_pairs = []

        for idx, father in selected_fathers.iterrows():
            # Create Jr/Sr pair
            senior, junior = self._create_jr_sr_pair(father)
            jr_sr_pairs.extend([senior, junior])

            # Mark original patient as jr_sr
            patients_df.loc[idx, 'case_type'] = 'jr_sr'

        # Create DataFrame with Jr/Sr pairs
        jr_sr_df = pd.DataFrame(jr_sr_pairs)

        # Remove original selected patients and add Jr/Sr pairs
        patients_df = patients_df.drop(selected_fathers.index)
        patients_df = pd.concat([patients_df, jr_sr_df], ignore_index=True)

        return patients_df

    def _create_jr_sr_pair(self, father: pd.Series) -> Tuple[Dict, Dict]:
        """
        Create a Jr/Sr pair from a single patient (father becomes Sr, son is Jr).

        Args:
            father: Patient record as pandas Series (will become Sr)

        Returns:
            Tuple of two dictionaries representing father (Sr) and son (Jr)
        """
        # Calculate age gap (random between min and max)
        age_gap_years = random.randint(self.age_gap_min, self.age_gap_max)

        # Create Senior (father - modify original patient)
        senior = father.to_dict()
        senior['SUFFIX'] = 'Sr.'
        senior['case_type'] = 'jr_sr'
        senior['jr_sr_pair_id'] = father['Id']  # Use original ID as pair identifier

        # Create Junior (son - born age_gap years later)
        junior = father.to_dict()
        junior['Id'] = f"{father['Id']}_jr"  # Generate related ID

        # Calculate son's birthdate (age_gap years after father)
        father_birthdate = pd.to_datetime(father['BIRTHDATE'])
        son_birthdate = father_birthdate + timedelta(days=365.25 * age_gap_years)
        junior['BIRTHDATE'] = son_birthdate.strftime('%Y-%m-%d')

        junior['SUFFIX'] = 'Jr.'
        junior['SSN'] = self._generate_different_ssn(father['SSN'])
        junior['DRIVERS'] = self._generate_different_license(father.get('DRIVERS', ''))
        junior['PASSPORT'] = self._generate_different_passport(father.get('PASSPORT', ''))

        # Sons often live at same address or nearby
        # 70% chance same address, 30% chance different street number
        if random.random() < 0.7:
            # Same address
            pass  # Keep father's address
        else:
            # Different street number (nearby)
            junior['ADDRESS'] = self._generate_nearby_address(father['ADDRESS'])

        junior['case_type'] = 'jr_sr'
        junior['jr_sr_pair_id'] = father['Id']  # Same pair identifier

        # Clear death date for son (son should be alive if father is alive)
        junior['DEATHDATE'] = None

        return senior, junior

    def _generate_different_ssn(self, original_ssn: str) -> str:
        """Generate different SSN (different last 4 digits)."""
        if not original_ssn or pd.isna(original_ssn):
            return original_ssn

        # Parse SSN format: ###-##-####
        parts = original_ssn.split('-')
        if len(parts) != 3:
            return original_ssn

        # Generate different last 4 digits
        new_last_group = (int(parts[2]) + random.randint(1000, 5000)) % 10000

        return f"{parts[0]}-{parts[1]}-{new_last_group:04d}"

    def _generate_different_license(self, original_license: str) -> str:
        """Generate different driver's license."""
        if not original_license or pd.isna(original_license):
            return original_license

        # Change middle digits
        if len(original_license) > 4:
            middle = random.randint(10, 99)
            return original_license[:3] + str(middle) + original_license[5:]
        return original_license

    def _generate_different_passport(self, original_passport: str) -> str:
        """Generate different passport number."""
        if not original_passport or pd.isna(original_passport):
            return original_passport

        # Change some middle characters
        if len(original_passport) > 4:
            passport_list = list(original_passport)
            passport_list[2] = str(random.randint(0, 9))
            passport_list[3] = str(random.randint(0, 9))
            return ''.join(passport_list)
        return original_passport

    def _generate_nearby_address(self, original_address: str) -> str:
        """
        Generate nearby address (different street number).

        Args:
            original_address: Original street address

        Returns:
            Modified address with different street number
        """
        if not original_address or pd.isna(original_address):
            return original_address

        # Try to extract and modify street number
        parts = original_address.split()
        if len(parts) > 0 and parts[0].isdigit():
            # Change street number (add random offset)
            original_number = int(parts[0])
            offset = random.randint(-50, 50)
            new_number = max(1, original_number + offset)
            return f"{new_number} {' '.join(parts[1:])}"

        return original_address
