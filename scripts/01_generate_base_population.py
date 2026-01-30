#!/usr/bin/env python3
"""
Base Population Generator - Phase 2

This script:
1. Loads Synthea CSV output from output/synthea_raw/csv/patients.csv
2. Applies special case transformations (twins, Jr/Sr, common names)
3. Outputs base_population.csv with case_type metadata

Usage:
    python scripts/01_generate_base_population.py

Configuration:
    All parameters read from config/scale_config.yaml
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from special_cases.twin_generator import TwinGenerator
from special_cases.jr_sr_generator import JrSrGenerator
from special_cases.common_name_generator import CommonNameGenerator


def load_config(config_path: str = 'config/scale_config.yaml') -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_synthea_patients(synthea_csv_path: str) -> pd.DataFrame:
    """
    Load Synthea patients CSV.

    Args:
        synthea_csv_path: Path to Synthea patients.csv file

    Returns:
        DataFrame with patient records
    """
    print(f"Loading Synthea patients from: {synthea_csv_path}")
    patients_df = pd.read_csv(synthea_csv_path)
    print(f"Loaded {len(patients_df)} patients from Synthea")
    return patients_df


def apply_special_cases(patients_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply special case transformations to patient population.

    Order of operations:
    1. Generate twin pairs
    2. Generate Jr/Sr pairs
    3. Apply common names

    Note: Common names applied last so twins/Jr-Sr can also have common names.

    Args:
        patients_df: DataFrame of patients from Synthea
        config: Configuration dictionary

    Returns:
        DataFrame with special cases applied
    """
    print("\n" + "="*60)
    print("APPLYING SPECIAL CASE TRANSFORMATIONS")
    print("="*60)

    original_count = len(patients_df)

    # 1. Generate twin pairs
    print("\n1. Generating twin pairs...")
    twin_generator = TwinGenerator(config)
    if twin_generator.enabled:
        patients_df = twin_generator.generate_twin_pairs(patients_df)
        twin_count = len(patients_df[patients_df['case_type'].str.contains('twin', na=False)])
        print(f"   Created {twin_count} twin patients ({twin_count/len(patients_df)*100:.1f}%)")
    else:
        print("   Twin generation disabled in config")

    # 2. Generate Jr/Sr pairs
    print("\n2. Generating Jr/Sr pairs...")
    jr_sr_generator = JrSrGenerator(config)
    if jr_sr_generator.enabled:
        patients_df = jr_sr_generator.generate_jr_sr_pairs(patients_df)
        jr_sr_count = len(patients_df[patients_df['case_type'].str.contains('jr_sr', na=False)])
        print(f"   Created {jr_sr_count} Jr/Sr patients ({jr_sr_count/len(patients_df)*100:.1f}%)")
    else:
        print("   Jr/Sr generation disabled in config")

    # 3. Apply common names
    print("\n3. Applying common names...")
    common_name_generator = CommonNameGenerator(config)
    if common_name_generator.enabled:
        patients_df = common_name_generator.apply_common_names(patients_df)
        common_name_count = len(patients_df[patients_df['case_type'].str.contains('common_name', na=False)])
        print(f"   Applied common names to {common_name_count} patients ({common_name_count/len(patients_df)*100:.1f}%)")
    else:
        print("   Common names disabled in config")

    final_count = len(patients_df)
    print(f"\nPopulation size: {original_count} → {final_count} patients")

    return patients_df


def save_base_population(patients_df: pd.DataFrame, output_path: str):
    """
    Save base population with metadata.

    Args:
        patients_df: DataFrame with special cases applied
        output_path: Path to save base_population.csv
    """
    print(f"\nSaving base population to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    patients_df.to_csv(output_path, index=False)
    print(f"Saved {len(patients_df)} patients to {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("BASE POPULATION SUMMARY")
    print("="*60)
    print(f"Total patients: {len(patients_df)}")
    print(f"\nCase type distribution:")

    case_type_counts = patients_df['case_type'].value_counts()
    for case_type, count in case_type_counts.items():
        percentage = (count / len(patients_df)) * 100
        print(f"  {case_type:20s}: {count:4d} ({percentage:5.1f}%)")

    print(f"\nGender distribution:")
    gender_counts = patients_df['GENDER'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(patients_df)) * 100
        print(f"  {gender}: {count:4d} ({percentage:5.1f}%)")

    print(f"\nLiving vs Deceased:")
    living = patients_df['DEATHDATE'].isna().sum()
    deceased = len(patients_df) - living
    print(f"  Living:   {living:4d} ({living/len(patients_df)*100:5.1f}%)")
    print(f"  Deceased: {deceased:4d} ({deceased/len(patients_df)*100:5.1f}%)")


def main():
    """Main execution function."""
    print("="*60)
    print("PHASE 2: BASE POPULATION GENERATION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load configuration
    print("Loading configuration...")
    config = load_config('config/scale_config.yaml')
    print(f"Base population size: {config['population']['base_size']}")

    # Load Synthea patients
    synthea_csv_path = 'output/synthea_raw/csv/patients.csv'
    patients_df = load_synthea_patients(synthea_csv_path)

    # Apply special case transformations
    patients_df = apply_special_cases(patients_df, config)

    # Save base population
    output_path = 'output/base_population.csv'
    save_base_population(patients_df, output_path)

    print(f"\n{'='*60}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print("\nNext step: Run scripts/02_create_duplicates.py (Phase 3)")


if __name__ == '__main__':
    main()
