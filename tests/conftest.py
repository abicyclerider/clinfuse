"""
Shared pytest fixtures for SyntheticMass testing.

This module provides reusable test fixtures for configuration loading,
sample patient data, and temporary directories.
"""

import os
import tempfile
from pathlib import Path
import pytest
import yaml
import pandas as pd


@pytest.fixture
def sample_config():
    """
    Provides a minimal test configuration for unit tests.

    Returns a dictionary with reduced scale for fast testing:
    - 10 base patients instead of 500
    - Lower percentages for special cases
    - Same error distribution structure as production
    """
    return {
        'population': {
            'base_size': 10
        },
        'special_cases': {
            'twins': {'enabled': True, 'percentage': 0.20},
            'jr_sr': {'enabled': True, 'percentage': 0.20, 'age_gap_min': 25, 'age_gap_max': 35},
            'common_names': {'enabled': True, 'percentage': 0.20},
            'overlays': {'enabled': True, 'percentage': 0.10}
        },
        'duplicates': {
            'no_duplicate': 0.25,
            'single_duplicate': 0.50,
            'multiple_duplicate': 0.20,
            'hard_case': 0.05,
            'multiple_duplicate_range': {'min': 2, 'max': 3},
            'hard_case_range': {'min': 4, 'max': 6}
        },
        'error_distribution': {
            'middle_name_discrepancy': 0.583,
            'ssn_error': 0.535,
            'first_name_typo': 0.531,
            'last_name_typo': 0.336,
            'address_variation': 0.35,
            'date_variation': 0.25,
            'missing_field': 0.45
        },
        'error_injection': {
            'typo_types': {
                'substitution': 0.60,
                'deletion': 0.20,
                'insertion': 0.10,
                'transposition': 0.10
            },
            'address_types': {
                'abbreviation': 0.40,
                'formatting': 0.30,
                'unit_variation': 0.15,
                'typo': 0.15
            },
            'date_types': {
                'format_change': 0.50,
                'day_month_swap': 0.25,
                'transposition': 0.15,
                'ocr_error': 0.10
            }
        },
        'keyboard_proximity': {
            'a': ['q', 'w', 's', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'e': ['w', 'r', 'd', 's'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l', 'k'],
            'p': ['o', 'l'],
            'q': ['w', 'a'],
            'r': ['e', 't', 'f', 'd'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'y', 'g', 'f'],
            'u': ['y', 'i', 'j', 'h'],
            'v': ['c', 'f', 'g', 'b'],
            'w': ['q', 'e', 's', 'a'],
            'x': ['z', 's', 'd', 'c'],
            'y': ['t', 'u', 'h', 'g'],
            'z': ['a', 's', 'x']
        },
        'random_seed': 42
    }


@pytest.fixture
def sample_patients():
    """
    Provides a small DataFrame of realistic test patient data.

    Returns a pandas DataFrame with 20 patients containing:
    - Patient ID, first/middle/last names
    - Date of birth, SSN
    - Address fields (line1, line2, city, state, zip)
    - Phone number

    Suitable for testing error injection and duplicate generation.
    """
    return pd.DataFrame([
        {
            'patient_id': 'P001',
            'first_name': 'John',
            'middle_name': 'Michael',
            'last_name': 'Smith',
            'dob': '1985-03-15',
            'ssn': '123-45-6789',
            'address_line1': '123 Main Street',
            'address_line2': 'Apt 4B',
            'city': 'Boston',
            'state': 'MA',
            'zip': '02101',
            'phone': '617-555-0101'
        },
        {
            'patient_id': 'P002',
            'first_name': 'Mary',
            'middle_name': 'Anne',
            'last_name': 'Johnson',
            'dob': '1990-07-22',
            'ssn': '234-56-7890',
            'address_line1': '456 Oak Avenue',
            'address_line2': '',
            'city': 'Cambridge',
            'state': 'MA',
            'zip': '02139',
            'phone': '617-555-0102'
        },
        {
            'patient_id': 'P003',
            'first_name': 'Robert',
            'middle_name': 'James',
            'last_name': 'Williams',
            'dob': '1978-11-30',
            'ssn': '345-67-8901',
            'address_line1': '789 Elm Street',
            'address_line2': 'Unit 12',
            'city': 'Somerville',
            'state': 'MA',
            'zip': '02143',
            'phone': '617-555-0103'
        },
        {
            'patient_id': 'P004',
            'first_name': 'Jennifer',
            'middle_name': 'Lynn',
            'last_name': 'Brown',
            'dob': '1982-05-18',
            'ssn': '456-78-9012',
            'address_line1': '321 Pine Road',
            'address_line2': '',
            'city': 'Brookline',
            'state': 'MA',
            'zip': '02445',
            'phone': '617-555-0104'
        },
        {
            'patient_id': 'P005',
            'first_name': 'Michael',
            'middle_name': 'David',
            'last_name': 'Davis',
            'dob': '1995-09-08',
            'ssn': '567-89-0123',
            'address_line1': '654 Maple Drive',
            'address_line2': 'Apt 2A',
            'city': 'Newton',
            'state': 'MA',
            'zip': '02458',
            'phone': '617-555-0105'
        },
        {
            'patient_id': 'P006',
            'first_name': 'Patricia',
            'middle_name': 'Rose',
            'last_name': 'Garcia',
            'dob': '1988-12-25',
            'ssn': '678-90-1234',
            'address_line1': '147 Cedar Lane',
            'address_line2': '',
            'city': 'Quincy',
            'state': 'MA',
            'zip': '02169',
            'phone': '617-555-0106'
        },
        {
            'patient_id': 'P007',
            'first_name': 'James',
            'middle_name': 'Robert',
            'last_name': 'Martinez',
            'dob': '1992-04-14',
            'ssn': '789-01-2345',
            'address_line1': '258 Birch Street',
            'address_line2': 'Suite 5',
            'city': 'Medford',
            'state': 'MA',
            'zip': '02155',
            'phone': '617-555-0107'
        },
        {
            'patient_id': 'P008',
            'first_name': 'Linda',
            'middle_name': 'Marie',
            'last_name': 'Rodriguez',
            'dob': '1980-08-03',
            'ssn': '890-12-3456',
            'address_line1': '369 Spruce Avenue',
            'address_line2': '',
            'city': 'Waltham',
            'state': 'MA',
            'zip': '02451',
            'phone': '617-555-0108'
        },
        {
            'patient_id': 'P009',
            'first_name': 'William',
            'middle_name': 'Charles',
            'last_name': 'Miller',
            'dob': '1987-01-29',
            'ssn': '901-23-4567',
            'address_line1': '741 Willow Court',
            'address_line2': 'Apt 8C',
            'city': 'Arlington',
            'state': 'MA',
            'zip': '02474',
            'phone': '617-555-0109'
        },
        {
            'patient_id': 'P010',
            'first_name': 'Barbara',
            'middle_name': 'Jean',
            'last_name': 'Wilson',
            'dob': '1993-06-17',
            'ssn': '012-34-5678',
            'address_line1': '852 Poplar Place',
            'address_line2': '',
            'city': 'Malden',
            'state': 'MA',
            'zip': '02148',
            'phone': '617-555-0110'
        },
        {
            'patient_id': 'P011',
            'first_name': 'Richard',
            'middle_name': 'Allen',
            'last_name': 'Moore',
            'dob': '1975-10-12',
            'ssn': '111-22-3333',
            'address_line1': '963 Ash Boulevard',
            'address_line2': 'Unit 3',
            'city': 'Revere',
            'state': 'MA',
            'zip': '02151',
            'phone': '617-555-0111'
        },
        {
            'patient_id': 'P012',
            'first_name': 'Susan',
            'middle_name': 'Kay',
            'last_name': 'Taylor',
            'dob': '1991-02-28',
            'ssn': '222-33-4444',
            'address_line1': '147 Cherry Street',
            'address_line2': '',
            'city': 'Everett',
            'state': 'MA',
            'zip': '02149',
            'phone': '617-555-0112'
        },
        {
            'patient_id': 'P013',
            'first_name': 'Thomas',
            'middle_name': 'Edward',
            'last_name': 'Anderson',
            'dob': '1984-07-05',
            'ssn': '333-44-5555',
            'address_line1': '258 Walnut Avenue',
            'address_line2': 'Apt 7D',
            'city': 'Chelsea',
            'state': 'MA',
            'zip': '02150',
            'phone': '617-555-0113'
        },
        {
            'patient_id': 'P014',
            'first_name': 'Jessica',
            'middle_name': 'Lynn',
            'last_name': 'Thomas',
            'dob': '1989-11-19',
            'ssn': '444-55-6666',
            'address_line1': '369 Hickory Drive',
            'address_line2': '',
            'city': 'Watertown',
            'state': 'MA',
            'zip': '02472',
            'phone': '617-555-0114'
        },
        {
            'patient_id': 'P015',
            'first_name': 'Christopher',
            'middle_name': 'Paul',
            'last_name': 'Jackson',
            'dob': '1996-03-24',
            'ssn': '555-66-7777',
            'address_line1': '741 Sycamore Lane',
            'address_line2': 'Suite 2B',
            'city': 'Belmont',
            'state': 'MA',
            'zip': '02478',
            'phone': '617-555-0115'
        },
        {
            'patient_id': 'P016',
            'first_name': 'Sarah',
            'middle_name': 'Elizabeth',
            'last_name': 'White',
            'dob': '1981-09-11',
            'ssn': '666-77-8888',
            'address_line1': '852 Magnolia Court',
            'address_line2': '',
            'city': 'Winchester',
            'state': 'MA',
            'zip': '01890',
            'phone': '617-555-0116'
        },
        {
            'patient_id': 'P017',
            'first_name': 'Daniel',
            'middle_name': 'Joseph',
            'last_name': 'Harris',
            'dob': '1994-12-07',
            'ssn': '777-88-9999',
            'address_line1': '963 Dogwood Place',
            'address_line2': 'Apt 1A',
            'city': 'Lexington',
            'state': 'MA',
            'zip': '02420',
            'phone': '617-555-0117'
        },
        {
            'patient_id': 'P018',
            'first_name': 'Nancy',
            'middle_name': 'Grace',
            'last_name': 'Martin',
            'dob': '1986-05-02',
            'ssn': '888-99-0000',
            'address_line1': '147 Redwood Street',
            'address_line2': '',
            'city': 'Woburn',
            'state': 'MA',
            'zip': '01801',
            'phone': '617-555-0118'
        },
        {
            'patient_id': 'P019',
            'first_name': 'Matthew',
            'middle_name': 'William',
            'last_name': 'Thompson',
            'dob': '1979-08-20',
            'ssn': '999-00-1111',
            'address_line1': '258 Cypress Avenue',
            'address_line2': 'Unit 6',
            'city': 'Burlington',
            'state': 'MA',
            'zip': '01803',
            'phone': '617-555-0119'
        },
        {
            'patient_id': 'P020',
            'first_name': 'Betty',
            'middle_name': 'Jane',
            'last_name': 'Lee',
            'dob': '1997-01-15',
            'ssn': '000-11-2222',
            'address_line1': '369 Sequoia Drive',
            'address_line2': '',
            'city': 'Bedford',
            'state': 'MA',
            'zip': '01730',
            'phone': '617-555-0120'
        }
    ])


@pytest.fixture
def temp_output_dir():
    """
    Provides a temporary directory for test outputs.

    Yields a Path object pointing to a temporary directory.
    The directory is automatically cleaned up after the test completes.

    Use this for tests that generate CSV files or other outputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def project_config():
    """
    Loads the actual project configuration from config/scale_config.yaml.

    Returns the full production configuration dictionary.
    Use this for integration tests that need real config values.
    """
    config_path = Path(__file__).parent.parent / 'config' / 'scale_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
