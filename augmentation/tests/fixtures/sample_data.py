"""Sample test data fixtures."""

from datetime import datetime, timedelta

import pandas as pd


def create_sample_patients(num_patients: int = 10) -> pd.DataFrame:
    """Create sample patient data."""
    patients = []
    base_date = datetime(1980, 1, 1)

    for i in range(num_patients):
        patients.append(
            {
                "Id": f"patient-{i:03d}",
                "BIRTHDATE": base_date + timedelta(days=i * 365),
                "DEATHDATE": None,
                "SSN": f"999-{i:02d}-{i:04d}",
                "DRIVERS": f"S999{i:05d}",
                "PASSPORT": f"X{i:07d}",
                "PREFIX": "Mr." if i % 2 == 0 else "Ms.",
                "FIRST": f"FirstName{i}",
                "LAST": f"LastName{i}",
                "MAIDEN": f"Maiden{i}" if i % 2 == 1 else None,
                "MARITAL": "M" if i % 2 == 0 else "S",
                "RACE": "white",
                "ETHNICITY": "nonhispanic",
                "GENDER": "M" if i % 2 == 0 else "F",
                "BIRTHPLACE": "Boston MA",
                "ADDRESS": f"{100 + i} Main Street",
                "CITY": "Boston",
                "STATE": "MA",
                "ZIP": f"0211{i}",
                "LAT": 42.3601 + i * 0.01,
                "LON": -71.0589 + i * 0.01,
            }
        )

    return pd.DataFrame(patients)


def create_sample_encounters(
    patients_df: pd.DataFrame, encounters_per_patient: int = 10
) -> pd.DataFrame:
    """Create sample encounter data."""
    encounters = []
    base_date = datetime(2014, 1, 1)

    encounter_id = 0
    for _, patient in patients_df.iterrows():
        for j in range(encounters_per_patient):
            start_date = base_date + timedelta(days=encounter_id * 30)
            stop_date = start_date + timedelta(hours=2)

            encounters.append(
                {
                    "Id": f"encounter-{encounter_id:04d}",
                    "START": start_date,
                    "STOP": stop_date,
                    "PATIENT": patient["Id"],
                    "ORGANIZATION": "org-001",
                    "PROVIDER": "provider-001",
                    "PAYER": "payer-001",
                    "ENCOUNTERCLASS": "ambulatory",
                    "CODE": "185345009",
                    "DESCRIPTION": "Encounter for symptom",
                    "BASE_ENCOUNTER_COST": 125.0,
                    "TOTAL_CLAIM_COST": 125.0,
                    "PAYER_COVERAGE": 100.0,
                    "REASONCODE": None,
                    "REASONDESCRIPTION": None,
                }
            )

            encounter_id += 1

    return pd.DataFrame(encounters)


def create_sample_conditions(encounters_df: pd.DataFrame) -> pd.DataFrame:
    """Create sample condition data."""
    conditions = []

    for i, row in encounters_df.iterrows():
        if i % 3 == 0:  # Not all encounters have conditions
            conditions.append(
                {
                    "START": row["START"],
                    "STOP": row["STOP"],
                    "PATIENT": row["PATIENT"],
                    "ENCOUNTER": row["Id"],
                    "CODE": "44054006",
                    "DESCRIPTION": "Diabetes",
                }
            )

    return pd.DataFrame(conditions)


def create_sample_payer_transitions(patients_df: pd.DataFrame) -> pd.DataFrame:
    """Create sample payer transition data."""
    transitions = []
    base_date = datetime(2014, 1, 1)

    for _, patient in patients_df.iterrows():
        # Each patient has 2-3 payer transitions
        num_transitions = 2 + (int(patient["Id"].split("-")[1]) % 2)

        for j in range(num_transitions):
            start = base_date + timedelta(days=j * 1000)
            stop = start + timedelta(days=999) if j < num_transitions - 1 else None

            transitions.append(
                {
                    "PATIENT": patient["Id"],
                    "START_DATE": start,
                    "END_DATE": stop,
                    "PAYER": f"payer-{j + 1:03d}",
                    "SECONDARY_PAYER": None,
                    "OWNERSHIP": "Guardian",
                }
            )

    return pd.DataFrame(transitions)


def create_sample_organizations() -> pd.DataFrame:
    """Create sample organizations data."""
    orgs = []

    for i in range(5):
        orgs.append(
            {
                "Id": f"org-{i:03d}",
                "NAME": f"Hospital {i}",
                "ADDRESS": f"{200 + i * 10} Hospital Road",
                "CITY": "Boston",
                "STATE": "MA",
                "ZIP": f"0212{i}",
                "LAT": 42.3601 + i * 0.02,
                "LON": -71.0589 + i * 0.02,
            }
        )

    return pd.DataFrame(orgs)


def create_sample_synthea_csvs(
    num_patients: int = 10, encounters_per_patient: int = 10
) -> dict:
    """Create a complete set of sample Synthea CSVs."""
    patients_df = create_sample_patients(num_patients)
    encounters_df = create_sample_encounters(patients_df, encounters_per_patient)
    conditions_df = create_sample_conditions(encounters_df)
    payer_transitions_df = create_sample_payer_transitions(patients_df)
    organizations_df = create_sample_organizations()

    # Create minimal dataframes for other required tables
    empty_columns = {
        "medications.csv": [
            "START",
            "STOP",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
        ],
        "procedures.csv": [
            "START",
            "STOP",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
        ],
        "observations.csv": [
            "DATE",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
            "VALUE",
            "UNITS",
        ],
        "immunizations.csv": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
        "allergies.csv": [
            "START",
            "STOP",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
        ],
        "careplans.csv": [
            "START",
            "STOP",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
        ],
        "imaging_studies.csv": ["Id", "DATE", "PATIENT", "ENCOUNTER"],
        "devices.csv": ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
        "supplies.csv": ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
        "claims.csv": ["Id", "PATIENTID", "APPOINTMENTID"],
        "claims_transactions.csv": ["Id", "CLAIMID"],
        "providers.csv": ["Id", "ORGANIZATION", "NAME"],
        "payers.csv": ["Id", "NAME"],
    }

    csvs = {
        "patients.csv": patients_df,
        "encounters.csv": encounters_df,
        "conditions.csv": conditions_df,
        "payer_transitions.csv": payer_transitions_df,
        "organizations.csv": organizations_df,
    }

    # Add empty dataframes for other tables
    for filename, columns in empty_columns.items():
        csvs[filename] = pd.DataFrame(columns=columns)

    return csvs
