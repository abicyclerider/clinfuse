"""Data reading and writing utilities for Synthea CSV input and Parquet output."""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class DataHandler:
    """Reads Synthea CSV input and writes facility Parquet output."""

    # Standard CSV files in Synthea output
    SYNTHEA_CSV_FILES = [
        "patients.csv",
        "encounters.csv",
        "conditions.csv",
        "medications.csv",
        "observations.csv",
        "procedures.csv",
        "immunizations.csv",
        "allergies.csv",
        "careplans.csv",
        "imaging_studies.csv",
        "devices.csv",
        "supplies.csv",
        "claims.csv",
        "claims_transactions.csv",
        "payer_transitions.csv",
        "organizations.csv",
        "providers.csv",
        "payers.csv",
    ]

    # Reference tables (copied to all facilities unchanged)
    REFERENCE_TABLES = [
        "organizations.csv",
        "providers.csv",
        "payers.csv",
    ]

    # Tables with PATIENT foreign key
    PATIENT_LINKED_TABLES = [
        "encounters.csv",
        "payer_transitions.csv",
    ]

    # Tables with ENCOUNTER foreign key
    ENCOUNTER_LINKED_TABLES = [
        "conditions.csv",
        "medications.csv",
        "observations.csv",
        "procedures.csv",
        "immunizations.csv",
        "allergies.csv",
        "careplans.csv",
        "imaging_studies.csv",
        "devices.csv",
        "supplies.csv",
    ]

    @staticmethod
    def read_csv(
        file_path: Path,
        dtype: Optional[Dict] = None,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Read CSV file with standard settings.

        Args:
            file_path: Path to CSV file
            dtype: Optional dtype specifications
            parse_dates: Optional list of date columns to parse

        Returns:
            DataFrame with CSV contents
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        return pd.read_csv(
            file_path,
            dtype=dtype,
            parse_dates=parse_dates,
            low_memory=False,
        )

    @staticmethod
    def write_csv(df: pd.DataFrame, file_path: Path, create_dirs: bool = True) -> None:
        """
        Write DataFrame to CSV with standard settings.

        Args:
            df: DataFrame to write
            file_path: Output path
            create_dirs: Create parent directories if they don't exist
        """
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(file_path, index=False)

    @classmethod
    def load_synthea_csvs(cls, input_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all Synthea CSV files from directory.

        Args:
            input_dir: Directory containing Synthea CSV files

        Returns:
            Dictionary mapping filename to DataFrame
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        csvs = {}
        for filename in cls.SYNTHEA_CSV_FILES:
            file_path = input_dir / filename
            if file_path.exists():
                # Parse date columns for relevant files
                parse_dates = None
                if filename == "encounters.csv":
                    parse_dates = ["START", "STOP"]
                elif filename == "payer_transitions.csv":
                    parse_dates = ["START_DATE", "END_DATE"]
                elif filename == "patients.csv":
                    parse_dates = ["BIRTHDATE", "DEATHDATE"]

                csvs[filename] = cls.read_csv(file_path, parse_dates=parse_dates)

                # Strip Synthea numbers from name fields in patients.csv
                if filename == "patients.csv":
                    csvs[filename] = cls._strip_synthea_numbers(csvs[filename])
            else:
                raise FileNotFoundError(f"Required CSV file not found: {file_path}")

        return csvs

    @classmethod
    def load_single_csv(cls, input_dir: Path, filename: str) -> pd.DataFrame:
        """
        Load one Synthea CSV file with standard date parsing and name stripping.

        Args:
            input_dir: Directory containing Synthea CSV files
            filename: CSV filename (e.g. 'patients.csv')

        Returns:
            DataFrame with CSV contents
        """
        file_path = input_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {file_path}")

        parse_dates = None
        if filename == "encounters.csv":
            parse_dates = ["START", "STOP"]
        elif filename == "payer_transitions.csv":
            parse_dates = ["START_DATE", "END_DATE"]
        elif filename == "patients.csv":
            parse_dates = ["BIRTHDATE", "DEATHDATE"]

        df = cls.read_csv(file_path, parse_dates=parse_dates)

        if filename == "patients.csv":
            df = cls._strip_synthea_numbers(df)

        return df

    @classmethod
    def write_facility_table(
        cls,
        df: pd.DataFrame,
        output_dir: Path,
        facility_id: int,
        filename: str,
    ) -> None:
        """
        Write one Parquet table for one facility.

        Args:
            df: DataFrame to write
            output_dir: Base output directory (parent of facility_NNN dirs)
            facility_id: Facility identifier
            filename: Original CSV filename (e.g. 'patients.csv')
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        facility_dir.mkdir(parents=True, exist_ok=True)
        parquet_name = Path(filename).with_suffix(".parquet").name
        df.to_parquet(facility_dir / parquet_name, index=False)

    @classmethod
    def read_facility_table(
        cls,
        output_dir: Path,
        facility_id: int,
        filename: str,
    ) -> pd.DataFrame:
        """
        Read one Parquet file back for a facility.

        Args:
            output_dir: Base output directory (parent of facility_NNN dirs)
            facility_id: Facility identifier
            filename: Original CSV filename (e.g. 'encounters.csv')

        Returns:
            DataFrame with Parquet contents
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        parquet_name = Path(filename).with_suffix(".parquet").name
        parquet_path = facility_dir / parquet_name
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        return pd.read_parquet(parquet_path)

    @classmethod
    def read_all_facility_tables(
        cls,
        output_dir: Path,
        facility_id: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        Read all Parquet files for one facility into a dict.

        Args:
            output_dir: Base output directory (parent of facility_NNN dirs)
            facility_id: Facility identifier

        Returns:
            Dictionary mapping original CSV filename to DataFrame
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        result = {}
        for filename in cls.SYNTHEA_CSV_FILES:
            parquet_name = Path(filename).with_suffix(".parquet").name
            parquet_path = facility_dir / parquet_name
            if parquet_path.exists():
                result[filename] = pd.read_parquet(parquet_path)
        return result

    @classmethod
    def write_facility_data(
        cls,
        facility_csvs: Dict[str, pd.DataFrame],
        output_dir: Path,
        facility_id: int,
    ) -> None:
        """
        Write all data files for a facility as Parquet.

        Args:
            facility_csvs: Dictionary mapping filename (e.g. 'patients.csv') to DataFrame
            output_dir: Base output directory
            facility_id: Facility identifier
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        facility_dir.mkdir(parents=True, exist_ok=True)

        for filename, df in facility_csvs.items():
            # Convert .csv key to .parquet extension
            parquet_name = Path(filename).with_suffix(".parquet").name
            parquet_path = facility_dir / parquet_name
            df.to_parquet(parquet_path, index=False)

    @classmethod
    def get_patient_ids(cls, df: pd.DataFrame, table_name: str) -> pd.Series:
        """
        Extract patient IDs from a DataFrame.

        Args:
            df: DataFrame to extract from
            table_name: Name of the table (for determining column name)

        Returns:
            Series of patient UUIDs
        """
        if table_name == "patients.csv":
            return df["Id"]
        elif "PATIENT" in df.columns:
            return df["PATIENT"]
        elif "PATIENTID" in df.columns:  # claims.csv uses PATIENTID
            return df["PATIENTID"]
        else:
            raise ValueError(f"Cannot determine patient ID column for {table_name}")

    @classmethod
    def get_encounter_ids(cls, df: pd.DataFrame, table_name: str) -> pd.Series:
        """
        Extract encounter IDs from a DataFrame.

        Args:
            df: DataFrame to extract from
            table_name: Name of the table

        Returns:
            Series of encounter UUIDs
        """
        if table_name == "encounters.csv":
            return df["Id"]
        elif "ENCOUNTER" in df.columns:
            return df["ENCOUNTER"]
        elif "APPOINTMENTID" in df.columns:  # claims.csv uses APPOINTMENTID
            return df["APPOINTMENTID"]
        else:
            raise ValueError(f"Cannot determine encounter ID column for {table_name}")

    @staticmethod
    def _strip_synthea_numbers(patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip Synthea-generated numbers from name fields.

        Synthea appends numbers to names for uniqueness (e.g., "Nathan164" → "Nathan").
        We strip these to create realistic name collisions for entity resolution.

        Args:
            patients_df: Patients DataFrame with Synthea name fields

        Returns:
            DataFrame with numbers removed from name fields
        """
        df = patients_df.copy()

        # Name fields to process
        name_fields = ["FIRST", "LAST", "MAIDEN"]

        for field in name_fields:
            if field in df.columns:
                # Remove all digits, preserving original case and spacing
                df[field] = df[field].apply(
                    lambda x: re.sub(r"\d+", "", str(x)) if pd.notna(x) else x
                )
                # Clean up any extra whitespace created by digit removal
                df[field] = df[field].apply(
                    lambda x: " ".join(str(x).split()) if pd.notna(x) else x
                )

        return df
