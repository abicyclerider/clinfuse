"""Data reading and writing utilities for Parquet-based pipeline."""

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd


class DataHandler:
    """Reads and writes facility Parquet data."""

    # Standard tables in Synthea output (plain names, no extensions)
    SYNTHEA_TABLES = [
        "patients",
        "encounters",
        "conditions",
        "medications",
        "observations",
        "procedures",
        "immunizations",
        "allergies",
        "careplans",
        "imaging_studies",
        "devices",
        "supplies",
        "claims",
        "claims_transactions",
        "payer_transitions",
        "organizations",
        "providers",
        "payers",
    ]

    # Reference tables (copied to all facilities unchanged)
    REFERENCE_TABLES = [
        "organizations",
        "providers",
        "payers",
    ]

    # Tables with PATIENT foreign key
    PATIENT_LINKED_TABLES = [
        "encounters",
        "payer_transitions",
    ]

    # Tables with ENCOUNTER foreign key
    ENCOUNTER_LINKED_TABLES = [
        "conditions",
        "medications",
        "observations",
        "procedures",
        "immunizations",
        "allergies",
        "careplans",
        "imaging_studies",
        "devices",
        "supplies",
    ]

    @classmethod
    def load_table(
        cls,
        input_dir: Path,
        table_name: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a single Parquet table.

        Args:
            input_dir: Directory containing Parquet files
            table_name: Table name (e.g. 'patients')
            columns: Optional list of columns to load (reduces memory)

        Returns:
            DataFrame with table contents
        """
        parquet_path = input_dir / f"{table_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        return pd.read_parquet(parquet_path, columns=columns)

    @classmethod
    def stream_table_chunks(
        cls,
        input_dir: Path,
        table_name: str,
        chunksize: int = 500_000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Yield DataFrames chunk by chunk from a Parquet file.

        Args:
            input_dir: Directory containing Parquet files
            table_name: Table name (e.g. 'encounters')
            chunksize: Number of rows per chunk
            columns: Optional list of columns to load

        Yields:
            DataFrame chunks
        """
        import pyarrow.parquet as pq

        parquet_path = input_dir / f"{table_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        pf = pq.ParquetFile(str(parquet_path))
        for batch in pf.iter_batches(batch_size=chunksize, columns=columns):
            yield batch.to_pandas()

    @classmethod
    def facility_parquet_path(
        cls, output_dir: Path, facility_id: int, table_name: str
    ) -> Path:
        """Return the Parquet path for a facility table, creating dirs as needed."""
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        facility_dir.mkdir(parents=True, exist_ok=True)
        return facility_dir / f"{table_name}.parquet"

    @classmethod
    def write_facility_table(
        cls,
        df: pd.DataFrame,
        output_dir: Path,
        facility_id: int,
        table_name: str,
    ) -> None:
        """
        Write one Parquet table for one facility.

        Args:
            df: DataFrame to write
            output_dir: Base output directory (parent of facility_NNN dirs)
            facility_id: Facility identifier
            table_name: Table name (e.g. 'patients')
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        facility_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(facility_dir / f"{table_name}.parquet", index=False)

    @classmethod
    def read_facility_table(
        cls,
        output_dir: Path,
        facility_id: int,
        table_name: str,
    ) -> pd.DataFrame:
        """
        Read one Parquet file back for a facility.

        Args:
            output_dir: Base output directory (parent of facility_NNN dirs)
            facility_id: Facility identifier
            table_name: Table name (e.g. 'encounters')

        Returns:
            DataFrame with Parquet contents
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        parquet_path = facility_dir / f"{table_name}.parquet"
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
            Dictionary mapping table name to DataFrame
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        result = {}
        for table_name in cls.SYNTHEA_TABLES:
            parquet_path = facility_dir / f"{table_name}.parquet"
            if parquet_path.exists():
                result[table_name] = pd.read_parquet(parquet_path)
        return result

    @classmethod
    def write_facility_data(
        cls,
        facility_tables: Dict[str, pd.DataFrame],
        output_dir: Path,
        facility_id: int,
    ) -> None:
        """
        Write all data files for a facility as Parquet.

        Args:
            facility_tables: Dictionary mapping table name to DataFrame
            output_dir: Base output directory
            facility_id: Facility identifier
        """
        facility_dir = output_dir / f"facility_{facility_id:03d}"
        facility_dir.mkdir(parents=True, exist_ok=True)

        for table_name, df in facility_tables.items():
            df.to_parquet(facility_dir / f"{table_name}.parquet", index=False)

    @classmethod
    def get_patient_ids(cls, df: pd.DataFrame, table_name: str) -> pd.Series:
        """
        Extract patient IDs from a DataFrame.

        Args:
            df: DataFrame to extract from
            table_name: Name of the table

        Returns:
            Series of patient UUIDs
        """
        if table_name == "patients":
            return df["Id"]
        elif "PATIENT" in df.columns:
            return df["PATIENT"]
        elif "PATIENTID" in df.columns:  # claims uses PATIENTID
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
        if table_name == "encounters":
            return df["Id"]
        elif "ENCOUNTER" in df.columns:
            return df["ENCOUNTER"]
        elif "APPOINTMENTID" in df.columns:  # claims uses APPOINTMENTID
            return df["APPOINTMENTID"]
        else:
            raise ValueError(f"Cannot determine encounter ID column for {table_name}")
