"""Streaming utilities for chunked table-to-facility distribution."""

import pandas as pd

from pathlib import Path

from ..utils import DataHandler
from .data_splitter import DataSplitter


def stream_table_to_facilities(
    data_handler: DataHandler,
    data_splitter: DataSplitter,
    input_dir: Path,
    table_name: str,
    facilities_dir: Path,
    all_facilities: list[int],
    per_facility_patients: dict[int, set[str]],
    per_facility_claim_ids: dict[int, set[str]] | None = None,
    chunksize: int = 500_000,
    payer_transitions_date_ranges: dict[int, pd.DataFrame] | None = None,
    enc_to_fac_map: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Stream a Parquet table in chunks, filter per facility, write Parquet incrementally.

    Uses PyArrow ParquetWriter so only one chunk resides in memory at a time.
    For payer_transitions, uses pre-cached date ranges instead of re-reading
    encounters.parquet per facility per chunk.  For claims, captures
    per-facility claim IDs into *per_facility_claim_ids* (mutated in place).

    For encounter-linked tables (conditions, medications, observations, etc.),
    uses a single map+groupby per chunk instead of N × isin per facility.

    Returns:
        (total_rows, num_chunks) for progress reporting.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    writers: dict[int, pq.ParquetWriter] = {}
    schema: pa.Schema | None = None
    total_rows = 0
    num_chunks = 0

    # Determine which fast path applies
    is_encounter_linked = table_name in data_handler.ENCOUNTER_LINKED_TABLES
    is_claims = table_name == "claims"
    is_encounters = table_name == "encounters"
    is_reference = table_name in data_handler.REFERENCE_TABLES

    enc_to_fac: dict[str, int] | None = enc_to_fac_map

    def _write_subset(facility_id: int, subset: pd.DataFrame) -> None:
        nonlocal schema
        if len(subset) == 0:
            return
        table = pa.Table.from_pandas(subset, preserve_index=False)
        if schema is None:
            schema = table.schema
        if facility_id not in writers:
            path = data_handler.facility_parquet_path(
                facilities_dir, facility_id, table_name
            )
            writers[facility_id] = pq.ParquetWriter(str(path), schema)
        writers[facility_id].write_table(table)

    for chunk in data_handler.stream_table_chunks(input_dir, table_name, chunksize):
        total_rows += len(chunk)
        num_chunks += 1

        if is_reference:
            # Reference tables: write full chunk to every facility
            for facility_id in all_facilities:
                _write_subset(facility_id, chunk)

        elif is_encounters:
            # Fast path for encounters: map Id → facility
            fac_col = chunk["Id"].map(enc_to_fac)
            mask = fac_col.notna()
            if mask.any():
                matched = chunk[mask]
                fac_ids = fac_col[mask].astype(int)
                for facility_id, group_df in matched.groupby(fac_ids):
                    _write_subset(facility_id, group_df)

        elif is_encounter_linked:
            # Fast path: single map + groupby instead of N × isin
            fac_col = chunk["ENCOUNTER"].map(enc_to_fac)
            mask = fac_col.notna()
            if mask.any():
                matched = chunk[mask]
                fac_ids = fac_col[mask].astype(int)
                for facility_id, group_df in matched.groupby(fac_ids):
                    _write_subset(facility_id, group_df)

        elif is_claims:
            # Fast path for claims: map APPOINTMENTID → facility
            fac_col = chunk["APPOINTMENTID"].map(enc_to_fac)
            mask = fac_col.notna()
            if mask.any():
                matched = chunk[mask]
                fac_ids = fac_col[mask].astype(int)
                for facility_id, group_df in matched.groupby(fac_ids):
                    # Capture claim IDs
                    if per_facility_claim_ids is not None:
                        per_facility_claim_ids[facility_id].update(
                            group_df["Id"].values
                        )
                    _write_subset(facility_id, group_df)

        elif table_name == "payer_transitions":
            # Use cached date ranges instead of re-reading parquet each time
            _empty_enc: set[str] = set()
            for facility_id in all_facilities:
                cached_ranges = None
                if payer_transitions_date_ranges is not None:
                    cached_ranges = payer_transitions_date_ranges.get(facility_id)

                if cached_ranges is not None:
                    subset = data_splitter.filter_table_for_facility(
                        table_name,
                        chunk,
                        facility_id,
                        per_facility_patients[facility_id],
                        _empty_enc,
                        patient_date_ranges=cached_ranges,
                        copy=False,
                    )
                else:
                    # Fallback: read from disk
                    try:
                        fac_encounters_df = data_handler.read_facility_table(
                            facilities_dir, facility_id, "encounters"
                        )
                    except FileNotFoundError:
                        continue
                    subset = data_splitter.filter_table_for_facility(
                        table_name,
                        chunk,
                        facility_id,
                        per_facility_patients[facility_id],
                        _empty_enc,
                        facility_encounters_df=fac_encounters_df,
                        copy=False,
                    )
                    del fac_encounters_df
                _write_subset(facility_id, subset)

        else:
            # Generic fallback (claims_transactions only — all other table
            # types have dedicated fast paths above)
            _empty_set: set[str] = set()
            for facility_id in all_facilities:
                subset = data_splitter.filter_table_for_facility(
                    table_name,
                    chunk,
                    facility_id,
                    per_facility_patients[facility_id],
                    _empty_set,
                    facility_claim_ids=(
                        per_facility_claim_ids.get(facility_id)
                        if per_facility_claim_ids
                        else None
                    ),
                    copy=False,
                )
                _write_subset(facility_id, subset)

    # Close all writers
    for w in writers.values():
        w.close()

    # Write empty Parquet for facilities that had no rows for this table
    if schema is not None:
        empty_table = schema.empty_table()
        for facility_id in all_facilities:
            if facility_id not in writers:
                path = data_handler.facility_parquet_path(
                    facilities_dir, facility_id, table_name
                )
                pq.write_table(empty_table, str(path))

    return total_rows, num_chunks
