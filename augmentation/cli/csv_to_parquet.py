"""Convert Synthea CSV files to Parquet format.

This is the ONLY code that reads raw CSVs. It applies date/timestamp typing
and strips Synthea-generated numbers from patient name fields during conversion.
"""

import re
import time
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from rich.console import Console

console = Console()

# Columns that should be parsed as timestamps per table
_TIMESTAMP_COLUMNS: dict[str, list[str]] = {
    "encounters": ["START", "STOP"],
    "payer_transitions": ["START_DATE", "END_DATE"],
    "patients": ["BIRTHDATE", "DEATHDATE"],
}

# Threshold for streaming mode (1 GB)
_STREAM_THRESHOLD = 1 * 1024 * 1024 * 1024


def _strip_synthea_numbers_arrow(table: pa.Table) -> pa.Table:
    """Strip Synthea-generated numbers from name fields in a patients table.

    Synthea appends numbers to names for uniqueness (e.g., "Nathan164" → "Nathan").
    We strip these to create realistic name collisions for entity resolution.
    """
    for field in ["FIRST", "LAST", "MAIDEN"]:
        col_idx = table.schema.get_field_index(field)
        if col_idx < 0:
            continue
        col = table.column(col_idx)
        # Process via Python (PyArrow compute doesn't have regex replace on all versions)
        new_values = []
        for val in col.to_pylist():
            if val is None:
                new_values.append(None)
            else:
                stripped = re.sub(r"\d+", "", str(val)).strip()
                stripped = re.sub(r"\s+", " ", stripped)
                new_values.append(stripped)
        table = table.set_column(
            col_idx, table.schema.field(col_idx), pa.array(new_values, type=pa.string())
        )
    return table


def _build_convert_options(table_name: str) -> tuple[pcsv.ConvertOptions, bool]:
    """Build PyArrow CSV convert options for a table.

    Returns (convert_options, needs_name_stripping).
    """
    ts_cols = _TIMESTAMP_COLUMNS.get(table_name, [])
    column_types = {}
    for col in ts_cols:
        column_types[col] = pa.timestamp("us")

    convert_opts = pcsv.ConvertOptions(
        column_types=column_types if column_types else None,
        timestamp_parsers=["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"],
    )
    needs_stripping = table_name == "patients"
    return convert_opts, needs_stripping


def _convert_file(csv_path: Path, output_dir: Path, table_name: str) -> int:
    """Convert a single CSV to Parquet. Returns row count."""
    convert_opts, needs_stripping = _build_convert_options(table_name)
    read_opts = pcsv.ReadOptions(block_size=64 * 1024 * 1024)
    parse_opts = pcsv.ParseOptions()
    out_path = output_dir / f"{table_name}.parquet"

    file_size = csv_path.stat().st_size

    if file_size > _STREAM_THRESHOLD:
        # Streaming mode for large files
        reader = pcsv.open_csv(
            str(csv_path),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts,
        )
        writer = None
        total_rows = 0
        for batch in reader:
            table = pa.Table.from_batches([batch])
            if needs_stripping:
                table = _strip_synthea_numbers_arrow(table)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            total_rows += len(batch)
        if writer is not None:
            writer.close()
        return total_rows
    else:
        # Read entire file at once (faster for small/medium files)
        table = pcsv.read_csv(
            str(csv_path),
            read_options=read_opts,
            parse_options=parse_opts,
            convert_options=convert_opts,
        )
        if needs_stripping:
            table = _strip_synthea_numbers_arrow(table)
        pq.write_table(table, str(out_path))
        return len(table)


@click.command()
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to Synthea CSV directory",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output Parquet directory",
)
def main(input_dir: Path, output_dir: Path):
    """Convert Synthea CSV files to Parquet format."""
    from ..utils import DataHandler

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    console.print(f"[cyan]Converting CSVs from[/cyan] {input_dir}")
    console.print(f"[cyan]Writing Parquet to[/cyan]   {output_dir}")

    converted = 0
    for table_name in DataHandler.SYNTHEA_TABLES:
        csv_path = input_dir / f"{table_name}.csv"
        if not csv_path.exists():
            console.print(f"  [dim]skip {table_name}.csv (not found)[/dim]")
            continue

        rows = _convert_file(csv_path, output_dir, table_name)
        size_mb = (output_dir / f"{table_name}.parquet").stat().st_size / (1024 * 1024)
        console.print(
            f"  [green]✓[/green] {table_name}: {rows:,} rows, {size_mb:.1f} MB"
        )
        converted += 1

    elapsed = time.monotonic() - t0
    console.print(
        f"\n[bold green]✓ Converted {converted} tables[/bold green] "
        f"[dim]({elapsed:.1f}s)[/dim]"
    )


if __name__ == "__main__":
    main()
