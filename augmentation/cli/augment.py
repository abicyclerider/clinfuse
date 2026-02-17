"""Command-line interface for data augmentation."""

import platform
import resource
import time
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from ..config import AugmentationConfig
from ..core import DataSplitter, ErrorInjector, FacilityAssigner, GroundTruthTracker
from ..generators import FacilityGenerator
from ..utils import DataHandler, DataValidator

console = Console()


_IS_MACOS = platform.system() == "Darwin"


def _log_memory(label: str) -> None:
    """Log current peak RSS memory usage with a label."""
    # ru_maxrss is in KB on Linux, bytes on macOS
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = ru.ru_maxrss / (1024 * 1024) if _IS_MACOS else ru.ru_maxrss / 1024
    console.print(f"  [dim][mem] {label}: peak RSS = {rss_mb:.0f} MB[/dim]")


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
    help="Path to output directory",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to configuration YAML file",
)
@click.option(
    "--facilities",
    type=int,
    default=None,
    help="Number of facilities to generate (overrides config)",
)
@click.option(
    "--error-rate",
    type=float,
    default=None,
    help="Global error rate (overrides config)",
)
@click.option(
    "--random-seed",
    type=int,
    default=None,
    help="Random seed (overrides config)",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Run validation after augmentation",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform dry run without writing output",
)
def main(
    input_dir: Path,
    output_dir: Path,
    config_file: Path,
    facilities: int,
    error_rate: float,
    random_seed: int,
    validate: bool,
    dry_run: bool,
):
    """
    Entity Resolution Data Augmentation System

    Augments Synthea medical data with realistic demographic errors
    and multi-facility distribution for entity resolution testing.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Entity Resolution Data Augmentation System[/bold cyan]",
            subtitle="Powered by Synthea",
        )
    )

    # Load configuration
    config = load_configuration(input_dir, output_dir, config_file)

    # Apply CLI overrides
    if facilities is not None:
        config.facility_distribution.num_facilities = facilities
    if error_rate is not None:
        config.error_injection.global_error_rate = error_rate
    if random_seed is not None:
        config.random_seed = random_seed

    # Display configuration
    display_configuration(config)

    if dry_run:
        console.print("\n[yellow]Dry run mode - no output will be written[/yellow]")
        return

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / f"run_{timestamp}"

    console.print(f"\n[green]Output directory:[/green] {run_output_dir}")

    # Run augmentation pipeline
    try:
        run_augmentation_pipeline(config, run_output_dir, validate)
        console.print(
            "\n[bold green]✓ Augmentation completed successfully![/bold green]"
        )
    except Exception as e:
        console.print(f"\n[bold red]✗ Error during augmentation:[/bold red] {e}")
        raise


def load_configuration(
    input_dir: Path,
    output_dir: Path,
    config_file: Path = None,
) -> AugmentationConfig:
    """Load and validate configuration."""
    with console.status("[bold blue]Loading configuration..."):
        if config_file:
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            # Use default config
            default_config_path = (
                Path(__file__).parent.parent / "config" / "default_config.yaml"
            )
            with open(default_config_path, "r") as f:
                config_dict = yaml.safe_load(f)

        # Override paths
        config_dict["paths"] = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }

        config = AugmentationConfig(**config_dict)

    console.print("[green]✓[/green] Configuration loaded")
    return config


def display_configuration(config: AugmentationConfig):
    """Display configuration summary."""
    table = Table(title="Configuration Summary", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row(
        "Number of Facilities", str(config.facility_distribution.num_facilities)
    )
    table.add_row(
        "Global Error Rate", f"{config.error_injection.global_error_rate:.0%}"
    )
    table.add_row(
        "Multiple Errors Probability",
        f"{config.error_injection.multiple_errors_probability:.0%}",
    )
    table.add_row(
        "Primary Facility Weight",
        f"{config.facility_distribution.primary_facility_weight:.0%}",
    )
    table.add_row("Random Seed", str(config.random_seed))

    console.print(table)


def _build_encounter_to_facility_map(
    per_facility_encounters: dict[int, set[str]],
) -> dict[str, int]:
    """Build a single encounter_id → facility_id dict from per-facility sets."""
    enc_to_fac: dict[str, int] = {}
    for fid, enc_ids in per_facility_encounters.items():
        for eid in enc_ids:
            enc_to_fac[eid] = fid
    return enc_to_fac


def _stream_table_to_facilities(
    data_handler: DataHandler,
    data_splitter: DataSplitter,
    input_dir: Path,
    table_name: str,
    facilities_dir: Path,
    all_facilities: list[int],
    per_facility_patients: dict[int, set[str]],
    per_facility_encounters: dict[int, set[str]] | None = None,
    per_facility_claim_ids: dict[int, set[str]] | None = None,
    chunksize: int = 500_000,
    payer_transitions_date_ranges: dict[int, pd.DataFrame] | None = None,
    enc_to_fac_map: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Stream a CSV in chunks, filter per facility, write Parquet incrementally.

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
    is_claims = table_name == "claims.csv"
    is_encounters = table_name == "encounters.csv"
    is_reference = table_name in data_handler.REFERENCE_TABLES

    # Use pre-built map if provided, otherwise build from per-facility sets
    enc_to_fac: dict[str, int] | None = enc_to_fac_map
    if enc_to_fac is None and (is_encounter_linked or is_claims or is_encounters):
        if per_facility_encounters is not None:
            enc_to_fac = _build_encounter_to_facility_map(per_facility_encounters)

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

    for chunk in data_handler.stream_csv_chunks(input_dir, table_name, chunksize):
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

        elif table_name == "payer_transitions.csv":
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
                            facilities_dir, facility_id, "encounters.csv"
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


def run_augmentation_pipeline(
    config: AugmentationConfig,
    output_dir: Path,
    validate: bool,
):
    """Run the complete augmentation pipeline.

    Uses chunked CSV streaming so that no single source table needs to fit
    in memory.  Peak memory is bounded by one CSV chunk (~500 K rows) plus
    the assignment dicts and ground-truth tracker.

      Phase A: Lightweight encounters load (3 cols) for assignment, then
               patients error-injection/write, encounters chunked write.
      Phase B: Stream each remaining table in chunks with incremental
               Parquet writing via PyArrow.
      Phase C: Validate each facility by reading back Parquet files.
      Phase D: Write metadata (facilities, ground truth, config, statistics).
    """
    input_dir = config.paths.input_dir
    facilities_dir = output_dir / "facilities"
    t0 = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # ── Task 1: Load patients + lightweight encounters + organizations ──
        task1 = progress.add_task("[cyan]Loading core CSV files...", total=None)
        data_handler = DataHandler()
        patients_df = data_handler.load_single_csv(input_dir, "patients.csv")
        # Only 3 columns needed for assignment — full encounters streamed later
        encounters_light = data_handler.load_single_csv(
            input_dir, "encounters.csv", usecols=["Id", "PATIENT", "START"]
        )
        organizations_df = data_handler.load_single_csv(input_dir, "organizations.csv")
        num_patients_loaded = len(patients_df)
        num_encounters_loaded = len(encounters_light)
        progress.update(task1, completed=True, total=1)
        console.print(
            f"[green]✓[/green] Loaded {num_patients_loaded} patients, "
            f"{num_encounters_loaded} encounters (lightweight)"
        )
        _log_memory("after loading CSV data")

        # ── Task 2: Generate facility metadata ──
        task2 = progress.add_task("[cyan]Generating facility metadata...", total=None)
        facility_generator = FacilityGenerator(random_seed=config.random_seed)
        facilities_df = facility_generator.generate_facilities(
            config.facility_distribution.num_facilities,
            organizations_df,
        )
        progress.update(task2, completed=True, total=1)
        console.print(f"[green]✓[/green] Generated {len(facilities_df)} facilities")

        # ── Task 3: Assign patients to facilities ──
        task3 = progress.add_task(
            "[cyan]Assigning patients to facilities...", total=None
        )
        facility_assigner = FacilityAssigner(
            config.facility_distribution, random_seed=config.random_seed
        )
        patient_facilities, encounter_facilities = (
            facility_assigner.assign_patients_to_facilities(
                patients_df, encounters_light
            )
        )
        # Free the lightweight DF immediately — the dicts hold all we need
        del encounters_light
        progress.update(task3, completed=True, total=1)
        assignment_stats = facility_assigner.get_facility_statistics(
            patient_facilities, encounter_facilities
        )
        console.print("[green]✓[/green] Assigned patients across facilities")

        # ── Precompute per-facility ID sets ──
        all_facilities = sorted(
            {fid for fids in patient_facilities.values() for fid in fids}
        )
        num_facilities = len(all_facilities)

        per_facility_patients: dict[int, set[str]] = {
            fid: set() for fid in all_facilities
        }
        for patient_uuid, fids in patient_facilities.items():
            for fid in fids:
                per_facility_patients[fid].add(patient_uuid)

        per_facility_encounters: dict[int, set[str]] = {
            fid: set() for fid in all_facilities
        }
        for enc_id, fid in encounter_facilities.items():
            per_facility_encounters[fid].add(enc_id)

        # Build encounter counts per patient per facility by streaming
        # (avoids holding encounters_light in memory alongside the dicts)
        facility_enc_counts: dict[int, dict[str, int]] = {
            fid: {} for fid in all_facilities
        }
        enc_fac_series = pd.Series(encounter_facilities)
        for chunk in data_handler.stream_csv_chunks(
            input_dir, "encounters.csv", usecols=["Id", "PATIENT"]
        ):
            # Map encounter IDs to facility IDs in one vectorized pass
            fac_ids = chunk["Id"].map(enc_fac_series)
            mapped = chunk[fac_ids.notna()].copy()
            mapped["_fac"] = fac_ids[fac_ids.notna()].astype(int)
            # Group by (facility, patient) and count
            counts = mapped.groupby(["_fac", "PATIENT"]).size()
            for (fid, patient), cnt in counts.items():
                fec = facility_enc_counts[fid]
                fec[patient] = fec.get(patient, 0) + cnt
        del enc_fac_series
        # Free the large assignment dicts — all data is now in per_facility_*
        # sets, facility_enc_counts, and assignment_stats.
        del patient_facilities, encounter_facilities
        _log_memory("after facility assignment")

        # ── Phase A: patients — error injection, ground truth, write ──
        task_a = progress.add_task(
            "[cyan]Phase A: patients & encounters...", total=num_facilities + 1
        )

        data_splitter = DataSplitter()
        error_injector = ErrorInjector(
            config.error_injection, random_seed=config.random_seed
        )
        ground_truth_tracker = GroundTruthTracker()
        all_error_logs = []

        for facility_id in all_facilities:
            fac_patients_set = per_facility_patients[facility_id]

            # Filter patients
            fac_patients_df = data_splitter.filter_table_for_facility(
                "patients.csv",
                patients_df,
                facility_id,
                fac_patients_set,
                per_facility_encounters[facility_id],
            )

            # Inject errors
            errored_patients_df, error_log = error_injector.inject_errors_into_patients(
                fac_patients_df, facility_id
            )

            # Track ground truth (using precomputed encounter counts)
            enc_counts = facility_enc_counts[facility_id]

            # Build patient→error_types lookup (avoids O(patients × errors) scan)
            errors_by_patient: dict[str, list[str]] = {}
            for err in error_log:
                errors_by_patient.setdefault(err["patient_uuid"], []).append(
                    err["error_type"]
                )

            # Use itertuples() (much faster than iterrows())
            id_col_idx = errored_patients_df.columns.get_loc("Id")
            for row in errored_patients_df.itertuples(index=False):
                patient_uuid = row[id_col_idx]
                patient_errors = errors_by_patient.get(patient_uuid, [])
                ground_truth_tracker.add_patient_facility_mapping(
                    patient_uuid,
                    facility_id,
                    enc_counts.get(patient_uuid, 0),
                    row._asdict(),
                    patient_errors,
                )

            ground_truth_tracker.add_error_records(error_log)
            all_error_logs.extend(error_log)

            # Write patients (with errors) and organizations
            data_handler.write_facility_table(
                errored_patients_df, facilities_dir, facility_id, "patients.csv"
            )
            data_handler.write_facility_table(
                organizations_df, facilities_dir, facility_id, "organizations.csv"
            )

            progress.update(task_a, advance=1)

        del patients_df, organizations_df, facility_enc_counts

        # Stream full encounters.csv in chunks → per-facility Parquet
        progress.update(task_a, description="[cyan]Phase A: streaming encounters...")
        enc_rows, enc_chunks = _stream_table_to_facilities(
            data_handler,
            data_splitter,
            input_dir,
            "encounters.csv",
            facilities_dir,
            all_facilities,
            per_facility_patients,
            per_facility_encounters,
        )
        progress.update(task_a, advance=1)

        elapsed_a = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Phase A complete: {len(all_error_logs)} errors, "
            f"{enc_rows:,} encounters in {enc_chunks} chunks "
            f"[dim]({elapsed_a:.1f}s)[/dim]"
        )
        error_stats = error_injector.get_error_statistics(all_error_logs)
        del all_error_logs
        _log_memory("after Phase A")

        # Build encounter→facility map once, then free the large per-facility
        # encounter sets (~900MB for 10M UUIDs).  Phase B uses the map for
        # encounter-linked tables; encounters.csv is already on disk.
        enc_to_fac_map = _build_encounter_to_facility_map(per_facility_encounters)
        del per_facility_encounters
        _log_memory("after building enc_to_fac_map (freed per_facility_encounters)")

        # ── Phase B: stream remaining tables one at a time ──
        # Order: claims before claims_transactions; encounters already on disk.
        phase_b_tables = [
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
            "providers.csv",
            "payers.csv",
            "claims.csv",
            "claims_transactions.csv",
            "payer_transitions.csv",
        ]

        task_b = progress.add_task(
            "[cyan]Phase B: streaming tables...", total=len(phase_b_tables)
        )

        per_facility_claim_ids: dict[int, set[str]] = {
            fid: set() for fid in all_facilities
        }

        # Pre-cache per-facility patient date ranges for payer_transitions
        # Only stores the small aggregate (one row per patient), not the full
        # encounters DataFrame, to avoid OOM on large populations.
        payer_transitions_date_ranges: dict[int, pd.DataFrame] | None = None
        if (input_dir / "payer_transitions.csv").exists():
            payer_transitions_date_ranges = {}
            for facility_id in all_facilities:
                try:
                    enc_df = data_handler.read_facility_table(
                        facilities_dir, facility_id, "encounters.csv"
                    )
                    if len(enc_df) > 0:
                        date_ranges = (
                            enc_df.groupby("PATIENT")["START"]
                            .agg(min_date="min", max_date="max")
                            .reset_index()
                        )
                        payer_transitions_date_ranges[facility_id] = date_ranges
                    del enc_df
                except FileNotFoundError:
                    pass
            _log_memory("after caching payer_transitions date ranges")

        for table_name in phase_b_tables:
            file_path = input_dir / table_name
            if not file_path.exists():
                progress.update(task_b, advance=1)
                continue

            short_name = table_name.replace(".csv", "")
            progress.update(task_b, description=f"[cyan]Phase B: {short_name}...")

            rows, chunks = _stream_table_to_facilities(
                data_handler,
                data_splitter,
                input_dir,
                table_name,
                facilities_dir,
                all_facilities,
                per_facility_patients,
                per_facility_claim_ids=per_facility_claim_ids,
                payer_transitions_date_ranges=payer_transitions_date_ranges,
                enc_to_fac_map=enc_to_fac_map,
            )
            t_table = time.monotonic() - t0
            console.print(
                f"  [dim]{short_name}: {rows:,} rows in {chunks} chunk(s) "
                f"({t_table:.0f}s elapsed)[/dim]"
            )
            _log_memory(f"after {short_name}")
            progress.update(task_b, advance=1)

        del per_facility_claim_ids, payer_transitions_date_ranges, enc_to_fac_map
        elapsed_b = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Phase B complete: streamed {len(phase_b_tables)} tables "
            f"[dim]({elapsed_b:.1f}s)[/dim]"
        )
        _log_memory("after Phase B (freed maps)")

        # ── Phase C: validation ──
        validation_errors = []
        if validate:
            task_c = progress.add_task(
                "[cyan]Phase C: validating...", total=num_facilities
            )
            validator = DataValidator()

            for facility_id in all_facilities:
                facility_data = data_handler.read_all_facility_tables(
                    facilities_dir, facility_id
                )
                is_valid, errors = validator.validate_facility_csvs(facility_data)
                if not is_valid:
                    validation_errors.extend(
                        [f"Facility {facility_id}: {err}" for err in errors]
                    )
                del facility_data
                progress.update(task_c, advance=1)

            if validation_errors:
                console.print("\n[yellow]⚠ Validation warnings:[/yellow]")
                for error in validation_errors[:10]:
                    console.print(f"  • {error}")
            else:
                console.print("[green]✓[/green] All validations passed")

        # ── Phase D: metadata ──
        task_meta = progress.add_task("[cyan]Phase D: writing metadata...", total=5)

        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        facilities_df.to_parquet(metadata_dir / "facilities.parquet", index=False)
        progress.update(task_meta, advance=1)

        ground_truth_tracker.export_ground_truth(metadata_dir / "ground_truth.parquet")
        progress.update(task_meta, advance=1)

        ground_truth_tracker.export_error_log_jsonl(metadata_dir / "error_log.jsonl")
        progress.update(task_meta, advance=1)

        # Save run configuration
        with open(metadata_dir / "run_config.yaml", "w") as f:
            yaml.dump(config.model_dump(), f)
        progress.update(task_meta, advance=1)

        # Write statistics
        statistics_dir = output_dir / "statistics"
        ground_truth_stats = ground_truth_tracker.generate_statistics()

        combined_stats = {
            "assignment": assignment_stats,
            "errors": error_stats,
            "ground_truth": ground_truth_stats,
        }

        ground_truth_tracker.export_statistics_json(
            statistics_dir / "augmentation_report.json", additional_stats=combined_stats
        )
        progress.update(task_meta, advance=1)

        elapsed_total = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Output written to {output_dir} "
            f"[dim](total {elapsed_total:.1f}s)[/dim]"
        )

    # Display final summary
    display_summary(assignment_stats, error_stats, ground_truth_stats)


def display_summary(
    assignment_stats: dict, error_stats: dict, ground_truth_stats: dict
):
    """Display final summary statistics."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Augmentation Summary[/bold cyan]")
    console.print("=" * 60)

    console.print("\n[bold]Patients & Encounters:[/bold]")
    console.print(f"  Total Patients: {assignment_stats['total_patients']}")
    console.print(f"  Total Encounters: {assignment_stats['total_encounters']}")

    console.print("\n[bold]Facility Distribution:[/bold]")
    for count, num_patients in sorted(
        assignment_stats["facility_count_distribution"].items()
    ):
        pct = num_patients / assignment_stats["total_patients"] * 100
        console.print(
            f"  {num_patients} patients at {count} facility(ies) ({pct:.1f}%)"
        )

    console.print("\n[bold]Error Injection:[/bold]")
    console.print(f"  Total Errors Applied: {error_stats['total_errors']}")
    console.print(
        f"  Patients with Errors: {ground_truth_stats['patients_with_errors']}"
    )
    console.print(f"  Actual Error Rate: {ground_truth_stats['error_rate']:.1%}")

    console.print("\n[bold]Top Error Types:[/bold]")
    sorted_errors = sorted(
        error_stats["errors_by_type"].items(), key=lambda x: x[1], reverse=True
    )
    for error_type, count in sorted_errors[:5]:
        console.print(f"  {error_type}: {count}")

    console.print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
