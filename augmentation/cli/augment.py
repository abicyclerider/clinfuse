"""Command-line interface for data augmentation."""

import time
from datetime import datetime
from pathlib import Path

import click
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


def run_augmentation_pipeline(
    config: AugmentationConfig,
    output_dir: Path,
    validate: bool,
):
    """Run the complete augmentation pipeline.

    Uses two-phase table streaming to keep peak memory low:
      Phase A: Load patients + encounters + organizations, do assignment/errors/write.
      Phase B: Stream each remaining table from disk, filter per facility, write.
      Phase C: Validate each facility by reading back all Parquet files.
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
        # ── Task 1: Load core tables only (patients, encounters, organizations) ──
        task1 = progress.add_task("[cyan]Loading core CSV files...", total=None)
        data_handler = DataHandler()
        patients_df = data_handler.load_single_csv(input_dir, "patients.csv")
        encounters_df = data_handler.load_single_csv(input_dir, "encounters.csv")
        organizations_df = data_handler.load_single_csv(input_dir, "organizations.csv")
        progress.update(task1, completed=True, total=1)
        console.print(
            f"[green]✓[/green] Loaded {len(patients_df)} patients, "
            f"{len(encounters_df)} encounters"
        )

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
            facility_assigner.assign_patients_to_facilities(patients_df, encounters_df)
        )
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

        per_facility_patients: dict[int, set[str]] = {fid: set() for fid in all_facilities}
        for patient_uuid, fids in patient_facilities.items():
            for fid in fids:
                per_facility_patients[fid].add(patient_uuid)

        per_facility_encounters: dict[int, set[str]] = {fid: set() for fid in all_facilities}
        for enc_id, fid in encounter_facilities.items():
            per_facility_encounters[fid].add(enc_id)

        # ── Phase A: patients + encounters — error injection, ground truth, write ──
        task_a = progress.add_task(
            "[cyan]Phase A: patients & encounters...", total=num_facilities
        )

        data_splitter = DataSplitter()
        error_injector = ErrorInjector(
            config.error_injection, random_seed=config.random_seed
        )
        ground_truth_tracker = GroundTruthTracker()
        all_error_logs = []

        for facility_id in all_facilities:
            fac_patients = per_facility_patients[facility_id]
            fac_encounters = per_facility_encounters[facility_id]

            # Filter patients & encounters
            fac_patients_df = data_splitter.filter_table_for_facility(
                "patients.csv", patients_df, facility_id, fac_patients, fac_encounters
            )
            fac_encounters_df = data_splitter.filter_table_for_facility(
                "encounters.csv", encounters_df, facility_id, fac_patients, fac_encounters
            )

            # Inject errors into patients
            errored_patients_df, error_log = error_injector.inject_errors_into_patients(
                fac_patients_df, facility_id
            )

            # Track ground truth
            for _, patient in errored_patients_df.iterrows():
                patient_uuid = patient["Id"]
                num_enc = len(
                    fac_encounters_df[fac_encounters_df["PATIENT"] == patient_uuid]
                )
                patient_errors = [
                    err["error_type"]
                    for err in error_log
                    if err["patient_uuid"] == patient_uuid
                ]
                ground_truth_tracker.add_patient_facility_mapping(
                    patient_uuid,
                    facility_id,
                    num_enc,
                    patient.to_dict(),
                    patient_errors,
                )

            ground_truth_tracker.add_error_records(error_log)
            all_error_logs.extend(error_log)

            # Write patients (with errors) and encounters
            data_handler.write_facility_table(
                errored_patients_df, facilities_dir, facility_id, "patients.csv"
            )
            data_handler.write_facility_table(
                fac_encounters_df, facilities_dir, facility_id, "encounters.csv"
            )

            # Write organizations (reference table — same for all)
            data_handler.write_facility_table(
                organizations_df, facilities_dir, facility_id, "organizations.csv"
            )

            progress.update(task_a, advance=1)

        # Free core tables
        del patients_df, encounters_df, organizations_df

        elapsed_a = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Phase A complete: patients & encounters for "
            f"{num_facilities} facilities [dim]({elapsed_a:.1f}s)[/dim]"
        )
        error_stats = error_injector.get_error_statistics(all_error_logs)
        console.print(
            f"[green]✓[/green] Applied {error_stats['total_errors']} demographic errors"
        )

        # ── Phase B: stream remaining tables one at a time ──
        # Order matters: claims before claims_transactions, encounters already on disk.
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

        # Track per-facility claim IDs (populated when we process claims.csv)
        per_facility_claim_ids: dict[int, set[str]] = {fid: set() for fid in all_facilities}

        for table_name in phase_b_tables:
            file_path = input_dir / table_name
            if not file_path.exists():
                progress.update(task_b, advance=1)
                continue

            short_name = table_name.replace(".csv", "")
            progress.update(
                task_b,
                description=f"[cyan]Phase B: {short_name}...",
            )
            source_df = data_handler.load_single_csv(input_dir, table_name)
            mem_mb = source_df.memory_usage(deep=True).sum() / 1_048_576
            console.print(
                f"  [dim]{short_name}: {len(source_df):,} rows, "
                f"{mem_mb:.0f} MB[/dim]"
            )

            for facility_id in all_facilities:
                fac_patients = per_facility_patients[facility_id]
                fac_encounters = per_facility_encounters[facility_id]

                # payer_transitions needs facility encounters DataFrame
                fac_encounters_df = None
                if table_name == "payer_transitions.csv":
                    fac_encounters_df = data_handler.read_facility_table(
                        facilities_dir, facility_id, "encounters.csv"
                    )

                subset = data_splitter.filter_table_for_facility(
                    table_name,
                    source_df,
                    facility_id,
                    fac_patients,
                    fac_encounters,
                    facility_encounters_df=fac_encounters_df,
                    facility_claim_ids=per_facility_claim_ids.get(facility_id),
                )

                # Capture claim IDs after filtering claims.csv
                if table_name == "claims.csv" and len(subset) > 0:
                    per_facility_claim_ids[facility_id] = set(subset["Id"].values)

                data_handler.write_facility_table(
                    subset, facilities_dir, facility_id, table_name
                )

            del source_df
            progress.update(task_b, advance=1)

        del per_facility_claim_ids
        elapsed_b = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Phase B complete: streamed {len(phase_b_tables)} tables "
            f"[dim]({elapsed_b:.1f}s)[/dim]"
        )

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
