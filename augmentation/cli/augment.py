"""Command-line interface for data augmentation."""

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
    """Run the complete augmentation pipeline."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Task 1: Load CSV files
        task1 = progress.add_task("[cyan]Loading Synthea CSV files...", total=None)
        data_handler = DataHandler()
        synthea_csvs = data_handler.load_synthea_csvs(config.paths.input_dir)
        patients_df = synthea_csvs["patients.csv"]
        encounters_df = synthea_csvs["encounters.csv"]
        progress.update(task1, completed=True, total=1)
        console.print(
            f"[green]✓[/green] Loaded {len(patients_df)} patients, {len(encounters_df)} encounters"
        )

        # Task 2: Generate facility metadata
        task2 = progress.add_task("[cyan]Generating facility metadata...", total=None)
        facility_generator = FacilityGenerator(random_seed=config.random_seed)
        facilities_df = facility_generator.generate_facilities(
            config.facility_distribution.num_facilities,
            synthea_csvs["organizations.csv"],
        )
        progress.update(task2, completed=True, total=1)
        console.print(f"[green]✓[/green] Generated {len(facilities_df)} facilities")

        # Task 3: Assign patients to facilities
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

        # Task 4: Split CSV files by facility
        task4 = progress.add_task(
            "[cyan]Splitting CSV files by facility...", total=None
        )
        data_splitter = DataSplitter()
        facility_csvs = data_splitter.split_csvs_by_facility(
            synthea_csvs, patient_facilities, encounter_facilities
        )
        progress.update(task4, completed=True, total=1)
        console.print(
            f"[green]✓[/green] Split CSVs into {len(facility_csvs)} facility datasets"
        )

        # Task 5: Inject errors and track ground truth
        task5 = progress.add_task(
            "[cyan]Injecting demographic errors...", total=len(facility_csvs)
        )
        error_injector = ErrorInjector(
            config.error_injection, random_seed=config.random_seed
        )
        ground_truth_tracker = GroundTruthTracker()

        all_error_logs = []

        for facility_id in sorted(facility_csvs.keys()):
            patients_facility_df = facility_csvs[facility_id]["patients.csv"]

            # Inject errors
            errored_patients_df, error_log = error_injector.inject_errors_into_patients(
                patients_facility_df, facility_id
            )

            # Replace in facility CSVs
            facility_csvs[facility_id]["patients.csv"] = errored_patients_df

            # Track ground truth
            for _, patient in errored_patients_df.iterrows():
                patient_uuid = patient["Id"]
                num_encounters = len(
                    facility_csvs[facility_id]["encounters.csv"][
                        facility_csvs[facility_id]["encounters.csv"]["PATIENT"]
                        == patient_uuid
                    ]
                )

                # Get error types applied to this patient
                patient_errors = [
                    err["error_type"]
                    for err in error_log
                    if err["patient_uuid"] == patient_uuid
                ]

                ground_truth_tracker.add_patient_facility_mapping(
                    patient_uuid,
                    facility_id,
                    num_encounters,
                    patient.to_dict(),
                    patient_errors,
                )

            ground_truth_tracker.add_error_records(error_log)
            all_error_logs.extend(error_log)

            progress.update(task5, advance=1)

        error_stats = error_injector.get_error_statistics(all_error_logs)
        console.print(
            f"[green]✓[/green] Applied {error_stats['total_errors']} demographic errors"
        )

        # Task 6: Write output files
        task6 = progress.add_task(
            "[cyan]Writing output files...", total=len(facility_csvs) + 5
        )

        # Write facility data as Parquet
        for facility_id, csvs in facility_csvs.items():
            data_handler.write_facility_data(
                csvs, output_dir / "facilities", facility_id
            )
            progress.update(task6, advance=1)

        # Write metadata
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        facilities_df.to_parquet(metadata_dir / "facilities.parquet", index=False)
        progress.update(task6, advance=1)

        ground_truth_tracker.export_ground_truth(metadata_dir / "ground_truth.parquet")
        progress.update(task6, advance=1)

        ground_truth_tracker.export_error_log_jsonl(metadata_dir / "error_log.jsonl")
        progress.update(task6, advance=1)

        # Save run configuration
        with open(metadata_dir / "run_config.yaml", "w") as f:
            yaml.dump(config.model_dump(), f)
        progress.update(task6, advance=1)

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
        progress.update(task6, advance=1)

        console.print(f"[green]✓[/green] Output written to {output_dir}")

        # Task 7: Validation
        if validate:
            task7 = progress.add_task(
                "[cyan]Validating output...", total=len(facility_csvs)
            )

            validator = DataValidator()
            validation_errors = []

            for facility_id, csvs in facility_csvs.items():
                is_valid, errors = validator.validate_facility_csvs(csvs)
                if not is_valid:
                    validation_errors.extend(
                        [f"Facility {facility_id}: {err}" for err in errors]
                    )
                progress.update(task7, advance=1)

            if validation_errors:
                console.print("\n[yellow]⚠ Validation warnings:[/yellow]")
                for error in validation_errors[:10]:  # Show first 10
                    console.print(f"  • {error}")
            else:
                console.print("[green]✓[/green] All validations passed")

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
