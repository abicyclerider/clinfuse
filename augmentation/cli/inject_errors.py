"""Inject demographic errors into segmented facility data.

Reads CLEAN per-facility patients from the segment step, applies ErrorInjector,
and writes errored patients + copies all other tables unchanged. This is the
fast step that only needs to rerun when error configuration changes.
"""

import json
import platform
import resource
import shutil
import time
from pathlib import Path

import click
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from ..config import AugmentationConfig
from ..core import ErrorInjector, GroundTruthTracker
from ..utils import DataHandler

console = Console()

_IS_MACOS = platform.system() == "Darwin"

# Tables that reference encounters via ENCOUNTER column
_ENCOUNTER_LINKED = [
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


def _read_column(fac_dir: Path, table: str, col: str) -> set:
    """Read a single column from a facility parquet file as a set."""
    p = fac_dir / f"{table}.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(p, columns=[col])
    return set(df[col].values)


def _validate_facility_lightweight(fac_dir: Path) -> list[str]:
    """Validate referential integrity loading only the columns needed."""
    errors: list[str] = []

    # patients.Id
    patients_path = fac_dir / "patients.parquet"
    if not patients_path.exists():
        return ["Missing patients table"]
    patient_ids = set(pd.read_parquet(patients_path, columns=["Id"])["Id"].values)
    if not patient_ids:
        errors.append("Critical table is empty: patients")

    # encounters.Id + encounters.PATIENT
    encounters_path = fac_dir / "encounters.parquet"
    if not encounters_path.exists():
        errors.append("Missing critical table: encounters")
        return errors
    enc_df = pd.read_parquet(encounters_path, columns=["Id", "PATIENT"])
    encounter_ids = set(enc_df["Id"].values)
    if not encounter_ids:
        errors.append("Critical table is empty: encounters")
    invalid = set(enc_df["PATIENT"].values) - patient_ids
    if invalid:
        errors.append(f"encounters has {len(invalid)} invalid PATIENT references")
    del enc_df

    # payer_transitions.PATIENT
    invalid = _read_column(fac_dir, "payer_transitions", "PATIENT") - patient_ids
    if invalid:
        errors.append(
            f"payer_transitions has {len(invalid)} invalid PATIENT references"
        )

    # encounter-linked tables
    for table in _ENCOUNTER_LINKED:
        p = fac_dir / f"{table}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["ENCOUNTER"])
        invalid = set(df["ENCOUNTER"].values) - encounter_ids
        if invalid:
            errors.append(f"{table} has {len(invalid)} invalid ENCOUNTER references")
        del df

    # claims.APPOINTMENTID
    claims_path = fac_dir / "claims.parquet"
    if claims_path.exists():
        claims_df = pd.read_parquet(claims_path, columns=["Id", "APPOINTMENTID"])
        claim_ids = set(claims_df["Id"].values)
        invalid = set(claims_df["APPOINTMENTID"].values) - encounter_ids
        if invalid:
            errors.append(f"claims has {len(invalid)} invalid APPOINTMENTID references")
        del claims_df

        # claims_transactions.CLAIMID
        invalid = _read_column(fac_dir, "claims_transactions", "CLAIMID") - claim_ids
        if invalid:
            errors.append(
                f"claims_transactions has {len(invalid)} invalid CLAIMID references"
            )

    return errors


def _log_memory(label: str) -> None:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = ru.ru_maxrss / (1024 * 1024) if _IS_MACOS else ru.ru_maxrss / 1024
    console.print(f"  [dim][mem] {label}: peak RSS = {rss_mb:.0f} MB[/dim]")


@click.command()
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to segmented output directory (output of segment)",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to augmented output directory",
)
@click.option(
    "--error-seed",
    type=int,
    required=True,
    help="Random seed for error injection",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to configuration YAML file",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Run validation after error injection",
)
@click.option(
    "--min-errors",
    type=int,
    default=None,
    help="Min errors when multiple errors apply",
)
@click.option(
    "--max-errors",
    type=int,
    default=None,
    help="Max errors per record",
)
@click.option(
    "--error-weights",
    type=str,
    default=None,
    help="JSON error_type_weights override",
)
def main(
    input_dir: Path,
    output_dir: Path,
    error_seed: int,
    config_file: Path,
    validate: bool,
    min_errors: int | None,
    max_errors: int | None,
    error_weights: str | None,
):
    """Inject demographic errors into segmented facility data."""
    t0 = time.monotonic()

    # Load configuration
    if config_file:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        default_config_path = (
            Path(__file__).parent.parent / "config" / "default_config.yaml"
        )
        with open(default_config_path, "r") as f:
            config_dict = yaml.safe_load(f)

    config_dict["paths"] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    config_dict["random_seed"] = error_seed
    ei = config_dict.setdefault("error_injection", {})
    if min_errors is not None:
        ei["min_errors"] = min_errors
    if max_errors is not None:
        ei["max_errors"] = max_errors
    if error_weights is not None:
        ei["error_type_weights"] = json.loads(error_weights)
    config = AugmentationConfig(**config_dict)

    input_facilities_dir = input_dir / "facilities"
    output_facilities_dir = output_dir / "facilities"
    data_handler = DataHandler()

    # Discover facility directories
    facility_dirs = sorted(input_facilities_dir.glob("facility_*"))
    facility_ids = []
    for d in facility_dirs:
        if d.is_dir():
            try:
                fid = int(d.name.split("_")[1])
                facility_ids.append(fid)
            except (IndexError, ValueError):
                continue

    num_facilities = len(facility_ids)
    console.print(f"[cyan]Found {num_facilities} facilities in[/cyan] {input_dir}")

    error_injector = ErrorInjector(config.error_injection, random_seed=error_seed)
    ground_truth_tracker = GroundTruthTracker()
    all_error_logs = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # ── Inject errors into patients, copy other tables ──
        task1 = progress.add_task("[cyan]Injecting errors...", total=num_facilities)

        for facility_id in facility_ids:
            src_facility_dir = input_facilities_dir / f"facility_{facility_id:03d}"
            dst_facility_dir = output_facilities_dir / f"facility_{facility_id:03d}"
            dst_facility_dir.mkdir(parents=True, exist_ok=True)

            # Read clean patients, inject errors
            patients_df = data_handler.read_facility_table(
                input_facilities_dir, facility_id, "patients"
            )
            errored_patients_df, error_log = error_injector.inject_errors_into_patients(
                patients_df, facility_id
            )

            # Read encounters for ground truth encounter counts
            try:
                encounters_df = data_handler.read_facility_table(
                    input_facilities_dir, facility_id, "encounters"
                )
                enc_counts: dict[str, int] = (
                    encounters_df.groupby("PATIENT").size().to_dict()
                )
                del encounters_df
            except FileNotFoundError:
                enc_counts = {}

            # Build patient→error_types lookup
            errors_by_patient: dict[str, list[str]] = {}
            for err in error_log:
                errors_by_patient.setdefault(err["patient_uuid"], []).append(
                    err["error_type"]
                )

            # Track ground truth
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

            # Write errored patients
            data_handler.write_facility_table(
                errored_patients_df, output_facilities_dir, facility_id, "patients"
            )

            # Copy all other parquet files unchanged
            for parquet_file in sorted(src_facility_dir.glob("*.parquet")):
                if parquet_file.name == "patients.parquet":
                    continue  # already written with errors
                shutil.copy2(parquet_file, dst_facility_dir / parquet_file.name)

            progress.update(task1, advance=1)

        error_stats = error_injector.get_error_statistics(all_error_logs)
        elapsed_inject = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Injected {len(all_error_logs)} errors across "
            f"{num_facilities} facilities [dim]({elapsed_inject:.1f}s)[/dim]"
        )
        _log_memory("after error injection")

        # ── Validation ──
        validation_errors = []
        if validate:
            task_v = progress.add_task("[cyan]Validating...", total=num_facilities)

            for facility_id in facility_ids:
                fac_dir = output_facilities_dir / f"facility_{facility_id:03d}"
                fac_errors = _validate_facility_lightweight(fac_dir)
                if fac_errors:
                    validation_errors.extend(
                        [f"Facility {facility_id}: {err}" for err in fac_errors]
                    )
                progress.update(task_v, advance=1)

            if validation_errors:
                console.print("\n[yellow]⚠ Validation warnings:[/yellow]")
                for error in validation_errors[:10]:
                    console.print(f"  • {error}")
            else:
                console.print("[green]✓[/green] All validations passed")

        # ── Write metadata ──
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Copy facilities.parquet from segment output
        src_facilities = input_dir / "metadata" / "facilities.parquet"
        if src_facilities.exists():
            shutil.copy2(src_facilities, metadata_dir / "facilities.parquet")

        # Copy confusable_pairs.parquet if it exists
        src_confusable = input_dir / "metadata" / "confusable_pairs.parquet"
        if src_confusable.exists():
            shutil.copy2(src_confusable, metadata_dir / "confusable_pairs.parquet")

        ground_truth_tracker.export_ground_truth(metadata_dir / "ground_truth.parquet")
        ground_truth_tracker.export_error_log_jsonl(metadata_dir / "error_log.jsonl")

        with open(metadata_dir / "run_config.yaml", "w") as f:
            yaml.dump(config.model_dump(), f)

        # Write statistics
        statistics_dir = output_dir / "statistics"
        ground_truth_stats = ground_truth_tracker.generate_statistics()
        combined_stats = {
            "errors": error_stats,
            "ground_truth": ground_truth_stats,
        }
        ground_truth_tracker.export_statistics_json(
            statistics_dir / "augmentation_report.json",
            additional_stats=combined_stats,
        )

        elapsed_total = time.monotonic() - t0
        console.print(
            f"[bold green]✓ Error injection complete[/bold green] → {output_dir} "
            f"[dim](total {elapsed_total:.1f}s)[/dim]"
        )

    # Display summary
    console.print(f"\n  Total Errors: {error_stats['total_errors']}")
    console.print(
        f"  Patients with Errors: {ground_truth_stats['patients_with_errors']}"
    )
    console.print(f"  Error Rate: {ground_truth_stats['error_rate']:.1%}")


if __name__ == "__main__":
    main()
