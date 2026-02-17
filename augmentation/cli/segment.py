"""Segment Synthea data into per-facility Parquet files (no error injection).

Reads Parquet input (from csv_to_parquet), assigns patients to facilities,
and writes CLEAN (un-errored) per-facility tables. This is the expensive step
that only needs to rerun when the assignment seed or input data changes.
"""

import json
import platform
import resource
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
from ..core import DataSplitter, FacilityAssigner
from ..core.streaming import stream_table_to_facilities
from ..generators import FacilityGenerator
from ..utils import DataHandler

console = Console()

_IS_MACOS = platform.system() == "Darwin"


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
    help="Path to Parquet directory (output of csv_to_parquet)",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to segmented output directory",
)
@click.option(
    "--assignment-seed",
    type=int,
    required=True,
    help="Random seed for facility assignment",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to configuration YAML file",
)
def main(
    input_dir: Path,
    output_dir: Path,
    assignment_seed: int,
    config_file: Path,
):
    """Segment Synthea data into per-facility Parquet files."""
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
    config_dict["random_seed"] = assignment_seed
    config = AugmentationConfig(**config_dict)

    facilities_dir = output_dir / "facilities"
    data_handler = DataHandler()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # ── Load patients + lightweight encounters + organizations ──
        task1 = progress.add_task("[cyan]Loading core tables...", total=None)
        patients_df = data_handler.load_table(input_dir, "patients")
        organizations_df = data_handler.load_table(input_dir, "organizations")

        # Stream encounters into a dict
        encounters_by_patient: dict[str, list[tuple[str, str]]] = {}
        num_encounters_loaded = 0
        for chunk in data_handler.stream_table_chunks(
            input_dir, "encounters", columns=["Id", "PATIENT", "START"]
        ):
            num_encounters_loaded += len(chunk)
            for enc_id, patient, start in zip(
                chunk["Id"], chunk["PATIENT"], chunk["START"]
            ):
                encounters_by_patient.setdefault(patient, []).append(
                    (str(start), enc_id)
                )
        for enc_list in encounters_by_patient.values():
            enc_list.sort()

        progress.update(task1, completed=True, total=1)
        console.print(
            f"[green]✓[/green] Loaded {len(patients_df)} patients, "
            f"{num_encounters_loaded} encounters (streamed)"
        )
        _log_memory("after loading tables")

        # ── Generate facility metadata ──
        task2 = progress.add_task("[cyan]Generating facility metadata...", total=None)
        facility_generator = FacilityGenerator(random_seed=assignment_seed)
        facilities_df = facility_generator.generate_facilities(
            config.facility_distribution.num_facilities,
            organizations_df,
        )
        progress.update(task2, completed=True, total=1)
        console.print(f"[green]✓[/green] Generated {len(facilities_df)} facilities")

        # ── Assign patients to facilities ──
        task3 = progress.add_task(
            "[cyan]Assigning patients to facilities...", total=None
        )
        facility_assigner = FacilityAssigner(
            config.facility_distribution, random_seed=assignment_seed
        )
        patient_facilities, encounter_facilities = (
            facility_assigner.assign_patients_to_facilities(
                patients_df["Id"].values, encounters_by_patient
            )
        )
        del encounters_by_patient
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

        enc_to_fac_map = encounter_facilities
        del encounter_facilities
        del patient_facilities
        _log_memory("after facility assignment")

        # ── Phase A: Write CLEAN patients + organizations per facility ──
        task_a = progress.add_task(
            "[cyan]Phase A: patients & encounters...", total=num_facilities + 1
        )
        data_splitter = DataSplitter()

        for facility_id in all_facilities:
            fac_patients_set = per_facility_patients[facility_id]

            # Filter patients (NO error injection — clean copy)
            fac_patients_df = data_splitter.filter_table_for_facility(
                "patients",
                patients_df,
                facility_id,
                fac_patients_set,
                set(),
            )

            data_handler.write_facility_table(
                fac_patients_df, facilities_dir, facility_id, "patients"
            )
            data_handler.write_facility_table(
                organizations_df, facilities_dir, facility_id, "organizations"
            )
            progress.update(task_a, advance=1)

        del patients_df, organizations_df

        # Stream encounters to per-facility Parquet
        progress.update(task_a, description="[cyan]Phase A: streaming encounters...")
        enc_rows, enc_chunks = stream_table_to_facilities(
            data_handler,
            data_splitter,
            input_dir,
            "encounters",
            facilities_dir,
            all_facilities,
            per_facility_patients,
            enc_to_fac_map=enc_to_fac_map,
        )
        progress.update(task_a, advance=1)

        elapsed_a = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Phase A complete: "
            f"{enc_rows:,} encounters in {enc_chunks} chunks "
            f"[dim]({elapsed_a:.1f}s)[/dim]"
        )
        _log_memory("after Phase A")

        # ── Phase B: stream remaining tables ──
        phase_b_tables = [
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
            "providers",
            "payers",
            "claims",
            "claims_transactions",
            "payer_transitions",
        ]

        task_b = progress.add_task(
            "[cyan]Phase B: streaming tables...", total=len(phase_b_tables)
        )

        per_facility_claim_ids: dict[int, set[str]] = {
            fid: set() for fid in all_facilities
        }

        # Pre-cache per-facility patient date ranges for payer_transitions
        payer_transitions_date_ranges: dict[int, pd.DataFrame] | None = None
        payer_transitions_path = input_dir / "payer_transitions.parquet"
        if payer_transitions_path.exists():
            payer_transitions_date_ranges = {}
            for facility_id in all_facilities:
                try:
                    enc_df = data_handler.read_facility_table(
                        facilities_dir, facility_id, "encounters"
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
            file_path = input_dir / f"{table_name}.parquet"
            if not file_path.exists():
                progress.update(task_b, advance=1)
                continue

            progress.update(task_b, description=f"[cyan]Phase B: {table_name}...")

            rows, chunks = stream_table_to_facilities(
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
                f"  [dim]{table_name}: {rows:,} rows in {chunks} chunk(s) "
                f"({t_table:.0f}s elapsed)[/dim]"
            )
            _log_memory(f"after {table_name}")
            progress.update(task_b, advance=1)

        del per_facility_claim_ids, payer_transitions_date_ranges, enc_to_fac_map
        elapsed_b = time.monotonic() - t0
        console.print(
            f"[green]✓[/green] Phase B complete: streamed {len(phase_b_tables)} tables "
            f"[dim]({elapsed_b:.1f}s)[/dim]"
        )
        _log_memory("after Phase B")

        # ── Write metadata ──
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        facilities_df.to_parquet(metadata_dir / "facilities.parquet", index=False)

        # Save assignment info for reproducibility
        with open(metadata_dir / "assignment.json", "w") as f:
            json.dump(
                {
                    "assignment_seed": assignment_seed,
                    "num_facilities": num_facilities,
                    "assignment_stats": assignment_stats,
                },
                f,
                indent=2,
            )

        elapsed_total = time.monotonic() - t0
        console.print(
            f"[bold green]✓ Segmentation complete[/bold green] → {output_dir} "
            f"[dim](total {elapsed_total:.1f}s)[/dim]"
        )


if __name__ == "__main__":
    main()
