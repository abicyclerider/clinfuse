#!/usr/bin/env python3
"""
Command-line interface for entity resolution pipeline.

Usage:
    python run_pipeline.py --config config/matching_config.yaml
    python run_pipeline.py --config config/matching_config.yaml --output-dir output/ --verbose
"""

import sys
from pathlib import Path

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import (
    load_config,
    run_entity_resolution_pipeline,
)


@click.command()
@click.option(
    "--config",
    default="config/matching_config.yaml",
    help="Path to configuration YAML file",
    type=click.Path(exists=True),
)
@click.option(
    "--output-dir",
    default="output/",
    help="Output directory for results",
    type=click.Path(),
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(config, output_dir, verbose):
    """
    Run entity resolution pipeline on augmented patient data.

    This pipeline matches patient records across multiple healthcare facilities
    to create consolidated "golden records" while handling demographic errors.
    """
    try:
        # Load configuration
        config_dict = load_config(config)

        # Run pipeline
        results = run_entity_resolution_pipeline(
            config_dict, output_dir=output_dir, verbose=verbose
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Entity Resolution Complete!")
        print("=" * 60)
        print(f"\n  Golden records:   {results['num_golden_records']}")
        print(f"  Expected:         {results.get('num_true_patients', 'N/A')}")
        print(f"  Precision:        {results['precision']:.4f}")
        print(f"  Recall:           {results['recall']:.4f}")
        print(f"  F1 Score:         {results['f1_score']:.4f}")
        print(f"\nResults saved to: {output_dir}")
        print("=" * 60 + "\n")

        # Exit with success
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
