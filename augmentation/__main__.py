"""Entry point for running augmentation as a module.

The augmentation pipeline is split into three CLI commands:
  python -m augmentation.cli.csv_to_parquet  -- Convert CSVs to Parquet
  python -m augmentation.cli.segment         -- Segment into facilities
  python -m augmentation.cli.inject_errors   -- Inject demographic errors
"""

import sys


def main():
    print(__doc__.strip())
    sys.exit(1)


if __name__ == "__main__":
    main()
