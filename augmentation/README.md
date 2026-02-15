# Entity Resolution Data Augmentation System

A Python system to augment Synthea medical data with realistic demographic errors and multi-facility distribution for entity resolution testing.

## Overview

This system takes Synthea-generated patient data (571 patients, 18 CSV files) and:
1. Distributes patients across 1-5+ healthcare facilities
2. Chronologically splits encounters across facilities (simulating patient movement over time)
3. Injects realistic demographic errors (name variations, address typos, etc.)
4. Maintains UUID integrity and referential consistency
5. Produces ground truth mapping for entity resolution validation

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python -m augmentation.cli.augment \
  --input synthea_runner/output/synthea_raw/csv \
  --output output/augmented \
  --config augmentation/config/default_config.yaml
```

## Output Structure

```
output/augmented/run_[timestamp]/
├── facilities/
│   ├── facility_001/          # All 18 CSVs for facility 1
│   ├── facility_002/
│   └── ...
├── metadata/
│   ├── facilities.csv         # Facility information
│   ├── ground_truth.csv       # Patient→facility mapping
│   ├── error_log.jsonl        # Detailed error log
│   └── run_config.yaml        # Configuration snapshot
└── statistics/
    ├── distribution_report.json
    └── error_summary.json
```

## Configuration

See `config/default_config.yaml` for configuration options:

- **Facility Distribution**: Control how patients are distributed (40% at 1 facility, 30% at 2, etc.)
- **Error Injection**: Configure error rates and types (35% error rate by default)
- **Chronological Switching**: Primary facility gets 60% of encounters, remaining distributed to secondary facilities

## Architecture

### Core Components

- `config/`: Pydantic-based configuration validation
- `core/`: Core processing logic (facility assignment, CSV splitting, error injection)
- `errors/`: Pluggable error type implementations
- `generators/`: Facility metadata generation
- `utils/`: CSV handling and validation utilities
- `cli/`: Command-line interface

### Key Algorithms

**Facility Assignment**: Patients assigned to 1-5+ facilities using weighted distribution, encounters split chronologically with primary facility bias.

**CSV Splitting**: All 18 CSVs partitioned by facility maintaining UUID referential integrity. Encounter-driven tables follow encounter assignments, reference tables copied to all facilities.

**Error Injection**: Demographic fields modified with realistic errors (name variations, address typos, etc.) at configurable rates.

## Entity Resolution Testing

The system preserves patient UUIDs across facilities to enable ground truth validation, but entity resolution algorithms **must not use the UUID field** for matching. Match only on demographic fields (FIRST, LAST, BIRTHDATE, SSN, ADDRESS, etc.) to simulate realistic cross-facility patient matching scenarios.

Example:
- Facility 1: `{Id: "uuid-123", FIRST: "Alice", ADDRESS: "123 Main St"}`
- Facility 3: `{Id: "uuid-123", FIRST: "Alicia", ADDRESS: "123 Main Street Apt 5"}`

Same patient (same UUID) but different demographics due to errors.

## Development

Run tests:
```bash
pytest augmentation/tests/
```

Run with validation:
```bash
python -m augmentation.cli.augment --validate
```

## Future Enhancements

### Error Type Refinement

The current error types (name variations, address errors, date variations, SSN errors, formatting errors) are baseline implementations of common data quality issues. These can be refined and expanded based on research findings in `config/research/error_patterns.md`.

To integrate research findings:
1. Review error patterns identified in the research file
2. Update `config/default_config.yaml` with research-derived error rates and weights
3. Implement new error classes in `errors/` for any newly identified patterns
4. Re-run augmentation with updated configuration

The plugin-based error system makes it easy to add new error types without modifying core logic.

## License

MIT
