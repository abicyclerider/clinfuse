# SyntheticMass: Medical Entity Resolution Training Dataset Generator

Generate synthetic medical datasets with realistic, research-based errors for training and evaluating entity resolution algorithms.

## Overview

SyntheticMass creates realistic patient datasets with controlled duplicates and errors specifically designed for machine learning systems that need to learn medical record deduplication. The project generates three CSV files ready for ML training:

- `patients_with_duplicates.csv` - Patient records with injected duplicates and errors
- `ground_truth.csv` - Complete mapping of which records belong to the same entity
- `ground_truth_metadata.json` - Detailed tracking of original→modified field values

## Key Features

- **Research-Based Error Rates**: Default error distributions from healthcare data quality research
- **Challenging Test Cases**: Twins, Jr/Sr pairs, common names, and dangerous demographic overlays
- **100% Configurable**: All rates and percentages defined in `config/scale_config.yaml`
- **Complete Ground Truth**: Every record tracked with entity IDs and modification history
- **Realistic Errors**: Keyboard proximity-based typos, address variations, date format changes
- **Docker-Based**: Uses official Synthea Docker image - no source code cloning required

## Prerequisites

- **Docker**: For running Synthea synthetic patient generator
- **Python 3.8+**: For data processing and error injection
- **4GB+ RAM**: Required for Synthea generation
- **Git**: For cloning the repository

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SyntheticMass.git
cd SyntheticMass
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate Base Population with Synthea

```bash
# Run Synthea via Docker to generate 500 base patients
docker-compose up

# This will create output/synthea_raw/csv/ with Synthea-generated patient data
```

### 4. Generate Duplicates with Errors (Coming in Phase 2)

```bash
# Create base population with special cases
python scripts/01_generate_base_population.py

# Generate duplicates with error injection
python scripts/02_create_duplicates.py

# Validate output quality
python scripts/validate_dataset.py
```

## Project Structure

```
SyntheticMass/
├── config/
│   ├── scale_config.yaml          # ALL configuration parameters (error rates, percentages)
│   └── synthea.properties         # Synthea CSV export settings
│
├── scripts/
│   ├── 01_generate_base_population.py    # Parse Synthea + apply special cases
│   ├── 02_create_duplicates.py           # Main duplicate generation engine
│   ├── validate_dataset.py               # Data quality validation
│   │
│   ├── error_injection/                   # Error generator modules
│   │   ├── typo_generator.py             # Keyboard proximity typos
│   │   ├── address_generator.py          # Address variations
│   │   ├── date_generator.py             # Date format changes
│   │   └── null_generator.py             # Missing field injection
│   │
│   └── special_cases/                     # Hard test case generators
│       ├── twin_generator.py             # Identical twins
│       ├── jr_sr_generator.py            # Jr/Sr pairs
│       ├── common_name_generator.py      # High-collision names
│       └── overlay_generator.py          # Demographic overlays
│
├── output/                        # Generated data (gitignored)
│   ├── synthea_raw/csv/          # Synthea output
│   ├── patients_with_duplicates.csv
│   ├── ground_truth.csv
│   └── ground_truth_metadata.json
│
├── tests/                         # Pytest test suite
│   ├── conftest.py               # Shared fixtures
│   ├── test_error_injection.py
│   ├── test_special_cases.py
│   ├── test_duplicate_generator.py
│   ├── test_data_quality.py
│   └── test_ground_truth.py
│
├── docs/
│   └── IMPLEMENTATION_PLAN.md     # Complete technical specification
│
├── docker-compose.yml             # Synthea service configuration
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
├── README.md                      # This file
└── LICENSE                        # Project license
```

## Configuration

All generation parameters are defined in `config/scale_config.yaml`:

### Population Settings

```yaml
population:
  base_size: 500  # Number of patients to generate
```

### Special Cases (Hard Test Cases)

```yaml
special_cases:
  twins:
    enabled: true
    percentage: 0.10    # 10% of patients are twins
  jr_sr:
    enabled: true
    percentage: 0.10    # 10% are Jr/Sr pairs
  common_names:
    enabled: true
    percentage: 0.15    # 15% have high-collision names
  overlays:
    enabled: true
    percentage: 0.05    # 5% are dangerous overlays
```

**Note**: Percentages can overlap and sum to >1.0. Set to 1.0 to make ALL patients that type.

### Error Distribution (Research-Based)

```yaml
error_distribution:
  middle_name_discrepancy: 0.583  # 58.3%
  ssn_error: 0.535                # 53.5%
  first_name_typo: 0.531          # 53.1%
  last_name_typo: 0.336           # 33.6%
  address_variation: 0.35         # 35%
  date_variation: 0.25            # 25%
  missing_field: 0.45             # 45%
```

Percentages sum to >100% because duplicates have multiple errors.

### Duplicate Generation

```yaml
duplicates:
  no_duplicate: 0.25      # 25% remain unique
  single_duplicate: 0.50  # 50% get 1 duplicate
  multiple_duplicate: 0.20 # 20% get 2-3 duplicates
  hard_case: 0.05         # 5% get 4-6 duplicates
```

See `config/scale_config.yaml` for complete configuration with all parameters.

## Testing

The project uses **pytest** following Sandy Metz's testing principles:

```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
pytest

# Run with coverage report
pytest --cov=scripts --cov-report=html

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m algorithm         # Algorithm validation tests
pytest -m data_quality      # Data quality tests

# Run verbose with detailed output
pytest -v

# Run and show print statements
pytest -s
```

### Test Organization

- **Unit tests**: Test individual functions and algorithms (error distributions, keyboard proximity)
- **Integration tests**: Test complete workflows and data pipelines
- **Algorithm tests**: Validate error injection follows configuration (±5% tolerance)
- **Data quality tests**: Validate CSV schema, ground truth completeness

See `docs/IMPLEMENTATION_PLAN.md` for complete testing guidelines.

## Output Files

### patients_with_duplicates.csv

Patient records with injected duplicates and errors:

| record_id | entity_id | first_name | middle_name | last_name | dob | ssn | address_line1 | city | state | zip | phone |
|-----------|-----------|------------|-------------|-----------|-----|-----|---------------|------|-------|-----|-------|
| P001 | E001 | John | Michael | Smith | 1985-03-15 | 123-45-6789 | 123 Main St | Boston | MA | 02101 | 617-555-0101 |
| P001_dup1 | E001 | John | M. | Smith | 03/15/1985 | 123-45-6788 | 123 Main Street | Boston | MA | 02101 | 617-555-0101 |

### ground_truth.csv

Mapping of records to entities:

| entity_id | record_id | is_golden |
|-----------|-----------|-----------|
| E001 | P001 | true |
| E001 | P001_dup1 | false |

### ground_truth_metadata.json

Detailed modification tracking:

```json
{
  "P001_dup1": {
    "entity_id": "E001",
    "original_record_id": "P001",
    "modifications": {
      "middle_name": {
        "original": "Michael",
        "modified": "M.",
        "error_type": "middle_name_abbreviation"
      },
      "ssn": {
        "original": "123-45-6789",
        "modified": "123-45-6788",
        "error_type": "transposition"
      }
    }
  }
}
```

## Development Roadmap

### Phase 1: Foundation (Current)
- ✅ Directory structure
- ✅ Docker Compose configuration
- ✅ Configuration files (scale_config.yaml, synthea.properties)
- ✅ Python virtual environment
- ✅ Testing framework structure

### Phase 2: Base Data Generation
- 🔲 Run Synthea via Docker
- 🔲 Parse Synthea CSV output
- 🔲 Implement special case generators (twins, Jr/Sr, common names, overlays)
- 🔲 Output base_population.csv

### Phase 3: Duplicate Generation
- 🔲 Implement error injection modules (typos, address, date, nulls)
- 🔲 Main duplicate generation engine
- 🔲 Ground truth tracking
- 🔲 Output final CSV files

### Phase 4: Validation
- 🔲 Data quality validation script
- 🔲 Error distribution verification
- 🔲 Ground truth completeness check

### Phase 5: Documentation
- 🔲 Usage examples
- 🔲 Configuration guide
- 🔲 Integration guide for ML pipelines

## Implementation Notes

### What We ARE Doing
- ✅ Using Synthea via Docker image (`synthetichealth/synthea:latest`)
- ✅ CSV output only (no databases)
- ✅ Configuration-driven (all rates in `scale_config.yaml`)
- ✅ Research-based error rates
- ✅ Complete ground truth tracking

### What We're NOT Doing
- ❌ PostgreSQL database integration (out of scope)
- ❌ Hardcoded error rates (all configurable)
- ❌ Cloning Synthea source (Docker image only)
- ❌ Random typos (keyboard proximity-based)

## Contributing

This project follows specific implementation guidelines for AI-assisted development:

1. **Read first**: Review `docs/IMPLEMENTATION_PLAN.md` for complete specification
2. **Configuration-driven**: ALL parameters in `config/scale_config.yaml`
3. **Testing**: Follow Sandy Metz principles (test public interfaces, not implementation)
4. **No databases**: CSV output only, PostgreSQL excluded
5. **See claude.md**: Project-specific AI implementation guidelines

## Documentation

- **Complete Specification**: `docs/IMPLEMENTATION_PLAN.md`
- **Project Guidelines**: `claude.md` (AI implementation context)
- **Configuration Reference**: `config/scale_config.yaml` (with extensive comments)
- **Testing Guidelines**: `docs/IMPLEMENTATION_PLAN.md` (Phase 4 section)

## External Resources

- [Synthea Wiki](https://github.com/synthetichealth/synthea/wiki) - Synthetic patient generator documentation
- [Synthea Docker Hub](https://hub.docker.com/r/synthetichealth/synthea/) - Official Docker image
- [FHIR Standard](https://www.hl7.org/fhir/) - Healthcare data standard (not used, but informative)

## License

See `LICENSE` file for details.

## Support

For issues, questions, or contributions:
1. Check `docs/IMPLEMENTATION_PLAN.md` for detailed specifications
2. Review `config/scale_config.yaml` for configuration options
3. Open an issue on GitHub with detailed description

---

**Status**: Phase 1 Complete - Foundation established. Phase 2 (Base Data Generation) coming next.
