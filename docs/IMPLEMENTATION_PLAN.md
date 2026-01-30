# Synthetic Medical Dataset for Entity Resolution Testing

## Overview

Create a synthetic medical dataset using Synthea to test entity resolution algorithms. The system will generate realistic patient data, inject controlled errors based on real-world healthcare data quality research, track ground truth for evaluation, and load data into PostgreSQL.

## Key Requirements

- **Data Source**: Synthea (MITRE's synthetic patient generator)
- **Output Format**: CSV (optimized for PostgreSQL loading)
- **Scale**: Start small (100-500 patients), scalable to thousands
- **Error Types**: Typos, address variations, date variations, missing fields
- **Hard Cases**: Twins, father/son (Jr/Sr), overlays
- **Ground Truth**: Complete tracking for algorithm evaluation
- **Use Case**: Fine-tuned LLM for medical record entity resolution

## Research-Based Error Distribution (Configurable)

**All error rates are configurable via `config/scale_config.yaml`**. Default values based on real healthcare systems research:

- Middle name discrepancies: 58.30% of mismatches
- SSN errors: 53.54%
- First name misspellings: 53.14%
- Last name misspellings: 33.62%
- Address variations: 35%
- Date variations: 25%
- Missing/null fields: 45%

Note: Percentages sum >100% because duplicates often have multiple errors.

**Implementation requirement**: Error rates must NEVER be hardcoded in Python scripts. All code should read from the configuration file to allow users to adjust distributions for different testing scenarios.

**Source**: Error rates from Bohensky MA et al. (2010) "Data Linkage: A powerful research tool with potential problems." BMC Health Services Research, validated in Christen P (2012) "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection."

## Architecture Components

### 1. Synthea Data Generation (Docker)

Use Docker Compose to run Synthea with proper configuration:
- Docker image: `synthetichealth/synthea:latest`
- Mount volumes for configuration and output
- Configure for CSV export via `synthea.properties`
- Generate base population (500 patients initially)
- Output files: `patients.csv`, `encounters.csv`, `conditions.csv`, etc.

**Docker Compose Configuration**:
```yaml
version: '3.8'

services:
  synthea:
    image: synthetichealth/synthea:latest
    container_name: synthea-generator
    volumes:
      - ./output/synthea_raw:/output
      - ./config/synthea.properties:/synthea.properties:ro
    command: [
      "-p", "500",              # Generate 500 patients
      "--exporter.csv.export", "true",
      "--exporter.fhir.export", "false",
      "--generate.only_alive_patients", "false",
      "Massachusetts"           # State for generation
    ]
    environment:
      - JAVA_OPTS=-Xmx4g
```

**Running Synthea**:
```bash
# Generate patients
docker-compose up synthea

# Output will be in ./output/synthea_raw/csv/
```

### 2. Base Population with Special Cases

Post-process Synthea output to create configurable special case cohorts:

- **Twin Cohort (configurable %, default 10%)**: Same last name, nearly identical first names, same DOB, sequential MRNs
- **Jr/Sr Cohort (configurable %, default 10%)**: Father/son with same name, 25-35 year age gap, similar addresses
- **Common Name Cohort (configurable %, default 10%)**: High-frequency names to increase collision probability
- **Overlay Records (configurable %, default 2%)**: Mixed demographics from different patients

**Important**: Percentages are configurable and can overlap. A patient can be both a twin AND have a common name. To test extreme scenarios, any percentage can be set to 1.0 (100%) to make all patients that case type.

### 3. Error Injection Engine

**Typo Generation**:
- **Keyboard proximity errors**: Adjacent key substitutions on QWERTY layout
  - 'e' can become 'w', 'r', 's', 'd' (surrounding keys)
  - Based on physical keyboard layout, not random selection
- **Character transposition**: Swap adjacent characters ('smith' → 'smiht')
- **Character doubling**: Duplicate a character ('smith' → 'smmith')
- **Character omission**: Drop a character ('smith' → 'smth')
- **OCR-style errors**: Shape confusions from scanned documents
  - O↔0 (letter O vs zero)
  - l↔1↔I (lowercase L vs one vs uppercase i)
  - S↔5 (letter S vs five)
  - m↔rn (letter m vs letter r+n)
  - cl↔d (letters c+l vs letter d)
- **Phonetic variations**: Sound-alike substitutions from dictation/transcription
  - ph↔f (Philip↔Filip)
  - rie↔ry (Marie↔Mary)
  - gail↔gayle, katherine↔kathryn
  - stephen↔steven, michael↔micheal
- **Position-based error distribution**: Research shows errors cluster at middle/end
  - Start position: 15% of errors
  - Middle position: 50% of errors
  - End position: 35% of errors
  - Not uniformly distributed across string length

**Address Variations**:
- Street type abbreviations (Street↔St, Avenue↔Ave)
- Directional variations (North↔N, Northwest↔NW)
- Unit number formats (Apt 4↔Apartment 4↔#4)
- ZIP+4 variations
- Typos in street names

**Date Variations**:
- Off-by-one day errors
- Month/day transposition
- Year typos
- Age calculation errors

**Missing Fields**:
- Middle name null (most common)
- Middle initial only
- Phone/email missing
- Suffix missing (critical for Jr/Sr confusion)

**Overlay Records** (Most Dangerous):
- Take demographics from Patient A
- Merge with medical history from Patient B
- Keep Patient A's MRN
- Result: Patient B's data under Patient A's identity

**Duplicate Strategy**:
- 50% no duplicates (clean records)
- 30% single duplicate (1-2 errors)
- 15% multiple duplicates (2-3 records, varying errors)
- 5% hard cases (twins, Jr/Sr, overlays)

### 4. Ground Truth Tracking

**Entity Cluster Format**:
```csv
entity_id,record_id,is_golden,error_types,difficulty,case_type
E001,P001,true,"",easy,standard
E001,P001_dup1,false,"middle_name,ssn",medium,standard
E001,P001_dup2,false,"first_name_typo,address_abbrev",medium,standard
E003,P003_twin1,false,"first_name_variation",hard,twin
E003,P003_twin2,false,"first_name_variation",hard,twin
```

**Metadata File** (JSON):
- Detailed error injection tracking per record
- Similarity scores
- Case type classification

### 5. Output Files

The system generates three primary CSV files:

**patients_with_duplicates.csv**:
- All patient records including originals and duplicates
- Columns: record_id, mrn, first_name, middle_name, last_name, suffix, date_of_birth, ssn, gender, street_address, city, state, zip, phone, email

**ground_truth.csv**:
- Entity cluster mappings for evaluation
- Columns: entity_id, record_id, is_golden, difficulty, case_type, error_types

**ground_truth_metadata.json**:
- Detailed error injection tracking per record
- Similarity scores between duplicates
- Case type classification
- Original values for all modified fields

## Project Structure

```
SyntheticMass/
├── config/
│   ├── synthea.properties          # Synthea CSV export configuration
│   └── scale_config.yaml           # Population size and error rates
│
├── docker-compose.yml              # Docker service for Synthea
│
├── scripts/
│   ├── 01_generate_base_population.py
│   ├── 02_create_duplicates.py     # Main orchestrator
│   │
│   ├── error_injection/
│   │   ├── __init__.py
│   │   ├── typo_generator.py       # Keyboard proximity, OCR, phonetic typos
│   │   ├── address_variations.py   # Address format variations
│   │   ├── date_variations.py      # DOB errors
│   │   ├── field_nullifier.py      # Missing fields
│   │   └── duplicate_generator.py  # Main error injection engine
│   │
│   ├── special_cases/
│   │   ├── __init__.py
│   │   ├── twin_generator.py
│   │   ├── jr_sr_generator.py
│   │   └── overlay_generator.py
│   │
│   └── validate_dataset.py         # Quality validation
│
├── output/                         # Generated data (gitignored)
│   ├── synthea_raw/csv/
│   ├── patients_with_duplicates.csv
│   ├── ground_truth.csv
│   └── ground_truth_metadata.json
│
├── tests/
│   ├── test_error_injection.py
│   ├── test_data_quality.py
│   └── test_ground_truth.py
│
├── docs/
│   ├── setup_guide.md
│   └── error_types.md
│
├── requirements.txt
└── README.md
```

## Implementation Steps

### Phase 1: Foundation
1. Create docker-compose.yml with Synthea service
2. Configure Synthea for CSV export via synthea.properties
3. Create Python virtual environment and install dependencies
4. Create project structure

### Phase 2: Base Data Generation
1. Generate base_size patients with Synthea via Docker (default 500)
2. Parse CSV output from synthea_raw/csv/
3. Apply special case transformations based on configured percentages:
   - Calculate how many patients for each case type
   - Randomly select patients and transform them
   - Allow overlaps (same patient can be multiple types)
4. Output base_population.csv with case_type metadata

### Phase 3: Error Injection
1. Implement configuration loader to read scale_config.yaml
2. Implement error generators (typos, address, dates, nulls) - all using config values:
   - **Keyboard proximity typos**: Adjacent key substitutions on QWERTY layout
   - **OCR-style errors**: Shape confusions (O↔0, l↔1, S↔5, m↔rn, cl↔d)
   - **Phonetic variations**: Sound-alike substitutions (ph↔f, rie↔ry, gail↔gayle)
   - **Position-based errors**: Weight errors toward middle/end of strings (research-validated)
   - **Transposition errors**: Swap adjacent characters
   - **Omission/duplication**: Drop or double characters
3. Implement address variation generator:
   - Abbreviations (Street↔St, Avenue↔Ave)
   - Directional variations (North↔N, Northwest↔NW)
   - Unit number formats (Apt 4↔#4↔Apartment 4)
   - ZIP+4 variations
4. Implement date variation generator:
   - Off-by-one day errors
   - Month/day transposition
   - Year typos
5. Implement field nullifier for missing data
6. Implement duplicate generator with error distribution sampling from config
7. Generate duplicates for each patient based on strategy
8. Handle special cases (twins, Jr/Sr, overlays)
9. Export patients_with_duplicates.csv

### Phase 4: Ground Truth & Export
1. Create entity cluster mappings
2. Export ground_truth.csv
3. Export metadata JSON with detailed error tracking
4. Validate completeness

### Phase 5: Validation & Quality Check
1. Run validation script
2. Check error distribution matches targets (±5%)
3. Verify ground truth covers 100% of entities
4. Manual review of 20-30 sample duplicate pairs
5. Verify hard cases (twins, Jr/Sr, overlays) are challenging
6. Generate summary statistics report

## Critical Files

1. **docker-compose.yml** - Synthea service orchestration
2. **scripts/error_injection/duplicate_generator.py** - Core engine orchestrating error injection
3. **scripts/02_create_duplicates.py** - Main entry point and orchestrator
4. **scripts/error_injection/typo_generator.py** - Name typo generation (most common error)
5. **scripts/validate_dataset.py** - Dataset quality validation and reporting
6. **config/scale_config.yaml** - Configuration for population size and error rates

## Testing Philosophy

We follow Sandy Metz's testing principles adapted for data generation algorithms.

### Core Principles

1. **Test the public interface** - Test what the module exposes, not internal helpers
2. **Test incoming messages** - Assert on return values and state changes
3. **Test outgoing command messages** - Verify side effects (files written, data modified)
4. **Don't test outgoing query messages** - Don't assert on internal method calls
5. **Don't mock what you don't own** - Use real pandas/yaml, not mocks
6. **Tests should survive refactoring** - Focus on behavior, not implementation details

### What TO Test

#### 1. Public Interfaces
```python
# ✅ Test the public API
def test_typo_generator_public_interface():
    generator = TypoGenerator(config)
    result = generator.generate_typo("Smith")
    assert result != "Smith"
    assert len(result) in [4, 5, 6]
```

#### 2. Algorithm Correctness
```python
# ✅ Test that error distributions match config
def test_error_distribution_matches_config():
    config = {'error_distribution': {'first_name_typo': 0.531}}
    duplicates = generate_duplicates(patients, config)

    typo_count = count_errors(duplicates, 'first_name_typo')
    expected = 0.531
    actual = typo_count / len(duplicates)

    assert abs(actual - expected) < 0.05  # Within ±5%

# ✅ Test that typos follow keyboard proximity (not random)
def test_typos_use_keyboard_proximity():
    results = [generate_typo("a") for _ in range(100)]
    nearby_keys = {'q', 'w', 's', 'z', 'a'}
    assert all(r in nearby_keys for r in results)
```

#### 3. Data Integrity
```python
# ✅ Test ground truth completeness
def test_ground_truth_covers_all_records():
    patients = generate_with_duplicates(base_data)
    ground_truth = generate_ground_truth(patients)

    all_record_ids = {p['record_id'] for p in patients}
    ground_truth_ids = {g['record_id'] for g in ground_truth}

    assert all_record_ids == ground_truth_ids
```

#### 4. Special Case Properties
```python
# ✅ Test twins have required properties
def test_twins_have_identical_dob():
    twins = generate_twin_pair(patient, config)
    assert twins[0]['date_of_birth'] == twins[1]['date_of_birth']
    assert twins[0]['last_name'] == twins[1]['last_name']
    assert similar_but_not_identical(
        twins[0]['first_name'],
        twins[1]['first_name']
    )
```

#### 5. Configuration Compliance
```python
# ✅ Test that config values are respected
def test_respects_configured_percentages():
    config = {'special_cases': {'twins': {'percentage': 1.0}}}
    result = generate_base_population(500, config)

    twin_count = count_twins(result)
    assert twin_count == 500
```

### What NOT to Test

#### ❌ Implementation Details
```python
# ❌ Don't test private methods directly
def test_internal_helper_function():
    result = _internal_select_random_patient()

# ❌ Don't test which data structure was used
def test_uses_dictionary_internally():
    assert isinstance(generator._cache, dict)
```

#### ❌ Internal Method Calls
```python
# ❌ Don't mock internal collaborators
def test_calls_helper_method(mocker):
    mock_helper = mocker.patch('module._internal_helper')
    generator.public_method()
    mock_helper.assert_called_once()
```

#### ❌ Third-Party Library Internals
```python
# ❌ Don't test pandas/yaml work correctly
def test_pandas_reads_csv():
    df = pd.read_csv("file.csv")
    assert isinstance(df, pd.DataFrame)
```

### Testing Strategy by Module

**Error Generators** (typo_generator.py, address_variations.py, etc.):
- ✅ Test public methods return valid transformations
- ✅ Test errors follow realistic patterns (keyboard proximity, abbreviations)
- ✅ Test configured rates are respected
- ❌ Don't test internal helper functions

**Special Case Generators** (twin_generator.py, jr_sr_generator.py):
- ✅ Test generated pairs have required properties
- ✅ Test relationships are preserved (same DOB, age gap)
- ❌ Don't test the random selection algorithm

**Duplicate Generator** (duplicate_generator.py):
- ✅ Test output has correct error distribution (±5%)
- ✅ Test ground truth is complete and accurate
- ✅ Test all CSV outputs are well-formed
- ❌ Don't test the order of operations

**Validation Script** (validate_dataset.py):
- ✅ Test it correctly identifies distribution mismatches
- ✅ Test it catches missing ground truth entries
- ❌ Don't test pandas aggregation internals

### Test Data Strategy

- Use **small, real config files** (not mocks)
- Use **small synthetic datasets** (10-20 patients for tests)
- Use **pytest fixtures** for common test data (sample patients, config)
- Use **property-based testing** where appropriate (hypothesis library)

### Testing Framework: pytest

All tests use **pytest**:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_error_injection.py
```

**Key pytest features:**
- **Fixtures**: Shared test data in `conftest.py`
- **Parametrize**: Test functions with multiple inputs
- **Approx**: Compare floats with tolerance (`pytest.approx(0.583, abs=0.05)`)
- **Hypothesis**: Property-based testing for algorithm invariants

### Test File Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_error_injection.py        # Error generator tests
├── test_special_cases.py          # Twin/Jr-Sr generator tests
├── test_duplicate_generator.py    # Main engine tests
├── test_data_quality.py           # Integration tests
└── test_ground_truth.py           # Ground truth validation tests
```

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
faker>=20.0.0
pyyaml>=6.0
pytest>=7.4.0
pytest-cov>=4.1.0          # Coverage reporting
hypothesis>=6.82.0          # Property-based testing
```

System requirements:
- Docker and Docker Compose
- Python 3.8+
- 4GB+ RAM recommended for Synthea generation

## Scalability

- Small (100-500): <5 minutes generation
- Medium (1K-5K): <30 minutes
- Large (10K+): <2 hours

Design uses:
- Batch processing for large populations
- Streaming CSV parsing (pandas chunksize)
- Multiprocessing for parallel duplicate generation
- PostgreSQL partitioning for very large datasets

## Configuration Files

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  # Synthea Patient Generator
  synthea:
    image: synthetichealth/synthea:latest
    container_name: synthea-generator
    volumes:
      - ./output/synthea_raw:/output
      - ./config/synthea.properties:/synthea.properties:ro
    command: [
      "-p", "500",
      "--exporter.csv.export", "true",
      "--exporter.fhir.export", "false",
      "--generate.only_alive_patients", "false",
      "Massachusetts"
    ]
    environment:
      - JAVA_OPTS=-Xmx4g
```

**config/scale_config.yaml**:
```yaml
# Population configuration
population:
  base_size: 500              # Number of base patients to generate with Synthea

# Special case configuration (percentages can overlap and sum > 1.0)
# Set any percentage to 1.0 to make ALL patients that case type
special_cases:
  twins:
    enabled: true
    percentage: 0.10          # 10% of population (50 patients as twins)

  jr_sr:
    enabled: true
    percentage: 0.10          # 10% of population (50 patients as Jr/Sr)

  common_names:
    enabled: true
    percentage: 0.10          # 10% of population (50 patients with common names)

  overlays:
    enabled: false            # Overlays are dangerous - usually created during error injection
    percentage: 0.02          # 2% if enabled

# Note: Percentages CAN overlap. A patient can be both a twin AND have a common name.
# To make ALL patients twins: set twins.percentage to 1.0
# To disable a case type: set enabled to false or percentage to 0.0

# Duplicate generation strategy
duplicates:
  no_duplicate_rate: 0.50     # 50% clean records with no duplicates
  single_duplicate_rate: 0.30 # 30% records with 1 duplicate
  multiple_duplicate_rate: 0.15  # 15% records with 2-3 duplicates
  hard_case_rate: 0.05        # 5% hard cases (twins, Jr/Sr, overlays)

# Error type probabilities (based on healthcare research)
# Note: Sum > 1.0 because duplicates often have multiple error types
error_distribution:
  middle_name_discrepancy: 0.583  # 58.3% - most common
  ssn_error: 0.535                # 53.5%
  first_name_typo: 0.531          # 53.1%
  last_name_typo: 0.336           # 33.6%
  address_variation: 0.35         # 35%
  date_variation: 0.25            # 25%
  missing_fields: 0.45            # 45%

# Advanced error injection settings (optional)
error_injection:
  # Typo error weights
  typo_types:
    keyboard_proximity: 0.35    # Adjacent key substitution
    transposition: 0.25         # Swap adjacent characters
    omission: 0.20              # Drop a character
    duplication: 0.10           # Double a character
    ocr_error: 0.10             # O↔0, l↔1, S↔5, m↔rn, cl↔d

  # Phonetic variation settings
  phonetic_variations:
    enabled: true
    probability: 0.15           # 15% of name errors are phonetic

  # Position-based error distribution (research-validated)
  error_position:
    start: 0.15                 # 15% errors at start of string
    middle: 0.50                # 50% errors in middle (most common)
    end: 0.35                   # 35% errors at end

  # Address variation weights
  address_types:
    abbreviation: 0.50          # Street ↔ St, Avenue ↔ Ave
    directional: 0.20           # North ↔ N
    unit_format: 0.20           # Apt 4 ↔ #4
    zip_format: 0.10            # ZIP+4 variations

  # Date variation weights
  date_types:
    off_by_one_day: 0.50        # ±1 day error
    month_day_swap: 0.30        # MM/DD ↔ DD/MM
    year_typo: 0.15             # Year digit error
    age_calculation: 0.05       # Incorrect age-derived DOB
```

## Verification

The validation script will check:
- Total record counts match expected
- Error distribution within 5% of targets
- Ground truth covers 100% of entities
- No orphaned records in ground truth
- Twin/Jr-Sr cases properly constructed
- All CSV files are well-formed and loadable
- Summary statistics report generated

## Success Criteria

1. Generate 500-patient base dataset with Synthea via Docker
2. Create duplicates with all four error types (typos, address, date, missing fields)
3. Error distribution matches research percentages (±5%)
4. Ground truth CSV complete and accurate for all entities
5. Metadata JSON contains detailed error tracking
6. Manual review confirms realistic and challenging duplicates
7. Hard cases (twins, Jr/Sr, overlays) validated as difficult to resolve
8. Complete dataset generated in <5 minutes
