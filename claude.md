# Claude AI Context Guide

## Project Overview

Generate synthetic medical datasets with realistic errors for testing entity resolution algorithms.

**Purpose**: Train and evaluate LLMs on medical record deduplication by providing patient data with controlled, research-based errors and complete ground truth.

**Output**: Three CSV files (patients, ground truth, metadata) ready for ML training.

---

## Critical Constraints (READ FIRST)

### ✅ What We ARE Doing
- **Synthea via Docker**: Use `synthetichealth/synthea:latest` image (don't clone repo)
- **CSV Output Only**: Generate three CSV files for downstream ML systems
- **Configuration-Driven**: ALL rates/percentages configurable via `config/scale_config.yaml`

### ❌ What We're NOT Doing
- ❌ **NO PostgreSQL** - Database integration is out of scope
- ❌ **NO Hardcoded Rates** - All error rates must read from config
- ❌ **NO Synthea Source** - Use Docker image, not git submodule

---

## Architecture (High-Level)

```
┌─────────────┐
│   Synthea   │  Generate 500 base patients
│   (Docker)  │  → output/synthea_raw/csv/
└──────┬──────┘
       │
       v
┌─────────────────────────────┐
│ 01_generate_base_population │  Parse Synthea CSV
│                             │  Apply special cases (twins, Jr/Sr)
└──────────┬──────────────────┘
           │
           v
┌──────────────────────┐
│ 02_create_duplicates │  Inject errors (typos, dates, missing)
│  (Error Injection)   │  Generate duplicates with research-based rates
└──────────┬───────────┘
           │
           v
┌────────────────────────────┐
│  Output (3 CSV files)      │
│  • patients_with_duplicates│
│  • ground_truth            │
│  • metadata.json           │
└────────────────────────────┘
```

---

## Configuration Philosophy

**Everything is configurable. Nothing is hardcoded.**

All rates/percentages/thresholds live in `config/scale_config.yaml`:

```yaml
population:
  base_size: 500

special_cases:
  twins:
    enabled: true
    percentage: 0.10    # Can set to 1.0 for ALL twins
  jr_sr:
    enabled: true
    percentage: 0.10

error_distribution:
  middle_name_discrepancy: 0.583
  first_name_typo: 0.531
  # ... all configurable
```

**Implementation rule**: All Python code must load config at startup and use those values. No magic numbers.

---

## Key Concepts

### 1. Special Cases (Hard Test Cases)
Transform Synthea patients into challenging scenarios:
- **Twins**: Identical DOB, similar names, same address
- **Jr/Sr**: Same name, 25-35 year gap, similar addresses
- **Common Names**: "John Smith" - high collision probability
- **Overlays**: Patient A's demographics + Patient B's history (dangerous!)

Percentages can overlap and sum > 1.0. Set to 1.0 to make ALL patients that type.

### 2. Error Injection (Research-Based)
Default error rates from healthcare research (all configurable):
- Middle name discrepancies: 58.3%
- SSN errors: 53.5%
- First name typos: 53.1%
- Last name typos: 33.6%
- Address variations: 35%
- Date variations: 25%
- Missing fields: 45%

Percentages sum >100% because duplicates have multiple errors.

### 3. Ground Truth Tracking
Every duplicate must be tracked:
- `entity_id`: E001, E002 (groups related records)
- `record_id`: P001, P001_dup1, P001_dup2
- `is_golden`: true only for original, false for duplicates
- 100% coverage required

---

## Common Pitfalls for AI Implementation

1. **Don't hardcode error rates** - Load from config/scale_config.yaml
2. **Don't add database code** - PostgreSQL is out of scope
3. **Don't clone Synthea** - Use Docker image `synthetichealth/synthea:latest`
4. **Don't make random typos** - Use keyboard proximity maps
5. **Don't skip ground truth** - Every record needs entity_id mapping
6. **Don't forget metadata** - Track original→modified values in JSON

---

## Project Structure

```
SyntheticMass/
├── docker-compose.yml              # Synthea service
├── config/
│   ├── synthea.properties
│   └── scale_config.yaml           # ALL configuration here
├── scripts/
│   ├── 01_generate_base_population.py
│   ├── 02_create_duplicates.py
│   ├── error_injection/            # Typos, address, date, nulls
│   ├── special_cases/              # Twins, Jr/Sr, overlays
│   └── validate_dataset.py
└── output/                         # Generated CSVs (gitignored)
    ├── patients_with_duplicates.csv
    ├── ground_truth.csv
    └── ground_truth_metadata.json
```

---

## Testing Philosophy

We follow **Sandy Metz's testing principles** using **pytest**.

**Key points:**
- ✅ Test public interfaces and algorithm correctness (error distributions, keyboard proximity)
- ✅ Test data integrity (ground truth completeness, CSV validity)
- ❌ Don't test implementation details (private methods, data structures)
- ❌ Don't mock what you don't own (use real pandas/yaml)

**See `docs/IMPLEMENTATION_PLAN.md` for:**
- Complete testing guidelines with examples
- What to test vs what not to test
- Module-specific testing strategies
- pytest usage and fixtures

---

## Quick Start for Implementation

1. **Read the full spec**: `docs/IMPLEMENTATION_PLAN.md` (single source of truth)
2. **Start with foundation**:
   - Create directory structure
   - Write docker-compose.yml
   - Create config/scale_config.yaml with all parameters
3. **Build incrementally**:
   - Error generators first (typo_generator.py, etc.)
   - Special case generators (twin_generator.py, etc.)
   - Main orchestrator last (duplicate_generator.py)
4. **Test continuously**: Validate error distributions match config (±5%)

---

## Complete Documentation

- **Full Specification**: `docs/IMPLEMENTATION_PLAN.md` ← READ THIS for all details
- **Config Examples**: See IMPLEMENTATION_PLAN.md for complete scale_config.yaml
- **Phase Details**: See IMPLEMENTATION_PLAN.md for 5-phase implementation guide

---

## External Resources

- Synthea Wiki: https://github.com/synthetichealth/synthea/wiki
- Synthea Docker: https://hub.docker.com/r/synthetichealth/synthea/
