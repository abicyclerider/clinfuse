# Synthea Runner

Simple Docker Compose setup for generating synthetic patient data using [Synthea](https://github.com/synthetichealth/synthea).

## Overview

This runner uses the [synthea-docker](https://github.com/mrreband/synthea-docker) submodule to run Synthea in a containerized environment, generating realistic synthetic medical records for testing and development.

## Prerequisites

- **Docker**: For running Synthea
- **Git**: For cloning repository and initializing submodules
- **4GB+ RAM**: Required for Synthea generation

## Quick Start

### 1. Initialize Submodule (if not already done)

```bash
# From repository root
git submodule update --init --recursive
```

### 2. Generate Synthetic Patients

```bash
# From synthea_runner directory
cd synthea_runner
docker compose up

# Output will be created in: ./output/synthea_raw/csv/
```

The docker-compose configuration:
- Generates **500 patients** from Massachusetts
- Uses fixed seed **12345** for reproducible output
- Allocates **4GB RAM** for Java heap
- Exports **CSV format only** (FHIR disabled)

## Generated Output

Synthea produces CSV files in `./output/synthea_raw/csv/`:

### Core Files
- `patients.csv` - Patient demographics (name, DOB, SSN, address, etc.)
- `encounters.csv` - Medical visits and encounters
- `conditions.csv` - Diagnoses and medical conditions
- `medications.csv` - Prescriptions and medication orders
- `observations.csv` - Lab results and vital signs
- `procedures.csv` - Medical procedures and surgeries

### Additional Files
- `allergies.csv` - Patient allergies
- `careplans.csv` - Care plans and treatment protocols
- `devices.csv` - Medical devices
- `imaging_studies.csv` - Radiology and imaging data
- `immunizations.csv` - Vaccination records
- `organizations.csv` - Healthcare organizations
- `payers.csv` - Insurance payers
- `providers.csv` - Healthcare providers

## Configuration

### Modify Generation Parameters

Edit `docker-compose.yml` to change:
- Population size: `-p 500` (change 500 to desired number)
- Random seed: `-s 12345` (change for different patient set)
- State/location: `Massachusetts` (change to other US state)
- Memory: `JAVA_OPTS=-Xmx4g` (increase for larger populations)

### Synthea Properties

The `config/synthea.properties` file customizes:
- CSV export settings
- Socioeconomic distributions
- Demographics and population characteristics
- Clinical modules and conditions

See [Synthea Configuration Guide](https://github.com/synthetichealth/synthea/wiki/Common-Configuration) for all options.

## Directory Structure

```
synthea_runner/
├── docker-compose.yml             # Docker Compose service definition
├── config/
│   └── synthea.properties         # Synthea configuration
├── synthea-docker/                # Synthea Docker submodule
└── output/                        # Generated data (gitignored)
    └── synthea_raw/csv/           # Synthea CSV output
```

## Documentation

- [Synthea Wiki](https://github.com/synthetichealth/synthea/wiki) - Complete documentation
- [CSV Data Dictionary](https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary) - Schema reference
- [Common Configuration](https://github.com/synthetichealth/synthea/wiki/Common-Configuration) - Configuration guide
- [synthea-docker Repo](https://github.com/mrreband/synthea-docker) - Docker wrapper documentation

## Troubleshooting

### Submodule not initialized
```bash
# From repository root
git submodule update --init --recursive
```

### Out of memory errors
Increase heap size in `docker-compose.yml`:
```yaml
environment:
  - JAVA_OPTS=-Xmx8g  # Increase from 4g to 8g
```

### Permission errors on output directory
```bash
chmod -R 755 output/
```

---

**Built with**: [Synthea](https://github.com/synthetichealth/synthea) | [synthea-docker](https://github.com/mrreband/synthea-docker)
