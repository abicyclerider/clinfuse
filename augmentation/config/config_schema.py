"""Configuration schema using Pydantic for type-safe validation."""

from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class FacilityDistributionConfig(BaseModel):
    """Configuration for patient-to-facility distribution."""

    num_facilities: int = Field(
        default=10, ge=1, description="Total number of facilities to generate"
    )

    strategy: Literal["chronological_switching"] = Field(
        default="chronological_switching",
        description="Strategy for distributing encounters across facilities",
    )

    facility_count_weights: Dict[int, float] = Field(
        default={
            1: 0.40,  # 40% patients at 1 facility
            2: 0.30,  # 30% at 2 facilities
            3: 0.15,
            4: 0.10,
            5: 0.05,  # 5% at 5+ facilities
        },
        description="Probability weights for number of facilities per patient",
    )

    primary_facility_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Proportion of encounters at primary facility (for multi-facility patients)",
    )

    @field_validator("facility_count_weights")
    @classmethod
    def validate_weights_sum_to_one(cls, v: Dict[int, float]) -> Dict[int, float]:
        """Ensure facility count weights sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Facility count weights must sum to 1.0, got {total}")
        return v

    @field_validator("facility_count_weights")
    @classmethod
    def validate_positive_keys(cls, v: Dict[int, float]) -> Dict[int, float]:
        """Ensure all keys are positive integers."""
        if any(k < 1 for k in v.keys()):
            raise ValueError("Facility count keys must be positive integers")
        return v


class ErrorInjectionConfig(BaseModel):
    """Configuration for demographic error injection."""

    global_error_rate: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Probability that a patient record has at least one error",
    )

    multiple_errors_probability: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Probability of multiple errors given at least one error",
    )

    error_type_weights: Dict[str, float] = Field(
        default={
            "name_variation": 0.30,
            "address_error": 0.25,
            "date_variation": 0.15,
            "ssn_error": 0.10,
            "formatting_error": 0.20,
        },
        description="Relative weights for different error types",
    )

    @field_validator("error_type_weights")
    @classmethod
    def validate_weights_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure error type weights sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Error type weights must sum to 1.0, got {total}")
        return v


class PathConfig(BaseModel):
    """Configuration for input/output paths."""

    input_dir: Path = Field(description="Path to Synthea CSV directory")

    output_dir: Path = Field(description="Path to output directory for augmented data")

    @field_validator("input_dir")
    @classmethod
    def validate_input_exists(cls, v: Path) -> Path:
        """Ensure input directory exists."""
        if not v.exists():
            raise ValueError(f"Input directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Input path is not a directory: {v}")
        return v


class AugmentationConfig(BaseModel):
    """Master configuration for the augmentation system."""

    facility_distribution: FacilityDistributionConfig = Field(
        default_factory=FacilityDistributionConfig
    )

    error_injection: ErrorInjectionConfig = Field(default_factory=ErrorInjectionConfig)

    paths: PathConfig

    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    validate_output: bool = Field(
        default=True, description="Run validation checks after augmentation"
    )

    @model_validator(mode="after")
    def validate_facility_distribution(self):
        """Ensure num_facilities is sufficient for weights."""
        max_facilities_in_weights = max(
            self.facility_distribution.facility_count_weights.keys()
        )
        if self.facility_distribution.num_facilities < max_facilities_in_weights:
            raise ValueError(
                f"num_facilities ({self.facility_distribution.num_facilities}) "
                f"must be >= max facility count in weights ({max_facilities_in_weights})"
            )
        return self
