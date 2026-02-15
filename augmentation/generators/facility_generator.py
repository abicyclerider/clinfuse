"""Generate synthetic facility metadata."""

from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker


class FacilityGenerator:
    """Generates realistic facility metadata."""

    FACILITY_TYPES = [
        "Hospital",
        "Medical Center",
        "Health Clinic",
        "Community Hospital",
        "Regional Medical Center",
    ]

    FACILITY_NAME_PATTERNS = [
        "{city} {type}",
        "{city} General {type}",
        "St. {name}'s {type}",
        "{name} Memorial {type}",
        "{city} Community {type}",
    ]

    def __init__(self, random_seed: int = 42):
        """
        Initialize facility generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        self.faker = Faker()
        Faker.seed(random_seed)

    def generate_facilities(
        self,
        num_facilities: int,
        organizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate facility metadata.

        Args:
            num_facilities: Number of facilities to generate
            organizations_df: Synthea organizations.csv for sampling locations

        Returns:
            DataFrame with columns: facility_id, name, address, city, state, zip, lat, lon, type
        """
        facilities = []

        for facility_id in range(1, num_facilities + 1):
            # Sample a random organization for geographic location
            org = organizations_df.sample(
                n=1, random_state=self.rng.integers(0, 2**31)
            ).iloc[0]

            # Generate facility name
            name = self._generate_facility_name()

            facilities.append(
                {
                    "facility_id": facility_id,
                    "name": name,
                    "address": org.get("ADDRESS", self.faker.street_address()),
                    "city": org.get("CITY", self.faker.city()),
                    "state": org.get("STATE", "MA"),
                    "zip": org.get("ZIP", self.faker.zipcode()),
                    "lat": org.get("LAT", self.faker.latitude()),
                    "lon": org.get("LON", self.faker.longitude()),
                    "type": self._select_facility_type(),
                }
            )

        return pd.DataFrame(facilities)

    def _generate_facility_name(self) -> str:
        """Generate a realistic facility name."""
        pattern = self.rng.choice(self.FACILITY_NAME_PATTERNS)
        facility_type = self._select_facility_type()

        if "{city}" in pattern:
            city = self.faker.city().replace("ville", "").replace("town", "")
            pattern = pattern.replace("{city}", city)

        if "{name}" in pattern:
            name = self.faker.last_name()
            pattern = pattern.replace("{name}", name)

        if "{type}" in pattern:
            pattern = pattern.replace("{type}", facility_type)

        return pattern

    def _select_facility_type(self) -> str:
        """Select a random facility type."""
        return self.rng.choice(self.FACILITY_TYPES)

    def save_facilities(self, facilities_df: pd.DataFrame, output_path: Path) -> None:
        """
        Save facilities metadata to CSV.

        Args:
            facilities_df: Facilities DataFrame
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        facilities_df.to_csv(output_path, index=False)
