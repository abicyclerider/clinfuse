"""
Golden record creation from matched patient records.

Merges matched records into consolidated "golden records" with conflict resolution.
"""

import logging
from collections import Counter
from typing import Any, List

import pandas as pd

logger = logging.getLogger(__name__)


def create_golden_records(
    matches: pd.Series, patient_data: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Create golden records by merging matched patient records.

    Args:
        matches: Boolean Series indicating which pairs are matches (MultiIndex)
        patient_data: DataFrame with all patient records
        config: Configuration with conflict resolution strategy

    Returns:
        DataFrame with golden records (one per unique patient)
    """
    # Build clusters of matched records
    clusters = build_match_clusters(matches)

    logger.info(f"Found {len(clusters)} matched clusters")

    # Find singleton records (not in any matched pair)
    matched_record_ids = set()
    for cluster in clusters:
        matched_record_ids.update(cluster)

    all_record_ids = set(patient_data["record_id"])
    singleton_ids = all_record_ids - matched_record_ids

    # Add singletons as their own clusters
    for rid in sorted(singleton_ids):
        clusters.append({rid})

    logger.info(
        f"Total clusters (including {len(singleton_ids)} singletons): {len(clusters)}"
    )

    # Create golden record for each cluster
    golden_records = []

    for cluster_id, record_ids in enumerate(clusters):
        cluster_records = patient_data[patient_data["record_id"].isin(record_ids)]

        golden_record = merge_cluster_records(
            cluster_records, cluster_id, config.get("golden_record", {})
        )

        golden_records.append(golden_record)

    golden_df = pd.DataFrame(golden_records)

    logger.info(f"Created {len(golden_df)} golden records")

    return golden_df


def build_match_clusters(matches: pd.Series) -> List[set]:
    """
    Build clusters of matching records using connected components.

    Args:
        matches: Boolean Series with MultiIndex (record_id_1, record_id_2)

    Returns:
        List of sets, each containing record_ids that belong together
    """
    # Get matched pairs
    matched_pairs = matches[matches].index.tolist()

    # Build adjacency graph
    from collections import defaultdict

    graph = defaultdict(set)

    for id1, id2 in matched_pairs:
        graph[id1].add(id2)
        graph[id2].add(id1)

    # Find connected components using DFS
    visited = set()
    clusters = []

    def dfs(node, cluster):
        visited.add(node)
        cluster.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for node in graph:
        if node not in visited:
            cluster = set()
            dfs(node, cluster)
            clusters.append(cluster)

    logger.debug(
        f"Built {len(clusters)} clusters from {len(matched_pairs)} matched pairs"
    )

    return clusters


def merge_cluster_records(
    cluster_records: pd.DataFrame, cluster_id: int, config: dict
) -> dict:
    """
    Merge all records in a cluster into a single golden record.

    Args:
        cluster_records: DataFrame with all records for this patient
        cluster_id: Unique identifier for this cluster
        config: Configuration with conflict resolution strategy

    Returns:
        Dictionary representing the golden record
    """
    strategy = config.get("conflict_resolution", "most_frequent")
    include_provenance = config.get("include_provenance", True)

    golden_record = {
        "golden_id": f"golden_{cluster_id:06d}",
        "num_facilities": cluster_records["facility_id"].nunique(),
        "num_records": len(cluster_records),
    }

    if include_provenance:
        golden_record["facilities"] = ",".join(
            sorted(cluster_records["facility_id"].unique())
        )
        golden_record["source_record_ids"] = ",".join(
            cluster_records["record_id"].tolist()
        )

    # Resolve each field
    fields_to_merge = [
        "first_name",
        "last_name",
        "maiden_name",
        "birthdate",
        "gender",
        "ssn",
        "address",
        "city",
        "state",
        "zip",
    ]

    for field in fields_to_merge:
        if field in cluster_records.columns:
            golden_record[field] = resolve_field_conflict(
                cluster_records[field].tolist(), strategy, field
            )

    return golden_record


def resolve_field_conflict(
    values: List[Any], strategy: str, field_name: str = None
) -> Any:
    """
    Resolve conflicting values for a field using specified strategy.

    Args:
        values: List of values from different records
        strategy: Conflict resolution strategy
        field_name: Name of the field (for field-specific rules)

    Returns:
        Resolved value
    """
    # Remove None/NaN values
    valid_values = [v for v in values if pd.notna(v) and v != ""]

    if not valid_values:
        return None

    if len(valid_values) == 1:
        return valid_values[0]

    if strategy == "most_frequent":
        # Democratic voting: most common value
        counter = Counter(valid_values)
        most_common = counter.most_common(1)[0][0]
        return most_common

    elif strategy == "most_recent":
        # Use last value (assumes records are temporally sorted)
        return valid_values[-1]

    elif strategy == "least_errors":
        # Prefer values that appear most consistently
        # For now, same as most_frequent
        return resolve_field_conflict(values, "most_frequent", field_name)

    elif strategy == "field_specific":
        # Apply field-specific resolution rules
        return apply_field_specific_rules(valid_values, field_name)

    else:
        logger.warning(
            f"Unknown conflict resolution strategy: {strategy}, using most_frequent"
        )
        return resolve_field_conflict(values, "most_frequent", field_name)


def apply_field_specific_rules(values: List[Any], field_name: str) -> Any:
    """
    Apply field-specific conflict resolution rules.

    Args:
        values: List of conflicting values
        field_name: Name of the field

    Returns:
        Resolved value
    """
    if field_name == "address":
        # Prefer non-abbreviated addresses (longer is usually better)
        return max(values, key=lambda x: len(str(x)))

    elif field_name == "ssn":
        # Prefer SSN with dashes (proper format)
        ssn_with_dashes = [v for v in values if "-" in str(v)]
        if ssn_with_dashes:
            return ssn_with_dashes[0]
        return values[0]

    elif field_name in ["first_name", "last_name"]:
        # Prefer title case (proper capitalization)
        title_case_values = [v for v in values if str(v).istitle()]
        if title_case_values:
            return title_case_values[0]
        return values[0]

    else:
        # Default: most frequent
        counter = Counter(values)
        return counter.most_common(1)[0][0]


def calculate_record_quality_score(record: pd.Series) -> float:
    """
    Calculate a quality score for a record based on completeness and consistency.

    Args:
        record: Series representing a patient record

    Returns:
        Quality score (0-1)
    """
    score = 0.0
    total_fields = 0

    # Check completeness
    important_fields = [
        "first_name",
        "last_name",
        "birthdate",
        "ssn",
        "address",
        "city",
        "state",
        "zip",
    ]

    for field in important_fields:
        if field in record.index:
            total_fields += 1
            if pd.notna(record[field]) and record[field] != "":
                score += 1

    # Check format consistency
    if "ssn" in record.index and pd.notna(record["ssn"]):
        # Prefer SSN with dashes
        if "-" in str(record["ssn"]):
            score += 0.5

    if "zip" in record.index and pd.notna(record["zip"]):
        # Prefer 5-digit ZIP
        if len(str(record["zip"])) == 5:
            score += 0.5

    return score / (total_fields + 1)  # +1 for consistency bonus


def merge_medical_histories(
    encounter_dfs: List[pd.DataFrame], record_id_mapping: dict
) -> pd.DataFrame:
    """
    Merge medical encounter histories from multiple facilities for a patient.

    Args:
        encounter_dfs: List of DataFrames with encounter records
        record_id_mapping: Mapping from facility record_id to golden_id

    Returns:
        Consolidated DataFrame with all encounters for the patient
    """
    # Tag each encounter with golden_id
    tagged_encounters = []

    for df in encounter_dfs:
        if "record_id" in df.columns or "PATIENT" in df.columns:
            df = df.copy()
            # Map to golden_id
            patient_col = "PATIENT" if "PATIENT" in df.columns else "record_id"
            df["golden_id"] = df[patient_col].map(record_id_mapping)
            tagged_encounters.append(df)

    if not tagged_encounters:
        return pd.DataFrame()

    # Combine all encounters
    all_encounters = pd.concat(tagged_encounters, ignore_index=True)

    # Sort by date
    if "START" in all_encounters.columns:
        all_encounters = all_encounters.sort_values("START")

    # Remove duplicates (same encounter at multiple facilities)
    # This requires encounter-level deduplication logic
    if "Id" in all_encounters.columns:
        all_encounters = all_encounters.drop_duplicates(subset=["Id"])

    return all_encounters


def validate_golden_record(
    golden_record: dict, required_fields: List[str] = None
) -> bool:
    """
    Validate that a golden record has all required fields.

    Args:
        golden_record: Dictionary representing a golden record
        required_fields: List of required field names

    Returns:
        True if valid, False otherwise
    """
    if required_fields is None:
        required_fields = ["golden_id", "first_name", "last_name", "birthdate"]

    for field in required_fields:
        if (
            field not in golden_record
            or pd.isna(golden_record[field])
            or golden_record[field] == ""
        ):
            logger.warning(
                f"Golden record {golden_record.get('golden_id')} missing required field: {field}"
            )
            return False

    return True
