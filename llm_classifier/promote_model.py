#!/usr/bin/env python3
"""Model promotion gate using MLflow Model Registry.

Compares a challenger model's eval_f1 against the current champion.
Promotes only if challenger >= champion (or no champion exists yet).

Usage:
    python promote_model.py <history_db> <metrics_json> <output_json>
"""

import json
import sys

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "medgemma-entity-resolution"
ADAPTER_REPO = "abicyclerider/medgemma-4b-entity-resolution-classifier"
METRIC_KEY = "eval_f1"


def promote(history_db: str, metrics_json: str, output_json: str) -> dict:
    """Run promotion gate and return the decision."""
    # Read challenger metrics
    with open(metrics_json) as f:
        metrics = json.load(f)
    challenger_f1 = float(metrics[METRIC_KEY])

    # Connect to the local MLflow history database
    client = MlflowClient(tracking_uri=f"sqlite:///{history_db}")

    # Ensure registered model exists
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass  # already exists

    # Create a new model version with eval metrics as tags
    source = f"https://huggingface.co/{ADAPTER_REPO}"
    tags = {
        k: str(v)
        for k, v in metrics.items()
        if k.startswith("eval_")
    }
    mv = client.create_model_version(MODEL_NAME, source=source, tags=tags)
    challenger_version = int(mv.version)

    # Look up current champion
    champion_f1 = None
    champion_version = None
    try:
        champion_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_version = int(champion_mv.version)
        champion_f1 = float(champion_mv.tags.get(METRIC_KEY, "0"))
    except MlflowException:
        pass  # no champion yet

    # Decide: promote if no champion or challenger >= champion
    promoted = champion_f1 is None or challenger_f1 >= champion_f1

    if promoted:
        client.set_registered_model_alias(
            MODEL_NAME, "champion", str(challenger_version)
        )

    decision = {
        "promoted": promoted,
        "challenger_f1": challenger_f1,
        "champion_f1": champion_f1,
        "model_version": challenger_version,
        "champion_version": champion_version,
    }

    with open(output_json, "w") as f:
        json.dump(decision, f, indent=2)

    return decision


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <history_db> <metrics_json> <output_json>")
        sys.exit(1)

    history_db, metrics_json, output_json = sys.argv[1], sys.argv[2], sys.argv[3]
    decision = promote(history_db, metrics_json, output_json)

    status = "PROMOTED" if decision["promoted"] else "NOT promoted"
    print(f"  Challenger v{decision['model_version']}: {METRIC_KEY}={decision['challenger_f1']:.4f}")
    if decision["champion_version"] is not None:
        print(f"  Champion   v{decision['champion_version']}: {METRIC_KEY}={decision['champion_f1']:.4f}")
    else:
        print("  Champion:  (none — bootstrap)")
    print(f"  Decision:  {status}")


if __name__ == "__main__":
    main()
