import os

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(f"Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    client = MlflowClient()

    experiments = client.search_experiments()
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f" - {exp.name} (ID: {exp.experiment_id})")


if __name__ == "__main__":
    main()
