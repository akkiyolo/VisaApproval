import os
import sys
from typing import Any, Dict, Optional

import mlflow

from us_visa.constants import (
    DAGSHUB_REPO,
    DAGSHUB_TOKEN,
    DAGSHUB_USERNAME,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from us_visa.exception import USvisaException
from us_visa.logger import logging


def setup_mlflow() -> None:
    """
    Configure MLflow to use the DAGsHub remote tracking server.

    DAGsHub provides an MLflow-compatible tracking server at:

        https://dagshub.com/<username>/<repository>.mlflow

    Authentication is handled using:
        MLFLOW_TRACKING_USERNAME
        MLFLOW_TRACKING_PASSWORD
    """

    try:

        if not DAGSHUB_USERNAME:
            raise ValueError(
                "DAGSHUB_USERNAME is not configured."
            )

        if not DAGSHUB_TOKEN:
            raise ValueError(
                "DAGSHUB_TOKEN is not configured."
            )

        # --------------------------------------------------------
        # Configure MLflow authentication
        # --------------------------------------------------------

        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME

        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        # --------------------------------------------------------
        # Configure remote tracking URI
        # --------------------------------------------------------

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        # --------------------------------------------------------
        # Configure experiment
        # --------------------------------------------------------

        mlflow.set_experiment(
            MLFLOW_EXPERIMENT_NAME
        )

        logging.info(
            f"MLflow tracking URI: "
            f"{MLFLOW_TRACKING_URI}"
        )

        logging.info(
            f"MLflow experiment: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

        logging.info(
            f"DAGsHub repository: "
            f"{DAGSHUB_USERNAME}/{DAGSHUB_REPO}"
        )

    except Exception as e:

        raise USvisaException(
            e,
            sys
        ) from e


def log_params(
    params: Dict[str, Any]
) -> None:
    """
    Log multiple parameters to MLflow.
    """

    try:

        cleaned_params = {}

        for key, value in params.items():

            if value is None:
                continue

            # MLflow parameters should be simple values.
            cleaned_params[str(key)] = str(value)

        if cleaned_params:

            mlflow.log_params(
                cleaned_params
            )

    except Exception as e:

        raise USvisaException(
            e,
            sys
        ) from e


def log_metrics(
    metrics: Dict[str, float]
) -> None:
    """
    Log multiple metrics to MLflow.
    """

    try:

        cleaned_metrics = {}

        for key, value in metrics.items():

            if value is None:
                continue

            cleaned_metrics[str(key)] = float(value)

        if cleaned_metrics:

            mlflow.log_metrics(
                cleaned_metrics
            )

    except Exception as e:

        raise USvisaException(
            e,
            sys
        ) from e


def log_tags(
    tags: Dict[str, str]
) -> None:
    """
    Log tags to MLflow.
    """

    try:

        cleaned_tags = {}

        for key, value in tags.items():

            if value is None:
                continue

            cleaned_tags[str(key)] = str(value)

        if cleaned_tags:

            mlflow.set_tags(
                cleaned_tags
            )

    except Exception as e:

        raise USvisaException(
            e,
            sys
        ) from e


def log_artifact(
    file_path: str,
    artifact_path: Optional[str] = None
) -> None:
    """
    Log a local file as an MLflow artifact.
    """

    try:

        if not os.path.exists(file_path):

            logging.warning(
                f"Artifact does not exist: "
                f"{file_path}"
            )

            return

        mlflow.log_artifact(
            file_path,
            artifact_path=artifact_path
        )

    except Exception as e:

        raise USvisaException(
            e,
            sys
        ) from e