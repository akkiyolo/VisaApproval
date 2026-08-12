import sys

import pandas as pd
import mlflow

from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import f1_score

from us_visa.constants import (
    CURRENT_YEAR,
    TARGET_COLUMN
)

from us_visa.entity.config_entity import (
    ModelEvaluationConfig
)

from us_visa.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)

from us_visa.entity.s3_estimator import (
    USvisaEstimator
)

from us_visa.entity.estimator import (
    TargetValueMapping
)

from us_visa.exception import (
    USvisaException
)

from us_visa.logger import logging

from us_visa.utils.mlflow_utils import (
    log_metrics,
    log_tags
)


@dataclass
class EvaluateModelResponse:

    trained_model_f1_score: float

    best_model_f1_score: Optional[float]

    is_model_accepted: bool

    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):

        try:

            self.model_eval_config = (
                model_eval_config
            )

            self.data_ingestion_artifact = (
                data_ingestion_artifact
            )

            self.model_trainer_artifact = (
                model_trainer_artifact
            )

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    def get_best_model(
        self
    ) -> Optional[USvisaEstimator]:
        """
        Get the currently deployed model
        from AWS S3.
        """

        try:

            bucket_name = (
                self.model_eval_config
                .bucket_name
            )

            model_path = (
                self.model_eval_config
                .s3_model_key_path
            )

            usvisa_estimator = (
                USvisaEstimator(
                    bucket_name=bucket_name,
                    model_path=model_path
                )
            )

            if usvisa_estimator.is_model_present(
                model_path=model_path
            ):

                return usvisa_estimator

            return None

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    def evaluate_model(
        self
    ) -> EvaluateModelResponse:
        """
        Compare the newly trained model
        against the currently deployed model.
        """

        try:

            # =====================================================
            # LOAD TEST DATA
            # =====================================================

            test_df = pd.read_csv(
                self.data_ingestion_artifact
                .test_file_path
            )

            # =====================================================
            # FEATURE ENGINEERING
            # =====================================================

            test_df["company_age"] = (
                CURRENT_YEAR
                - test_df["yr_of_estab"]
            )

            # =====================================================
            # X / Y
            # =====================================================

            x = test_df.drop(
                TARGET_COLUMN,
                axis=1
            )

            y = test_df[
                TARGET_COLUMN
            ]

            y = y.replace(
                TargetValueMapping()._asdict()
            )

            # =====================================================
            # NEW MODEL SCORE
            # =====================================================

            trained_model_f1_score = (
                self.model_trainer_artifact
                .metric_artifact
                .f1_score
            )

            # =====================================================
            # PRODUCTION MODEL
            # =====================================================

            best_model_f1_score = None

            best_model = (
                self.get_best_model()
            )

            if best_model is not None:

                y_hat_best_model = (
                    best_model.predict(x)
                )

                best_model_f1_score = (
                    f1_score(
                        y,
                        y_hat_best_model
                    )
                )

            # =====================================================
            # COMPARISON
            # =====================================================

            previous_score = (
                0
                if best_model_f1_score is None
                else best_model_f1_score
            )

            difference = (
                trained_model_f1_score
                - previous_score
            )

            is_model_accepted = (
                trained_model_f1_score
                > previous_score
            )

            result = EvaluateModelResponse(
                trained_model_f1_score=(
                    trained_model_f1_score
                ),
                best_model_f1_score=(
                    best_model_f1_score
                ),
                is_model_accepted=(
                    is_model_accepted
                ),
                difference=difference
            )

            # =====================================================
            # LOG TO MLFLOW
            # =====================================================

            log_metrics(
                {
                    "new_model_f1": (
                        trained_model_f1_score
                    ),
                    "production_model_f1": (
                        best_model_f1_score
                    ),
                    "f1_difference": (
                        difference
                    )
                }
            )

            log_tags(
                {
                    "model_status": (
                        "accepted"
                        if is_model_accepted
                        else "rejected"
                    ),
                    "production_model_available": (
                        "true"
                        if best_model is not None
                        else "false"
                    )
                }
            )

            logging.info(
                f"Model evaluation result: "
                f"{result}"
            )

            return result

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    def initiate_model_evaluation(
        self
    ) -> ModelEvaluationArtifact:
        """
        Initiates model evaluation.
        """

        try:

            evaluate_model_response = (
                self.evaluate_model()
            )

            s3_model_path = (
                self.model_eval_config
                .s3_model_key_path
            )

            model_evaluation_artifact = (
                ModelEvaluationArtifact(
                    is_model_accepted=(
                        evaluate_model_response
                        .is_model_accepted
                    ),
                    s3_model_path=s3_model_path,
                    trained_model_path=(
                        self.model_trainer_artifact
                        .trained_model_file_path
                    ),
                    changed_accuracy=(
                        evaluate_model_response
                        .difference
                    )
                )
            )

            logging.info(
                f"Model evaluation artifact: "
                f"{model_evaluation_artifact}"
            )

            return model_evaluation_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e