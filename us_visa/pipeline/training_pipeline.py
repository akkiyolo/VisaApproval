import sys

import mlflow

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import (
    DataIngestion
)

from us_visa.components.data_validation import (
    DataValidation
)

from us_visa.components.data_transformation import (
    DataTransformation
)

from us_visa.components.model_trainer import (
    ModelTrainer
)

from us_visa.components.model_evaluation import (
    ModelEvaluation
)

from us_visa.components.model_pusher import (
    ModelPusher
)

from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact
)

from us_visa.utils.mlflow_utils import (
    setup_mlflow,
    log_tags
)


class TrainPipeline:

    def __init__(self):

        self.data_ingestion_config = (
            DataIngestionConfig()
        )

        self.data_validation_config = (
            DataValidationConfig()
        )

        self.data_transformation_config = (
            DataTransformationConfig()
        )

        self.model_trainer_config = (
            ModelTrainerConfig()
        )

        self.model_evaluation_config = (
            ModelEvaluationConfig()
        )

        self.model_pusher_config = (
            ModelPusherConfig()
        )

    # ============================================================
    # DATA INGESTION
    # ============================================================

    def start_data_ingestion(
        self
    ) -> DataIngestionArtifact:

        try:

            logging.info(
                "Starting data ingestion"
            )

            data_ingestion = DataIngestion(
                data_ingestion_config=(
                    self.data_ingestion_config
                )
            )

            data_ingestion_artifact = (
                data_ingestion
                .initiate_data_ingestion()
            )

            logging.info(
                "Data ingestion completed"
            )

            return data_ingestion_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    # ============================================================
    # DATA VALIDATION
    # ============================================================

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:

            logging.info(
                "Starting data validation"
            )

            data_validation = DataValidation(
                data_ingestion_artifact=(
                    data_ingestion_artifact
                ),
                data_validation_config=(
                    self.data_validation_config
                )
            )

            data_validation_artifact = (
                data_validation
                .initiate_data_validation()
            )

            logging.info(
                "Data validation completed"
            )

            return data_validation_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    # ============================================================
    # DATA TRANSFORMATION
    # ============================================================

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:

            logging.info(
                "Starting data transformation"
            )

            data_transformation = (
                DataTransformation(
                    data_ingestion_artifact=(
                        data_ingestion_artifact
                    ),
                    data_transformation_config=(
                        self.data_transformation_config
                    ),
                    data_validation_artifact=(
                        data_validation_artifact
                    )
                )
            )

            data_transformation_artifact = (
                data_transformation
                .initiate_data_transformation()
            )

            logging.info(
                "Data transformation completed"
            )

            return data_transformation_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    # ============================================================
    # MODEL TRAINING
    # ============================================================

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:

            logging.info(
                "Starting model training"
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=(
                    data_transformation_artifact
                ),
                model_trainer_config=(
                    self.model_trainer_config
                )
            )

            model_trainer_artifact = (
                model_trainer
                .initiate_model_trainer()
            )

            logging.info(
                "Model training completed"
            )

            return model_trainer_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    # ============================================================
    # MODEL EVALUATION
    # ============================================================

    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:

        try:

            logging.info(
                "Starting model evaluation"
            )

            model_evaluation = ModelEvaluation(
                model_eval_config=(
                    self.model_evaluation_config
                ),
                data_ingestion_artifact=(
                    data_ingestion_artifact
                ),
                model_trainer_artifact=(
                    model_trainer_artifact
                )
            )

            model_evaluation_artifact = (
                model_evaluation
                .initiate_model_evaluation()
            )

            logging.info(
                "Model evaluation completed"
            )

            return model_evaluation_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    # ============================================================
    # MODEL PUSHER
    # ============================================================

    def start_model_pusher(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:

        try:

            logging.info(
                "Starting model pusher"
            )

            model_pusher = ModelPusher(
                model_evaluation_artifact=(
                    model_evaluation_artifact
                ),
                model_pusher_config=(
                    self.model_pusher_config
                )
            )

            model_pusher_artifact = (
                model_pusher
                .initiate_model_pusher()
            )

            logging.info(
                "Model pusher completed"
            )

            return model_pusher_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    # ============================================================
    # COMPLETE PIPELINE
    # ============================================================

    def run_pipeline(self) -> None:

        mlflow_run = None

        try:

            # =====================================================
            # SETUP MLFLOW
            # =====================================================

            setup_mlflow()

            logging.info(
                "MLflow setup completed"
            )

            # =====================================================
            # START MLFLOW RUN
            # =====================================================

            mlflow_run = mlflow.start_run(
                run_name="visa_approval_training"
            )

            log_tags(
                {
                    "project": "VisaApproval",
                    "pipeline": "training_pipeline",
                    "environment": "development"
                }
            )

            logging.info(
                f"MLflow run started: "
                f"{mlflow_run.info.run_id}"
            )

            # =====================================================
            # DATA INGESTION
            # =====================================================

            data_ingestion_artifact = (
                self.start_data_ingestion()
            )

            # =====================================================
            # DATA VALIDATION
            # =====================================================

            data_validation_artifact = (
                self.start_data_validation(
                    data_ingestion_artifact=(
                        data_ingestion_artifact
                    )
                )
            )

            # =====================================================
            # DATA TRANSFORMATION
            # =====================================================

            data_transformation_artifact = (
                self.start_data_transformation(
                    data_ingestion_artifact=(
                        data_ingestion_artifact
                    ),
                    data_validation_artifact=(
                        data_validation_artifact
                    )
                )
            )

            # =====================================================
            # MODEL TRAINING
            # =====================================================

            model_trainer_artifact = (
                self.start_model_trainer(
                    data_transformation_artifact=(
                        data_transformation_artifact
                    )
                )
            )

            # =====================================================
            # MODEL EVALUATION
            # =====================================================

            model_evaluation_artifact = (
                self.start_model_evaluation(
                    data_ingestion_artifact=(
                        data_ingestion_artifact
                    ),
                    model_trainer_artifact=(
                        model_trainer_artifact
                    )
                )
            )

            # =====================================================
            # MODEL ACCEPTANCE
            # =====================================================

            if not model_evaluation_artifact.is_model_accepted:

                log_tags(
                    {
                        "pipeline_status": "completed",
                        "model_pushed": "false"
                    }
                )

                logging.info(
                    "Model was not accepted."
                )

                return None

            # =====================================================
            # MODEL PUSHER
            # =====================================================

            model_pusher_artifact = (
                self.start_model_pusher(
                    model_evaluation_artifact=(
                        model_evaluation_artifact
                    )
                )
            )

            # =====================================================
            # SUCCESS
            # =====================================================

            log_tags(
                {
                    "pipeline_status": "completed",
                    "model_pushed": "true"
                }
            )

            logging.info(
                "Training pipeline completed successfully"
            )

            logging.info(
                f"Model pusher artifact: "
                f"{model_pusher_artifact}"
            )

            return None

        except Exception as e:

            logging.error(
                f"Training pipeline failed: {e}"
            )

            # -----------------------------------------------------
            # Mark MLflow run as failed
            # -----------------------------------------------------

            if mlflow_run is not None:

                try:

                    mlflow.set_tag(
                        "pipeline_status",
                        "failed"
                    )

                except Exception:
                    pass

            raise USvisaException(
                e,
                sys
            ) from e

        finally:

            # =====================================================
            # END MLFLOW RUN
            # =====================================================

            if mlflow.active_run() is not None:

                logging.info(
                    "Ending MLflow run"
                )

                mlflow.end_run()