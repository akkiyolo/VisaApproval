import os
import sys
from typing import Tuple

import mlflow
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from neuro_mf import ModelFactory

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object
)

from us_visa.utils.mlflow_utils import (
    log_artifact,
    log_metrics,
    log_params,
    log_tags
)

from us_visa.entity.config_entity import ModelTrainerConfig

from us_visa.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

from us_visa.entity.estimator import USvisaModel


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):

        self.data_transformation_artifact = (
            data_transformation_artifact
        )

        self.model_trainer_config = (
            model_trainer_config
        )

    def get_model_object_and_report(
        self,
        train: np.ndarray,
        test: np.ndarray
    ) -> Tuple[
        object,
        ClassificationMetricArtifact
    ]:
        """
        Find the best model using neuro_mf,
        evaluate it and log the experiment to MLflow.
        """

        try:

            logging.info(
                "Entered get_model_object_and_report method"
            )

            logging.info(
                "Using neuro_mf to get the best model"
            )

            # =====================================================
            # MODEL FACTORY
            # =====================================================

            model_factory = ModelFactory(
                model_config_path=(
                    self.model_trainer_config
                    .model_config_file_path
                )
            )

            # =====================================================
            # SPLIT DATA
            # =====================================================

            x_train = train[:, :-1]

            y_train = train[:, -1]

            x_test = test[:, :-1]

            y_test = test[:, -1]

            logging.info(
                f"x_train shape: {x_train.shape}"
            )

            logging.info(
                f"y_train shape: {y_train.shape}"
            )

            logging.info(
                f"x_test shape: {x_test.shape}"
            )

            logging.info(
                f"y_test shape: {y_test.shape}"
            )

            # =====================================================
            # LOG DATASET INFORMATION
            # =====================================================

            log_params(
                {
                    "train_rows": x_train.shape[0],
                    "train_features": x_train.shape[1],
                    "test_rows": x_test.shape[0],
                    "test_features": x_test.shape[1],
                    "expected_accuracy": (
                        self.model_trainer_config
                        .expected_accuracy
                    )
                }
            )

            # =====================================================
            # FIND BEST MODEL
            # =====================================================

            best_model_detail = (
                model_factory.get_best_model(
                    X=x_train,
                    y=y_train,
                    base_accuracy=(
                        self.model_trainer_config
                        .expected_accuracy
                    )
                )
            )

            model_obj = (
                best_model_detail.best_model
            )

            best_score = (
                best_model_detail.best_score
            )

            logging.info(
                f"Best model found: {model_obj}"
            )

            logging.info(
                f"Best model score: {best_score}"
            )

            # =====================================================
            # MODEL INFORMATION
            # =====================================================

            model_class = (
                model_obj.__class__.__name__
            )

            log_tags(
                {
                    "model_type": model_class,
                    "framework": "scikit-learn",
                    "model_selection": "neuro_mf"
                }
            )

            # =====================================================
            # LOG MODEL HYPERPARAMETERS
            # =====================================================

            try:

                model_params = (
                    model_obj.get_params()
                )

                log_params(
                    model_params
                )

            except Exception as e:

                logging.warning(
                    "Could not extract model "
                    f"hyperparameters: {e}"
                )

            # =====================================================
            # PREDICTION
            # =====================================================

            y_pred = model_obj.predict(
                x_test
            )

            # =====================================================
            # METRICS
            # =====================================================

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            logging.info(
                f"Accuracy: {accuracy}"
            )

            logging.info(
                f"F1 Score: {f1}"
            )

            logging.info(
                f"Precision: {precision}"
            )

            logging.info(
                f"Recall: {recall}"
            )

            # =====================================================
            # LOG METRICS TO MLFLOW
            # =====================================================

            log_metrics(
                {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "best_cv_score": best_score
                }
            )

            # =====================================================
            # CLASSIFICATION METRIC ARTIFACT
            # =====================================================

            metric_artifact = (
                ClassificationMetricArtifact(
                    f1_score=f1,
                    precision_score=precision,
                    recall_score=recall
                )
            )

            logging.info(
                "Created ClassificationMetricArtifact"
            )

            return (
                best_model_detail,
                metric_artifact
            )

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifact:
        """
        Initiates model training and logs the resulting
        model artifacts to MLflow.
        """

        logging.info(
            "Entered initiate_model_trainer method"
        )

        try:

            # =====================================================
            # LOAD TRAIN DATA
            # =====================================================

            train_arr = load_numpy_array_data(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_train_file_path
                )
            )

            # =====================================================
            # LOAD TEST DATA
            # =====================================================

            test_arr = load_numpy_array_data(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_test_file_path
                )
            )

            logging.info(
                "Loaded transformed train and test arrays"
            )

            # =====================================================
            # TRAIN MODEL
            # =====================================================

            (
                best_model_detail,
                metric_artifact
            ) = self.get_model_object_and_report(
                train=train_arr,
                test=test_arr
            )

            # =====================================================
            # LOAD PREPROCESSING OBJECT
            # =====================================================

            preprocessing_obj = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            logging.info(
                "Loaded preprocessing object"
            )

            # =====================================================
            # ACCURACY CHECK
            # =====================================================

            if (
                best_model_detail.best_score
                < self.model_trainer_config
                .expected_accuracy
            ):

                logging.info(
                    "No best model found with score "
                    "above expected accuracy"
                )

                raise Exception(
                    "No best model found with score "
                    "above expected accuracy"
                )

            logging.info(
                "Best model passed accuracy threshold"
            )

            # =====================================================
            # CREATE US VISA MODEL
            # =====================================================

            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=(
                    best_model_detail.best_model
                )
            )

            logging.info(
                "Created USvisaModel object"
            )

            # =====================================================
            # SAVE DEPLOYMENT MODEL
            # =====================================================

            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=usvisa_model
            )

            logging.info(
                "Saved trained model successfully"
            )

            # =====================================================
            # LOG MODEL ARTIFACT TO MLFLOW
            # =====================================================

            trained_model_path = (
                self.model_trainer_config
                .trained_model_file_path
            )

            log_artifact(
                trained_model_path,
                artifact_path="deployment_model"
            )

            # =====================================================
            # LOG PREPROCESSING ARTIFACT
            # =====================================================

            preprocessing_path = (
                self.data_transformation_artifact
                .transformed_object_file_path
            )

            log_artifact(
                preprocessing_path,
                artifact_path="preprocessing"
            )

            # =====================================================
            # LOG MODEL CONFIGURATION
            # =====================================================

            model_config_path = (
                self.model_trainer_config
                .model_config_file_path
            )

            log_artifact(
                model_config_path,
                artifact_path="configuration"
            )

            # =====================================================
            # MODEL TRAINER ARTIFACT
            # =====================================================

            model_trainer_artifact = (
                ModelTrainerArtifact(
                    trained_model_file_path=(
                        self.model_trainer_config
                        .trained_model_file_path
                    ),
                    metric_artifact=metric_artifact
                )
            )

            logging.info(
                f"Model trainer artifact: "
                f"{model_trainer_artifact}"
            )

            logging.info(
                "Exited initiate_model_trainer method"
            )

            return model_trainer_artifact

        except Exception as e:

            raise USvisaException(
                e,
                sys
            ) from e