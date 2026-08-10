import sys
from typing import Tuple

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
        """
        Constructor for ModelTrainer.

        Parameters
        ----------
        data_transformation_artifact:
            Output artifact from the data transformation stage.

        model_trainer_config:
            Configuration required for model training.
        """

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(
        self,
        train: np.ndarray,
        test: np.ndarray
    ) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Uses neuro_mf to find the best model and evaluates it
        on the test dataset.

        Returns
        -------
        best_model_detail:
            Object containing information about the best model.

        metric_artifact:
            Classification metrics for the selected model.
        """

        try:
            logging.info(
                "Entered get_model_object_and_report method"
            )

            logging.info(
                "Using neuro_mf to get the best model object and report"
            )

            model_factory = ModelFactory(
                model_config_path=
                self.model_trainer_config.model_config_file_path
            )

            # ---------------------------------------------------------
            # Split train and test arrays into X and y
            # ---------------------------------------------------------

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

            # ---------------------------------------------------------
            # Find the best model
            # ---------------------------------------------------------

            best_model_detail = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy
            )

            model_obj = best_model_detail.best_model

            logging.info(
                f"Best model found: {model_obj}"
            )

            logging.info(
                f"Best model score: {best_model_detail.best_score}"
            )

            # ---------------------------------------------------------
            # Make predictions
            # ---------------------------------------------------------

            y_pred = model_obj.predict(x_test)

            # ---------------------------------------------------------
            # Calculate classification metrics
            # ---------------------------------------------------------

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

            # ---------------------------------------------------------
            # Create metric artifact
            # ---------------------------------------------------------

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            logging.info(
                "Created ClassificationMetricArtifact"
            )

            return best_model_detail, metric_artifact

        except Exception as e:

            raise USvisaException(e, sys) from e

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        Returns
        -------
        ModelTrainerArtifact
            Contains the trained model path and evaluation metrics.
        """

        logging.info(
            "Entered initiate_model_trainer method of ModelTrainer class"
        )

        try:

            # ---------------------------------------------------------
            # Load transformed train and test data
            # ---------------------------------------------------------

            train_arr = load_numpy_array_data(
                file_path=
                self.data_transformation_artifact
                .transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                file_path=
                self.data_transformation_artifact
                .transformed_test_file_path
            )

            logging.info(
                "Loaded transformed train and test arrays"
            )

            # ---------------------------------------------------------
            # Find best model and calculate metrics
            # ---------------------------------------------------------

            best_model_detail, metric_artifact = (
                self.get_model_object_and_report(
                    train=train_arr,
                    test=test_arr
                )
            )

            # ---------------------------------------------------------
            # Load preprocessing object
            # ---------------------------------------------------------

            preprocessing_obj = load_object(
                file_path=
                self.data_transformation_artifact
                .transformed_object_file_path
            )

            logging.info(
                "Loaded preprocessing object"
            )

            # ---------------------------------------------------------
            # Check whether model meets expected accuracy
            # ---------------------------------------------------------

            if (
                best_model_detail.best_score
                < self.model_trainer_config.expected_accuracy
            ):

                logging.info(
                    "No best model found with score more than base score"
                )

                raise Exception(
                    "No best model found with score more than base score"
                )

            logging.info(
                "Best model passed the expected accuracy threshold"
            )

            # ---------------------------------------------------------
            # Create US Visa model
            # ---------------------------------------------------------

            usvisa_model = USvisaModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model_detail.best_model
            )

            logging.info(
                "Created USvisaModel object with "
                "preprocessor and trained model"
            )

            # ---------------------------------------------------------
            # Save trained model
            # ---------------------------------------------------------

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=usvisa_model
            )

            logging.info(
                "Saved trained model successfully"
            )

            # ---------------------------------------------------------
            # Create ModelTrainerArtifact
            # ---------------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=
                self.model_trainer_config.trained_model_file_path,

                metric_artifact=metric_artifact
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            logging.info(
                "Exited initiate_model_trainer method"
            )

            return model_trainer_artifact

        except Exception as e:

            raise USvisaException(e, sys) from e