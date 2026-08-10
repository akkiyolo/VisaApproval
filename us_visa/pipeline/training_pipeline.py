import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging

from us_visa.components.data_ingestion import DataIngestion
from us_visa.components.data_validation import DataValidation
from us_visa.components.data_transformation import DataTransformation
from us_visa.components.model_trainer import ModelTrainer

from us_visa.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible
        for starting the data ingestion component.
        """

        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )

            logging.info("Getting the data from MongoDB")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info("Got the train_set and test_set from MongoDB")

            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible
        for starting the data validation component.
        """

        logging.info(
            "Entered the start_data_validation method of TrainPipeline class"
        )

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible
        for starting the data transformation component.
        """

        try:
            logging.info(
                "Entered the start_data_transformation method of "
                "TrainPipeline class"
            )

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info("Performed the data transformation operation")

            logging.info(
                "Exited the start_data_transformation method of "
                "TrainPipeline class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible
        for starting the model training component.
        """

        try:
            logging.info(
                "Entered the start_model_trainer method of TrainPipeline class"
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )

            model_trainer_artifact = (
                model_trainer.initiate_model_trainer()
            )

            logging.info("Performed the model training operation")

            logging.info(
                "Exited the start_model_trainer method of TrainPipeline class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise USvisaException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible
        for running the complete training pipeline.
        """

        try:
            logging.info("Entered the run_pipeline method")

            # ---------------------------------------------------------
            # 1. Data Ingestion
            # ---------------------------------------------------------
            data_ingestion_artifact = self.start_data_ingestion()

            # ---------------------------------------------------------
            # 2. Data Validation
            # ---------------------------------------------------------
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # ---------------------------------------------------------
            # 3. Data Transformation
            # ---------------------------------------------------------
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )

            # ---------------------------------------------------------
            # 4. Model Training
            # ---------------------------------------------------------
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("Training pipeline completed successfully")

            logging.info("Model Trainer Artifact:")
            logging.info(model_trainer_artifact)

            logging.info("Exited the run_pipeline method")

        except Exception as e:
            raise USvisaException(e, sys) from e