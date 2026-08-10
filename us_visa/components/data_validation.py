import sys

import pandas as pd
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        """
        :param data_ingestion_artifact:
            Output reference of data ingestion artifact stage

        :param data_validation_config:
            Configuration for data validation
        """

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self._schema_config = read_yaml_file(
                file_path=SCHEMA_FILE_PATH
            )

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # 1. Validate number of columns
    # ------------------------------------------------------------------

    def validate_number_of_columns(
        self,
        dataframe: DataFrame
    ) -> bool:

        """
        Method Name:
            validate_number_of_columns

        Description:
            Validates whether the dataframe has the required
            number of columns.

        Returns:
            bool
        """

        try:

            status = (
                len(dataframe.columns)
                == len(self._schema_config["columns"])
            )

            logging.info(
                f"Is required column present: [{status}]"
            )

            return status

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # 2. Validate whether required columns exist
    # ------------------------------------------------------------------

    def is_column_exist(
        self,
        df: DataFrame
    ) -> bool:

        """
        Method Name:
            is_column_exist

        Description:
            Validates the existence of numerical and categorical
            columns defined in schema.yaml.

        Returns:
            bool
        """

        try:

            dataframe_columns = df.columns

            missing_numerical_columns = []
            missing_categorical_columns = []

            # Check numerical columns
            for column in self._schema_config["numerical_columns"]:

                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:

                logging.info(
                    f"Missing numerical columns: "
                    f"{missing_numerical_columns}"
                )

            # Check categorical columns
            for column in self._schema_config["categorical_columns"]:

                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:

                logging.info(
                    f"Missing categorical columns: "
                    f"{missing_categorical_columns}"
                )

            # If either list contains missing columns
            if (
                len(missing_numerical_columns) > 0
                or len(missing_categorical_columns) > 0
            ):
                return False

            return True

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # 3. Read CSV
    # ------------------------------------------------------------------

    @staticmethod
    def read_data(file_path) -> DataFrame:

        """
        Reads CSV file into pandas DataFrame.
        """

        try:

            return pd.read_csv(file_path)

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # 4. Detect dataset drift
    # ------------------------------------------------------------------

    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame
    ) -> bool:

        """
        Method Name:
            detect_dataset_drift

        Description:
            Detects dataset drift between reference and current
            datasets without using Evidently.

        Numerical columns:
            Compares mean and standard deviation.

        Categorical columns:
            Compares category distributions.

        Returns:
            True  -> Drift detected
            False -> No drift detected
        """

        try:

            logging.info(
                "Starting dataset drift detection..."
            )

            # ----------------------------------------------------------
            # Drift configuration
            # ----------------------------------------------------------

            MEAN_DRIFT_THRESHOLD = 0.20
            STD_DRIFT_THRESHOLD = 0.20
            CATEGORY_DRIFT_THRESHOLD = 0.20

            # ----------------------------------------------------------
            # Create drift report
            # ----------------------------------------------------------

            drift_report = {
                "data_drift": {
                    "data": {
                        "metrics": {
                            "n_features": len(reference_df.columns),
                            "n_drifted_features": 0,
                            "dataset_drift": False
                        },
                        "features": {}
                    }
                }
            }

            n_drifted_features = 0

            # ----------------------------------------------------------
            # Check every column
            # ----------------------------------------------------------

            for column in reference_df.columns:

                logging.info(
                    f"Checking drift for column: {column}"
                )

                reference_column = reference_df[column].dropna()
                current_column = current_df[column].dropna()

                # ======================================================
                # NUMERICAL COLUMN
                # ======================================================

                if pd.api.types.is_numeric_dtype(
                    reference_column
                ):

                    reference_mean = reference_column.mean()
                    current_mean = current_column.mean()

                    reference_std = reference_column.std()
                    current_std = current_column.std()

                    # ----------------------------------------------
                    # Mean change
                    # ----------------------------------------------

                    if reference_mean != 0:

                        mean_change = (
                            abs(current_mean - reference_mean)
                            / abs(reference_mean)
                        )

                    else:

                        mean_change = abs(
                            current_mean - reference_mean
                        )

                    # ----------------------------------------------
                    # Standard deviation change
                    # ----------------------------------------------

                    if reference_std != 0:

                        std_change = (
                            abs(current_std - reference_std)
                            / abs(reference_std)
                        )

                    else:

                        std_change = abs(
                            current_std - reference_std
                        )

                    # ----------------------------------------------
                    # Determine drift
                    # ----------------------------------------------

                    drift_detected = (
                        mean_change > MEAN_DRIFT_THRESHOLD
                        or
                        std_change > STD_DRIFT_THRESHOLD
                    )

                    # ----------------------------------------------
                    # Save report
                    # ----------------------------------------------

                    drift_report[
                        "data_drift"
                    ][
                        "data"
                    ][
                        "features"
                    ][column] = {

                        "type": "numerical",

                        "reference_mean": float(
                            reference_mean
                        ),

                        "current_mean": float(
                            current_mean
                        ),

                        "mean_change": float(
                            mean_change
                        ),

                        "reference_std": float(
                            reference_std
                        ),

                        "current_std": float(
                            current_std
                        ),

                        "std_change": float(
                            std_change
                        ),

                        "drift_detected": bool(
                            drift_detected
                        )
                    }

                # ======================================================
                # CATEGORICAL COLUMN
                # ======================================================

                else:

                    # ----------------------------------------------
                    # Calculate distributions
                    # ----------------------------------------------

                    reference_distribution = (
                        reference_column
                        .value_counts(normalize=True)
                    )

                    current_distribution = (
                        current_column
                        .value_counts(normalize=True)
                    )

                    # ----------------------------------------------
                    # Get all categories
                    # ----------------------------------------------

                    categories = set(
                        reference_distribution.index
                    ).union(
                        set(current_distribution.index)
                    )

                    max_distribution_difference = 0.0

                    # ----------------------------------------------
                    # Compare each category
                    # ----------------------------------------------

                    for category in categories:

                        reference_percentage = (
                            reference_distribution.get(
                                category,
                                0
                            )
                        )

                        current_percentage = (
                            current_distribution.get(
                                category,
                                0
                            )
                        )

                        difference = abs(
                            current_percentage
                            - reference_percentage
                        )

                        max_distribution_difference = max(
                            max_distribution_difference,
                            difference
                        )

                    # ----------------------------------------------
                    # Determine drift
                    # ----------------------------------------------

                    drift_detected = (
                        max_distribution_difference
                        > CATEGORY_DRIFT_THRESHOLD
                    )

                    # ----------------------------------------------
                    # Save report
                    # ----------------------------------------------

                    drift_report[
                        "data_drift"
                    ][
                        "data"
                    ][
                        "features"
                    ][column] = {

                        "type": "categorical",

                        "max_distribution_difference": float(
                            max_distribution_difference
                        ),

                        "drift_detected": bool(
                            drift_detected
                        )
                    }

                # ------------------------------------------------------
                # Count drifted features
                # ------------------------------------------------------

                if drift_detected:

                    n_drifted_features += 1

                    logging.info(
                        f"Drift detected in column: {column}"
                    )

                else:

                    logging.info(
                        f"No drift detected in column: {column}"
                    )

            # ----------------------------------------------------------
            # Dataset-level drift
            # ----------------------------------------------------------

            n_features = len(reference_df.columns)

            dataset_drift = (
                n_drifted_features > 0
            )

            drift_report[
                "data_drift"
            ][
                "data"
            ][
                "metrics"
            ][
                "n_drifted_features"
            ] = n_drifted_features

            drift_report[
                "data_drift"
            ][
                "data"
            ][
                "metrics"
            ][
                "dataset_drift"
            ] = dataset_drift

            # ----------------------------------------------------------
            # Save drift report
            # ----------------------------------------------------------

            write_yaml_file(
                file_path=(
                    self.data_validation_config
                    .drift_report_file_path
                ),
                content=drift_report
            )

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------

            logging.info(
                f"{n_drifted_features}/{n_features} "
                f"features have drift."
            )

            logging.info(
                f"Dataset drift status: {dataset_drift}"
            )

            return dataset_drift

        except Exception as e:
            raise USvisaException(e, sys) from e

    # ------------------------------------------------------------------
    # 5. Initiate data validation
    # ------------------------------------------------------------------

    def initiate_data_validation(
        self
    ) -> DataValidationArtifact:

        """
        Method Name:
            initiate_data_validation

        Description:
            Initiates the complete data validation component.

        Steps:
            1. Read train and test data
            2. Validate number of columns
            3. Validate required columns
            4. Detect dataset drift
            5. Create DataValidationArtifact

        Returns:
            DataValidationArtifact
        """

        try:

            validation_error_msg = ""

            logging.info(
                "Starting data validation"
            )

            # ----------------------------------------------------------
            # Read train and test data
            # ----------------------------------------------------------

            train_df = DataValidation.read_data(
                file_path=(
                    self.data_ingestion_artifact
                    .trained_file_path
                )
            )

            test_df = DataValidation.read_data(
                file_path=(
                    self.data_ingestion_artifact
                    .test_file_path
                )
            )

            logging.info(
                f"Training dataframe shape: {train_df.shape}"
            )

            logging.info(
                f"Testing dataframe shape: {test_df.shape}"
            )

            # ==========================================================
            # TRAIN DATA VALIDATION
            # ==========================================================

            status = self.validate_number_of_columns(
                dataframe=train_df
            )

            logging.info(
                "All required columns present in "
                f"training dataframe: {status}"
            )

            if not status:

                validation_error_msg += (
                    "Columns are missing in training dataframe. "
                )

            # ==========================================================
            # TEST DATA VALIDATION
            # ==========================================================

            status = self.validate_number_of_columns(
                dataframe=test_df
            )

            logging.info(
                "All required columns present in "
                f"testing dataframe: {status}"
            )

            if not status:

                validation_error_msg += (
                    "Columns are missing in test dataframe. "
                )

            # ==========================================================
            # CHECK COLUMN EXISTENCE IN TRAINING DATA
            # ==========================================================

            status = self.is_column_exist(
                df=train_df
            )

            if not status:

                validation_error_msg += (
                    "Columns are missing in training dataframe. "
                )

            # ==========================================================
            # CHECK COLUMN EXISTENCE IN TEST DATA
            # ==========================================================

            status = self.is_column_exist(
                df=test_df
            )

            if not status:

                validation_error_msg += (
                    "Columns are missing in test dataframe. "
                )

            # ==========================================================
            # CHECK VALIDATION STATUS
            # ==========================================================

            validation_status = (
                len(validation_error_msg) == 0
            )

            # ==========================================================
            # DATASET DRIFT
            # ==========================================================

            if validation_status:

                logging.info(
                    "Column validation successful."
                )

                drift_status = self.detect_dataset_drift(
                    reference_df=train_df,
                    current_df=test_df
                )

                if drift_status:

                    logging.info(
                        "Drift detected."
                    )

                    validation_error_msg = (
                        "Drift detected between "
                        "training and testing datasets."
                    )

                    validation_status = False

                else:

                    logging.info(
                        "Drift not detected."
                    )

                    validation_error_msg = (
                        "Data validation completed successfully. "
                        "No drift detected."
                    )

            else:

                logging.info(
                    f"Validation error: "
                    f"{validation_error_msg}"
                )

            # ==========================================================
            # CREATE DATA VALIDATION ARTIFACT
            # ==========================================================

            data_validation_artifact = DataValidationArtifact(

                validation_status=validation_status,

                message=validation_error_msg,

                drift_report_file_path=(
                    self.data_validation_config
                    .drift_report_file_path
                )
            )

            logging.info(
                f"Data validation artifact: "
                f"{data_validation_artifact}"
            )

            return data_validation_artifact

        except Exception as e:

            raise USvisaException(e, sys) from e