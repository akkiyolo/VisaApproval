import sys
from typing import Protocol

import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from us_visa.exception import USvisaException
from us_visa.logger import logging


class Predictor(Protocol):
    """Structural type for any fitted sklearn-compatible estimator."""
    def predict(self, X) -> np.ndarray: ...


class TargetValueMapping:
    Certified: int = 0
    Denied: int = 1

    @classmethod
    def _asdict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_")}

    @classmethod
    def reverse_mapping(cls):
        mapping_response = cls._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: Predictor):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Function accepts raw inputs and then transforms raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data.
        At last it performs prediction on transformed features.
        """
        logging.info("Entered predict method of USvisaModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"