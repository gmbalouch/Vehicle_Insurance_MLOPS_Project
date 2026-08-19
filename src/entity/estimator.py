import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(),mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        try:
            logging.info("Starting prediction process.")

            logging.info(f"Prediction dataframe columns: {dataframe.columns.tolist()}")

            # IMPORTANT: inspect the saved preprocessor
            logging.info(
                f"Preprocessor feature names in_: "
                f"{getattr(self.preprocessing_object, 'feature_names_in_', 'NOT FOUND')}"
            )

            logging.info(
                f"Preprocessor n_features_in_: "
                f"{getattr(self.preprocessing_object, 'n_features_in_', 'NOT FOUND')}"
            )

            preprocessor = self.preprocessing_object.named_steps.get("Preprocessor")

            if preprocessor:
                logging.info(
                    f"ColumnTransformer feature_names_in_: "
                    f"{getattr(preprocessor, 'feature_names_in_', 'NOT FOUND')}"
                )

                logging.info(
                    f"ColumnTransformer transformers: "
                    f"{preprocessor.transformers_}"
                )

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info(
                f"Transformed feature shape: {transformed_feature.shape}"
            )

            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"



#The src/entity/estimator.py file defines a custom class, often referred to as a wrapper or model wrapper, which bundles your trained machine learning model together with its necessary pre-processing pipeline