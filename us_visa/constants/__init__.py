import os
from datetime import date

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DATABASE
# ============================================================

DATABASE_NAME = os.getenv(
    "DATABASE_NAME",
    "US_VISA"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "visa_data"
)

MONGODB_URL_KEY = "CONNECTION_URL"


# ============================================================
# PIPELINE
# ============================================================

PIPELINE_NAME: str = "usvisa"

ARTIFACT_DIR: str = "artifact"


# ============================================================
# FILE NAMES
# ============================================================

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"

FILE_NAME: str = "usvisa.csv"

MODEL_FILE_NAME: str = "model.pkl"


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "case_status"

CURRENT_YEAR = date.today().year


# ============================================================
# PREPROCESSING
# ============================================================

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"


# ============================================================
# CONFIGURATION FILES
# ============================================================

SCHEMA_FILE_PATH = os.path.join(
    "config",
    "schema.yaml"
)


# ============================================================
# AWS
# ============================================================

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"

AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"

REGION_NAME = os.getenv(
    "AWS_DEFAULT_REGION",
    "us-east-1"
)


# ============================================================
# DATA INGESTION
# ============================================================

DATA_INGESTION_COLLECTION_NAME: str = "visa_data"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_INGESTED_DIR: str = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# ============================================================
# DATA VALIDATION
# ============================================================

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# ============================================================
# DATA TRANSFORMATION
# ============================================================

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


# ============================================================
# MODEL TRAINER
# ============================================================

MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

MODEL_TRAINER_MODEL_CONFIG_FILE_PATH = os.path.join(
    "config",
    "model.yaml"
)


# ============================================================
# MODEL EVALUATION
# ============================================================

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

MODEL_BUCKET_NAME = "akkiusvisa-model2026"

MODEL_PUSHER_S3_KEY = "model-registry"


# ============================================================
# MLflow / DAGsHub
# ============================================================

DAGSHUB_USERNAME = os.getenv(
    "DAGSHUB_USERNAME",
    "akkiyolo"
)

DAGSHUB_REPO = os.getenv(
    "DAGSHUB_REPO",
    "VisaApproval"
)

DAGSHUB_TOKEN = os.getenv(
    "DAGSHUB_TOKEN"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "VisaApproval"
)

MLFLOW_TRACKING_URI = (
    f"https://dagshub.com/"
    f"{DAGSHUB_USERNAME}/"
    f"{DAGSHUB_REPO}.mlflow"
)


# ============================================================
# APPLICATION
# ============================================================

APP_HOST = "0.0.0.0"

APP_PORT = 8080