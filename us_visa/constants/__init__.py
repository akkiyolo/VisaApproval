import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_NAME = os.getenv("DATABASE_NAME", "US_VISA")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "visa_data")
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "usvisa"
ARTIFACT_DIR = "artifact"

MODEL_FIT_NAME = "model.pkl"