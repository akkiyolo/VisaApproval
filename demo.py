import os
from dotenv import load_dotenv
from us_visa.constants import MONGODB_URL_KEY

load_dotenv()

mongo_db_url = os.getenv(MONGODB_URL_KEY)

print(mongo_db_url)