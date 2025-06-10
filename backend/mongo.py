
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "env/.env")
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["ticker_teller"]

def insert_llm_analysis(data):
    db.llm_analysis.insert_one(data)

def get_latest_llm():
    return db.llm_analysis.find_one(sort=[("_id", -1)])
