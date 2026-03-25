import os

from pymongo import MongoClient

_client = None


def get_collection():
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    return _client["tech_challenge"]["finetuning"]
