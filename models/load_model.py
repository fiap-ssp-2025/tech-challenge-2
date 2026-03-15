import joblib
from pathlib import Path

MODEL_PATH = Path("models/random_forest.pkl")


def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact
