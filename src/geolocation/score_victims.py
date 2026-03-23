"""
Scores de risco de recorrência para todas as vítimas do dataset preprocessado.

Replica a codificação (LabelEncoder) usada no treinamento (optimize_ga.py) para
garantir compatibilidade com o modelo RandomForest serializado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET_COLUMN: str = "OUT_VEZES"
BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent
DATA_PATH: Path = BASE_PATH / "data" / "processed" / "df_preprocessed.parquet"
MODEL_PATH: Path = BASE_PATH / "models" / "random_forest.pkl"
OUTPUT_PATH: Path = BASE_PATH / "data" / "processed" / "df_vitimas_risco.parquet"


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Re-aplica a mesma codificação LabelEncoder usada no treinamento."""
    X_enc = X.copy()
    for col in X_enc.columns:
        col_data = X_enc[col].astype(str).replace("nan", np.nan)
        le = LabelEncoder()
        mode_vals = col_data.mode()
        fill_value = mode_vals.iloc[0] if len(mode_vals) > 0 else "0"
        X_enc[col] = le.fit_transform(col_data.fillna(fill_value))
    return X_enc


def score_victims() -> pd.DataFrame:
    """
    Carrega o dataset preprocessado, aplica o modelo treinado e retorna
    DataFrame original acrescido de ``id_vitima`` e ``risco_prob``.
    """
    import joblib

    df = pd.read_parquet(DATA_PATH)
    artifact: Dict[str, Any] = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features: list[str] = artifact["features"]

    X = df.drop(columns=[TARGET_COLUMN])

    # Garantir mesma ordem de features do modelo
    X = X[features]
    X_enc = _encode_features(X)

    probs = model.predict_proba(X_enc)[:, 1]

    df_out = df.copy()
    df_out["risco_prob"] = probs
    df_out["id_vitima"] = [f"V{i:06d}" for i in range(len(df_out))]

    return df_out


def save_scored_dataset() -> Path:
    """Executa scoring e salva o resultado em parquet."""
    df = score_victims()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Dataset com risco_prob salvo em {OUTPUT_PATH} ({len(df)} registros)")
    return OUTPUT_PATH


if __name__ == "__main__":
    save_scored_dataset()
