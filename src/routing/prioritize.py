"""
Seleção e priorização de vítimas para visitação semanal (Top N por risco).

Seleciona as vítimas de maior ``risco_prob`` até lotar a capacidade semanal
estimada e gera flags de rastreabilidade (``incluida_semana`` e ``lista_espera``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.routing.config import (
    AVERAGE_SPEED_KMH,
    BASES,
    NUM_TEAMS,
    WORKING_DAYS,
    max_visits_total_week,
)

BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent
INPUT_PATH: Path = BASE_PATH / "data" / "processed" / "df_vitimas_geo.parquet"
OUTPUT_PATH: Path = BASE_PATH / "data" / "processed" / "df_vitimas_priorizadas.parquet"

# Tempo médio estimado de deslocamento entre visitas dentro do DF (minutos).
# Usado apenas para estimar a capacidade; o GA calculará tempos reais.
_AVG_TRAVEL_MIN: float = 15.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância haversine em quilômetros entre dois pontos."""
    R = 6_371.0
    rlat1, rlat2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def estimate_avg_travel_minutes(df: pd.DataFrame) -> float:
    """Estima o tempo médio de deslocamento por amostragem de pares."""
    rng = np.random.default_rng(42)
    n_sample = min(500, len(df))
    sample = df.sample(n=n_sample, random_state=42)
    dists: list[float] = []
    indices = sample.index.tolist()
    for _ in range(1000):
        i, j = rng.choice(len(indices), size=2, replace=False)
        ri, rj = sample.loc[indices[i]], sample.loc[indices[j]]
        km = _haversine_km(ri["lat"], ri["lon"], rj["lat"], rj["lon"])
        dists.append(km)
    avg_km = float(np.mean(dists))
    avg_min = (avg_km / AVERAGE_SPEED_KMH) * 60.0
    return avg_min


def prioritize_victims(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Ordena vítimas por ``risco_prob`` decrescente e marca Top N para visita.

    Returns
    -------
    DataFrame com colunas adicionais:
      - ``posicao_ranking`` : posição no ranking de risco (1 = maior risco)
      - ``incluida_semana`` : True se selecionada para visita na semana
      - ``lista_espera``    : True se acima da capacidade, mas ainda de alto risco
    """
    if df is None:
        df = pd.read_parquet(INPUT_PATH)

    # Estimar tempo médio de deslocamento com dados reais
    avg_travel = estimate_avg_travel_minutes(df)
    capacity = max_visits_total_week(avg_travel)
    print(f"Tempo médio estimado de deslocamento: {avg_travel:.1f} min")
    print(f"Capacidade semanal estimada: {capacity} visitas "
          f"({NUM_TEAMS} equipes × {WORKING_DAYS} dias)")

    df_sorted = df.sort_values("risco_prob", ascending=False).reset_index(drop=True)
    df_sorted["posicao_ranking"] = range(1, len(df_sorted) + 1)
    df_sorted["incluida_semana"] = df_sorted["posicao_ranking"] <= capacity
    df_sorted["lista_espera"] = (
        (~df_sorted["incluida_semana"]) & (df_sorted["risco_prob"] >= 0.5)
    )

    n_incluidas = df_sorted["incluida_semana"].sum()
    n_espera = df_sorted["lista_espera"].sum()
    print(f"Vítimas incluídas na semana: {n_incluidas}")
    print(f"Vítimas em lista de espera (risco >= 0.5): {n_espera}")
    print(f"Vítimas fora de cobertura: {len(df_sorted) - n_incluidas - n_espera}")

    return df_sorted


def save_prioritized_dataset() -> Path:
    """Prioriza e salva resultado."""
    df = prioritize_victims()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nDataset priorizado salvo em {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    save_prioritized_dataset()
