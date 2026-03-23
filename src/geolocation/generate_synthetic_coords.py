"""
Geração de coordenadas sintéticas anonimizadas para vítimas do DF.

Cada vítima recebe uma Região Administrativa (RA) sorteada proporcionalmente
à população da RA (peso_pop) e coordenadas lat/lon geradas por ruído
gaussiano em torno do centroide da RA, limitado ao raio_km da região.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.geolocation.ra_data import get_ra_dataframe

BASE_PATH: Path = Path(__file__).resolve().parent.parent.parent
INPUT_PATH: Path = BASE_PATH / "data" / "processed" / "df_vitimas_risco.parquet"
OUTPUT_PATH: Path = BASE_PATH / "data" / "processed" / "df_vitimas_geo.parquet"

# 1 grau de latitude ≈ 111 km (aproximação para conversão raio_km → graus)
_KM_PER_DEGREE: float = 111.0

RANDOM_SEED: int = 42


def _generate_point_in_ra(
    lat_ref: float,
    lon_ref: float,
    raio_km: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Gera um ponto com ruído gaussiano (σ = raio/3) limitado ao raio da RA."""
    sigma_deg = (raio_km / 3.0) / _KM_PER_DEGREE
    while True:
        lat = lat_ref + rng.normal(0, sigma_deg)
        lon = lon_ref + rng.normal(0, sigma_deg)
        # Verifica se o ponto está dentro do raio circular
        dist_km = np.sqrt(
            ((lat - lat_ref) * _KM_PER_DEGREE) ** 2
            + ((lon - lon_ref) * _KM_PER_DEGREE * np.cos(np.radians(lat_ref))) ** 2
        )
        if dist_km <= raio_km:
            return float(lat), float(lon)


def generate_synthetic_coords(df_vitimas: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Atribui RA e coordenadas sintéticas proporcionais à população de cada RA.

    Parameters
    ----------
    df_vitimas : DataFrame com coluna ``risco_prob`` (e opcionalmente ``id_vitima``).
                 Se None, carrega de ``INPUT_PATH``.

    Returns
    -------
    DataFrame acrescido de ``cod_ra``, ``nome_ra``, ``lat``, ``lon``.
    """
    if df_vitimas is None:
        df_vitimas = pd.read_parquet(INPUT_PATH)

    ra_df = get_ra_dataframe()
    n = len(df_vitimas)
    rng = np.random.default_rng(RANDOM_SEED)

    # Sortear RA para cada vítima proporcionalmente ao peso_pop
    ra_indices = rng.choice(
        len(ra_df),
        size=n,
        p=ra_df["peso_pop"].values,
    )

    cod_ra_arr = ra_df["cod_ra"].values[ra_indices]
    nome_ra_arr = ra_df["nome_ra"].values[ra_indices]

    lats: list[float] = []
    lons: list[float] = []

    for idx in ra_indices:
        row = ra_df.iloc[idx]
        lat, lon = _generate_point_in_ra(
            row["lat_ref"], row["lon_ref"], row["raio_km"], rng
        )
        lats.append(lat)
        lons.append(lon)

    df_out = df_vitimas.copy()
    df_out["cod_ra"] = cod_ra_arr
    df_out["nome_ra"] = nome_ra_arr
    df_out["lat"] = lats
    df_out["lon"] = lons

    return df_out


def save_geolocated_dataset() -> Path:
    """Executa geolocalização sintética e salva em parquet."""
    df = generate_synthetic_coords()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Dataset geolocalizado salvo em {OUTPUT_PATH} ({len(df)} registros)")

    # Verificação rápida de distribuição
    ra_df = get_ra_dataframe()
    dist_real = df["cod_ra"].value_counts(normalize=True).sort_index()
    dist_esperada = ra_df.set_index("cod_ra")["peso_pop"].sort_index()
    desvio_max = (dist_real - dist_esperada).abs().max()
    print(f"Desvio máximo vs. peso populacional: {desvio_max:.4f} ({desvio_max*100:.2f} p.p.)")

    return OUTPUT_PATH


if __name__ == "__main__":
    save_geolocated_dataset()
