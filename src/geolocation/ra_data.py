"""
Dados de referência das 31 Regiões Administrativas do Distrito Federal.

Fonte de população: PDAD/Codeplan 2021-2022 (estimativas arredondadas).
Coordenadas de referência: centroides aproximados de cada RA.
Raio (km): extensão estimada da área urbana para sorteio de pontos sintéticos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class RegiaoAdministrativa:
    cod_ra: int
    nome_ra: str
    populacao: int
    lat_ref: float
    lon_ref: float
    raio_km: float


# ── Tabela das 31 RAs do DF ────────────────────────────────────────────────
_RA_LIST: List[RegiaoAdministrativa] = [
    RegiaoAdministrativa(1,  "Plano Piloto",        220_000, -15.7939, -47.8828, 3.5),
    RegiaoAdministrativa(2,  "Gama",                135_000, -15.9868, -48.0625, 3.0),
    RegiaoAdministrativa(3,  "Taguatinga",          222_000, -15.8362, -48.0514, 2.5),
    RegiaoAdministrativa(4,  "Brazlândia",           55_000, -15.6757, -48.2067, 4.0),
    RegiaoAdministrativa(5,  "Sobradinho",           68_000, -15.6500, -47.7900, 3.0),
    RegiaoAdministrativa(6,  "Planaltina",          195_000, -15.6200, -47.6600, 5.0),
    RegiaoAdministrativa(7,  "Paranoá",              70_000, -15.7700, -47.7700, 3.0),
    RegiaoAdministrativa(8,  "Núcleo Bandeirante",   25_000, -15.8700, -47.9700, 1.5),
    RegiaoAdministrativa(9,  "Ceilândia",           490_000, -15.8200, -48.1100, 4.0),
    RegiaoAdministrativa(10, "Guará",               133_000, -15.8300, -47.9800, 2.0),
    RegiaoAdministrativa(11, "Cruzeiro",             33_000, -15.7900, -47.9400, 1.2),
    RegiaoAdministrativa(12, "Samambaia",           260_000, -15.8800, -48.0900, 3.0),
    RegiaoAdministrativa(13, "Santa Maria",         135_000, -16.0200, -48.0100, 3.0),
    RegiaoAdministrativa(14, "São Sebastião",       115_000, -15.9000, -47.7600, 3.5),
    RegiaoAdministrativa(15, "Recanto das Emas",    155_000, -15.9100, -48.0600, 2.5),
    RegiaoAdministrativa(16, "Lago Sul",             30_000, -15.8400, -47.8400, 3.0),
    RegiaoAdministrativa(17, "Riacho Fundo",         42_000, -15.8800, -48.0200, 2.0),
    RegiaoAdministrativa(18, "Lago Norte",           38_000, -15.7400, -47.8300, 2.5),
    RegiaoAdministrativa(19, "Candangolândia",       17_000, -15.8500, -47.9500, 1.0),
    RegiaoAdministrativa(20, "Águas Claras",        160_000, -15.8400, -48.0300, 2.0),
    RegiaoAdministrativa(21, "Riacho Fundo II",      55_000, -15.9000, -48.0500, 2.0),
    RegiaoAdministrativa(22, "Sudoeste/Octogonal",   55_000, -15.7900, -47.9200, 1.5),
    RegiaoAdministrativa(23, "Varjão",               10_000, -15.7100, -47.8700, 0.8),
    RegiaoAdministrativa(24, "Park Way",             22_000, -15.9100, -47.9600, 3.0),
    RegiaoAdministrativa(25, "SCIA/Estrutural",      40_000, -15.7800, -47.9900, 1.5),
    RegiaoAdministrativa(26, "Sobradinho II",       100_000, -15.6400, -47.8100, 3.5),
    RegiaoAdministrativa(27, "Jardim Botânico",      30_000, -15.8700, -47.8100, 3.0),
    RegiaoAdministrativa(28, "Itapoã",               70_000, -15.7500, -47.7700, 2.0),
    RegiaoAdministrativa(29, "SIA",                   2_000, -15.8100, -47.9500, 1.0),
    RegiaoAdministrativa(30, "Vicente Pires",        75_000, -15.8000, -48.0300, 2.0),
    RegiaoAdministrativa(31, "Fercal",               10_000, -15.5900, -47.8800, 3.0),
]


def get_ra_list() -> List[RegiaoAdministrativa]:
    """Retorna a lista imutável das 31 RAs."""
    return list(_RA_LIST)


def get_ra_dataframe() -> pd.DataFrame:
    """Retorna DataFrame com colunas cod_ra … raio_km e peso_pop calculado."""
    rows = [
        {
            "cod_ra": ra.cod_ra,
            "nome_ra": ra.nome_ra,
            "populacao": ra.populacao,
            "lat_ref": ra.lat_ref,
            "lon_ref": ra.lon_ref,
            "raio_km": ra.raio_km,
        }
        for ra in _RA_LIST
    ]
    df = pd.DataFrame(rows)
    df["peso_pop"] = df["populacao"] / df["populacao"].sum()
    return df
