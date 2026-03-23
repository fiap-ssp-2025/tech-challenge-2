"""
Parâmetros operacionais para o problema de roteirização (VRPTW) de visitas
a vítimas de violência doméstica no Distrito Federal.

Todas as constantes são centralizadas aqui para facilitar experimentação
no Algoritmo Genético e no notebook acadêmico.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ── Janela de tempo diária (restrição hard) ─────────────────────────────────
TIME_WINDOW_START: float = 7.0    # 07:00
TIME_WINDOW_END: float = 21.0     # 21:00
DAILY_HOURS: float = TIME_WINDOW_END - TIME_WINDOW_START  # 14 h

# ── Duração fixa de cada visita (minutos) ───────────────────────────────────
VISIT_DURATION_MIN: int = 30

# ── Velocidade média de deslocamento (km/h) — perímetro urbano do DF ────────
AVERAGE_SPEED_KMH: float = 30.0

# ── Semana operacional ──────────────────────────────────────────────────────
WORKING_DAYS: int = 5
NUM_TEAMS: int = 8

# ── Bases das equipes (coordenadas aproximadas das sedes dos órgãos) ────────


@dataclass(frozen=True)
class BaseEquipe:
    id_equipe: int
    orgao: str
    lat: float
    lon: float


BASES: List[BaseEquipe] = [
    BaseEquipe(1, "Secretaria de Segurança Pública",      -15.8047, -47.8645),
    BaseEquipe(2, "Secretaria da Mulher",                  -15.7907, -47.8822),
    BaseEquipe(3, "Secretaria de Justiça",                 -15.7940, -47.8750),
    BaseEquipe(4, "Secretaria de Desenvolvimento Social",  -15.7690, -47.8820),
    BaseEquipe(5, "Secretaria de Saúde",                   -15.7960, -47.8730),
    BaseEquipe(6, "Polícia Civil",                         -15.8080, -47.8750),
    BaseEquipe(7, "Polícia Militar",                       -15.8060, -47.9080),
    BaseEquipe(8, "Corpo de Bombeiros",                    -15.7990, -47.9130),
]

BASES_DICT: Dict[int, BaseEquipe] = {b.id_equipe: b for b in BASES}


def max_visits_per_team_per_day(avg_travel_min: float = 15.0) -> int:
    """
    Estimativa conservadora de visitas por equipe por dia.

    Parameters
    ----------
    avg_travel_min : tempo médio de deslocamento entre visitas (minutos).

    Returns
    -------
    Número inteiro de visitas que cabem na janela diária.
    """
    slot_min = VISIT_DURATION_MIN + avg_travel_min
    return int((DAILY_HOURS * 60) / slot_min)


def max_visits_total_week(avg_travel_min: float = 15.0) -> int:
    """Capacidade total semanal de visitas (todas as equipes)."""
    return max_visits_per_team_per_day(avg_travel_min) * WORKING_DAYS * NUM_TEAMS
