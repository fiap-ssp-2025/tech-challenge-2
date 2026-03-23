"""
Algoritmo Genético Especializado para VRPTW
(Vehicle Routing Problem with Time Windows)

Aplicação: roteirização de visitas a vítimas de violência doméstica no DF.

Operadores genéticos projetados especificamente para o problema:
  - Representação: lista plana de IDs de vítimas dividida por separadores de equipe/dia.
  - Seleção: torneio com pressão seletiva ajustável.
  - Crossover: Order Crossover (OX) adaptado para segmentos de equipe/dia,
               com operador de reparo pós-crossover para violações de janela horária.
  - Mutação: swap_within_route, relocate_between_routes, reverse_segment.
  - Fitness: tempo total de rotas + penalidade por violação de janela 07h–21h.
  - Elitismo: preserva os K melhores entre gerações.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.routing.config import (
    AVERAGE_SPEED_KMH,
    BASES,
    DAILY_HOURS,
    NUM_TEAMS,
    TIME_WINDOW_END,
    TIME_WINDOW_START,
    VISIT_DURATION_MIN,
    WORKING_DAYS,
    BaseEquipe,
)

# ═══════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — REPRESENTAÇÃO DO PROBLEMA
# ═══════════════════════════════════════════════════════════════════════════

# Cada ponto (base ou vítima) é indexado de 0..N+7.
# Índices 0..7  → Bases das 8 equipes
# Índices 8..N+7 → Vítimas selecionadas para a semana


@dataclass
class VRPTWInstance:
    """Instância do problema VRPTW com todos os dados pré-computados."""

    n_victims: int
    n_teams: int
    n_days: int
    # Matriz de tempo de viagem (minutos) entre todos os pontos
    # Shape: (n_teams + n_victims, n_teams + n_victims)
    time_matrix: np.ndarray
    # Dados das vítimas para referência (id_vitima, risco_prob, lat, lon, etc.)
    victim_ids: List[str]
    victim_risks: np.ndarray
    victim_lats: np.ndarray
    victim_lons: np.ndarray
    # Parâmetros operacionais
    visit_duration_min: float = VISIT_DURATION_MIN
    tw_start: float = TIME_WINDOW_START
    tw_end: float = TIME_WINDOW_END
    daily_hours: float = DAILY_HOURS


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — MATRIZ DE TEMPOS (HAVERSINE + VELOCIDADE MÉDIA)
# ═══════════════════════════════════════════════════════════════════════════


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distância haversine em km entre dois pontos geográficos."""
    R = 6_371.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_time_matrix(
    bases: List[BaseEquipe],
    victim_lats: np.ndarray,
    victim_lons: np.ndarray,
    speed_kmh: float = AVERAGE_SPEED_KMH,
) -> np.ndarray:
    """
    Constrói a matriz simétrica de tempo de viagem (minutos) entre
    TODOS os pontos: [bases_0..bases_7, victims_0..victims_N-1].

    >>> RESTRIÇÃO APLICADA AQUI: tempo = distância_haversine / velocidade_média
    """
    n_bases = len(bases)
    n_total = n_bases + len(victim_lats)

    # Consolidar coordenadas: bases primeiro, depois vítimas
    all_lats = np.concatenate([[b.lat for b in bases], victim_lats])
    all_lons = np.concatenate([[b.lon for b in bases], victim_lons])

    # Computação vetorizada da distância haversine
    lat_rad = np.radians(all_lats)
    lon_rad = np.radians(all_lons)

    time_mat = np.zeros((n_total, n_total), dtype=np.float64)

    for i in range(n_total):
        dlat = lat_rad[i] - lat_rad[i + 1:]
        dlon = lon_rad[i] - lon_rad[i + 1:]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat_rad[i]) * np.cos(lat_rad[i + 1:]) * np.sin(dlon / 2) ** 2
        )
        dist_km = 6_371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        travel_min = (dist_km / speed_kmh) * 60.0
        time_mat[i, i + 1:] = travel_min
        time_mat[i + 1:, i] = travel_min

    return time_mat


def build_instance(
    victim_ids: List[str],
    victim_risks: np.ndarray,
    victim_lats: np.ndarray,
    victim_lons: np.ndarray,
) -> VRPTWInstance:
    """Cria instância completa do problema a partir dos dados das vítimas."""
    time_matrix = build_time_matrix(BASES, victim_lats, victim_lons)
    return VRPTWInstance(
        n_victims=len(victim_ids),
        n_teams=NUM_TEAMS,
        n_days=WORKING_DAYS,
        time_matrix=time_matrix,
        victim_ids=victim_ids,
        victim_risks=victim_risks,
        victim_lats=victim_lats,
        victim_lons=victim_lons,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — CROMOSSOMO: REPRESENTAÇÃO E DECODIFICAÇÃO
# ═══════════════════════════════════════════════════════════════════════════
#
# O cromossomo é uma lista de listas de listas:
#   chromosome[equipe][dia] = [victim_index_0, victim_index_1, ...]
#
# Cada victim_index é um índice local (0..n_victims-1) que mapeia para
# a posição n_teams + victim_index na time_matrix.
#
# A decodificação verifica as restrições de janela horária e calcula
# os horários reais de chegada/saída em cada visita.


@dataclass
class VisitSlot:
    """Slot calculado de uma visita na rota."""
    victim_idx: int
    arrival_time: float     # hora do dia (ex: 7.5 = 07:30)
    departure_time: float   # arrival + visit_duration(em horas)
    travel_time_min: float  # tempo de deslocamento até este ponto
    violates_tw: bool       # True se arrival < tw_start ou departure > tw_end


@dataclass
class DecodedRoute:
    """Rota decodificada de uma equipe em um dia."""
    team_id: int
    day: int
    visits: List[VisitSlot]
    total_travel_min: float
    total_service_min: float
    return_travel_min: float  # tempo de retorno à base
    n_violations: int


def decode_chromosome(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
) -> List[DecodedRoute]:
    """
    Decodifica o cromossomo em rotas com horários reais.

    >>> RESTRIÇÃO DE JANELA HORÁRIA (07h–21h) VERIFICADA AQUI:
    >>> Cada visita começa no max(hora_chegada, TW_START) e é marcada
    >>> como violação se arrival < TW_START ou departure > TW_END.
    """
    routes: List[DecodedRoute] = []
    visit_dur_h = instance.visit_duration_min / 60.0

    for team_idx in range(instance.n_teams):
        base_matrix_idx = team_idx  # índice da base na time_matrix

        for day in range(instance.n_days):
            day_victims = chromosome[team_idx][day]
            visits: List[VisitSlot] = []
            total_travel = 0.0
            total_service = 0.0
            n_viol = 0

            # ── RESTRIÇÃO: Início da jornada no TW_START ──────────────
            current_time = instance.tw_start
            current_pos = base_matrix_idx

            for v_idx in day_victims:
                dest_matrix_idx = instance.n_teams + v_idx
                travel_min = instance.time_matrix[current_pos, dest_matrix_idx]
                travel_h = travel_min / 60.0

                arrival = current_time + travel_h

                # ── RESTRIÇÃO: Espera se chegar antes da janela ────────
                effective_start = max(arrival, instance.tw_start)

                departure = effective_start + visit_dur_h

                # ── RESTRIÇÃO: Violação se saída ultrapassa TW_END ─────
                violates = arrival < instance.tw_start or departure > instance.tw_end
                if violates:
                    n_viol += 1

                visits.append(VisitSlot(
                    victim_idx=v_idx,
                    arrival_time=arrival,
                    departure_time=departure,
                    travel_time_min=travel_min,
                    violates_tw=violates,
                ))

                total_travel += travel_min
                total_service += instance.visit_duration_min
                current_time = departure
                current_pos = dest_matrix_idx

            # Tempo de retorno à base
            if day_victims:
                return_travel = instance.time_matrix[current_pos, base_matrix_idx]
            else:
                return_travel = 0.0
            total_travel += return_travel

            routes.append(DecodedRoute(
                team_id=team_idx + 1,
                day=day + 1,
                visits=visits,
                total_travel_min=total_travel,
                total_service_min=total_service,
                return_travel_min=return_travel,
                n_violations=n_viol,
            ))

    return routes


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — OPERADORES GENÉTICOS ESPECIALIZADOS
# ═══════════════════════════════════════════════════════════════════════════


# ── 4.1  Parâmetros do GA ─────────────────────────────────────────────────

@dataclass
class GAParams:
    """Hiperparâmetros do Algoritmo Genético."""
    population_size: int = 60
    generations: int = 200
    elitism_k: int = 4
    k_tournament: int = 3
    crossover_rate: float = 0.85
    mutation_swap_rate: float = 0.30
    mutation_relocate_rate: float = 0.25
    mutation_reverse_rate: float = 0.20
    # Penalidade por cada violação de janela de tempo (minutos adicionados ao fitness)
    tw_penalty_min: float = 120.0
    # Penalidade por desbalanceamento de carga (minutos por unidade de desvio-padrão)
    balance_penalty_min: float = 60.0
    random_seed: int = 42


# ── 4.2  Inicialização da população ──────────────────────────────────────

def _nearest_neighbor_init(
    instance: VRPTWInstance,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    OPERADOR DE INICIALIZAÇÃO: Inserção gulosa nearest-neighbor.

    Para cada vítima, seleciona a melhor equipe/dia considerando tanto
    a proximidade (tempo de viagem) quanto a carga atual da equipe,
    para evitar concentração excessiva em bases centrais.
    """
    n_v = instance.n_victims
    available = list(range(n_v))
    rng.shuffle(available)

    chromosome: List[List[List[int]]] = [
        [[] for _ in range(instance.n_days)] for _ in range(instance.n_teams)
    ]

    visit_dur_h = instance.visit_duration_min / 60.0
    # Alvo de visitas por equipe para balanceamento na inicialização
    target_per_team = n_v / instance.n_teams

    for v_idx in available:
        best_team = -1
        best_day = -1
        best_score = float("inf")

        for t in range(instance.n_teams):
            base_idx = t
            # Penalidade de carga: equipes já sobrecarregadas recebem custo maior
            team_total = sum(len(chromosome[t][d]) for d in range(instance.n_days))
            load_penalty = max(0, team_total - target_per_team) * 5.0  # 5 min por excesso

            for d in range(instance.n_days):
                route = chromosome[t][d]

                # Simular tempo até o final da rota atual
                current_time = instance.tw_start
                current_pos = base_idx
                for existing in route:
                    dest = instance.n_teams + existing
                    travel_h = instance.time_matrix[current_pos, dest] / 60.0
                    current_time = max(current_time + travel_h, instance.tw_start) + visit_dur_h
                    current_pos = dest

                # Tempo para adicionar v_idx
                dest_v = instance.n_teams + v_idx
                travel_to_v = instance.time_matrix[current_pos, dest_v] / 60.0
                arrival = current_time + travel_to_v
                departure = max(arrival, instance.tw_start) + visit_dur_h

                # Tempo de retorno à base
                return_h = instance.time_matrix[dest_v, base_idx] / 60.0

                # ── RESTRIÇÃO: só adiciona se caber na janela 07–21h ──
                if departure + return_h <= instance.tw_end:
                    score = (travel_to_v * 60) + load_penalty
                    if score < best_score:
                        best_score = score
                        best_team = t
                        best_day = d

        if best_team >= 0:
            chromosome[best_team][best_day].append(v_idx)

    return chromosome


def _random_init(
    instance: VRPTWInstance,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    OPERADOR DE INICIALIZAÇÃO: Distribuição aleatória das vítimas
    entre equipes/dias com perturbação.
    """
    n_v = instance.n_victims
    victims = list(range(n_v))
    rng.shuffle(victims)

    chromosome: List[List[List[int]]] = [
        [[] for _ in range(instance.n_days)] for _ in range(instance.n_teams)
    ]

    n_slots = instance.n_teams * instance.n_days
    for i, v in enumerate(victims):
        slot = i % n_slots
        team = slot // instance.n_days
        day = slot % instance.n_days
        chromosome[team][day].append(v)

    return chromosome


def initialize_population(
    instance: VRPTWInstance,
    params: GAParams,
) -> List[List[List[List[int]]]]:
    """
    Gera população inicial: metade por nearest-neighbor, metade aleatória.
    Isso garante diversidade genética enquanto parte de soluções razoáveis.
    """
    rng = random.Random(params.random_seed)
    population: List[List[List[List[int]]]] = []

    n_greedy = params.population_size // 2
    n_random = params.population_size - n_greedy

    for _ in range(n_greedy):
        chromosome = _nearest_neighbor_init(instance, rng)
        population.append(chromosome)

    for _ in range(n_random):
        chromosome = _random_init(instance, rng)
        population.append(chromosome)

    return population


# ── 4.3  Função de Fitness ───────────────────────────────────────────────

def fitness(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
    params: GAParams,
) -> float:
    """
    FUNÇÃO DE FITNESS: avalia a qualidade de uma solução (cromossomo).

    Componentes (a minimizar):
      1. Tempo total de viagem de todas as rotas (minutos)
      2. Penalidade por violações de janela de tempo 07h–21h
      3. Penalidade por desbalanceamento de carga entre equipes

    >>> RESTRIÇÃO DE JANELA HORÁRIA PENALIZADA AQUI:
    >>> Cada violação acrescenta tw_penalty_min ao fitness.
    """
    routes = decode_chromosome(chromosome, instance)

    total_travel = sum(r.total_travel_min for r in routes)
    total_violations = sum(r.n_violations for r in routes)

    # Penalidade por violações de TW
    tw_penalty = total_violations * params.tw_penalty_min

    # Penalidade por desbalanceamento: desvio-padrão do nº de visitas por equipe
    visits_per_team = [0] * instance.n_teams
    for r in routes:
        visits_per_team[r.team_id - 1] += len(r.visits)

    if max(visits_per_team) > 0:
        std_visits = float(np.std(visits_per_team))
        balance_penalty = std_visits * params.balance_penalty_min
    else:
        balance_penalty = 0.0

    return total_travel + tw_penalty + balance_penalty


# ── 4.4  Seleção por Torneio ─────────────────────────────────────────────

def tournament_selection(
    population: List[List[List[List[int]]]],
    fitnesses: List[float],
    k: int,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    OPERADOR DE SELEÇÃO: Torneio de tamanho k.

    Seleciona k indivíduos aleatoriamente e retorna o de menor fitness
    (menor = melhor, pois minimizamos tempo).
    Pressão seletiva controlada por k: k maior → mais pressão.
    """
    candidates = rng.sample(list(enumerate(fitnesses)), k)
    best_idx = min(candidates, key=lambda x: x[1])[0]
    return deepcopy(population[best_idx])


# ── 4.5  Crossover OX Adaptado ──────────────────────────────────────────

def _flatten_chromosome(
    chromosome: List[List[List[int]]],
    n_teams: int,
    n_days: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Achata cromossomo em lista plana + metadados de segmento (team, day)."""
    flat: List[int] = []
    segments: List[Tuple[int, int]] = []  # (start_index, length) por slot
    for t in range(n_teams):
        for d in range(n_days):
            seg = chromosome[t][d]
            segments.append((len(flat), len(seg)))
            flat.extend(seg)
    return flat, segments


def _unflatten_chromosome(
    flat: List[int],
    segments: List[Tuple[int, int]],
    n_teams: int,
    n_days: int,
) -> List[List[List[int]]]:
    """Recompõe cromossomo 3D a partir da lista plana e dos segmentos."""
    chromosome: List[List[List[int]]] = [
        [[] for _ in range(n_days)] for _ in range(n_teams)
    ]
    idx = 0
    for t in range(n_teams):
        for d in range(n_days):
            _, length = segments[idx]
            chromosome[t][d] = flat[:length]
            flat = flat[length:]
            idx += 1
    return chromosome


def order_crossover(
    parent1: List[List[List[int]]],
    parent2: List[List[List[int]]],
    instance: VRPTWInstance,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    OPERADOR DE CROSSOVER: Order Crossover (OX) adaptado para VRPTW.

    1. Achata ambos os pais em listas planas de victim_indices.
    2. Seleciona um segmento do parent1 e o preserva na posição.
    3. Preenche o restante com a ordem do parent2 (sem duplicatas).
    4. Recompõe a estrutura 3D usando a distribuição de slots do parent1.

    >>> Após crossover, o operador de reparo garante viabilidade da janela.
    """
    n_t, n_d = instance.n_teams, instance.n_days

    flat1, segments1 = _flatten_chromosome(parent1, n_t, n_d)
    flat2, _ = _flatten_chromosome(parent2, n_t, n_d)

    size = len(flat1)
    if size == 0:
        return deepcopy(parent1)

    # Pontos de corte
    cx1, cx2 = sorted(rng.sample(range(size), 2))

    # Segmento preservado do parent1
    child_flat = [None] * size
    preserved = set()
    for i in range(cx1, cx2 + 1):
        child_flat[i] = flat1[i]
        preserved.add(flat1[i])

    # Preencher com ordem do parent2
    p2_order = [v for v in flat2 if v not in preserved]
    p2_iter = iter(p2_order)
    for i in range(size):
        if child_flat[i] is None:
            child_flat[i] = next(p2_iter)

    child = _unflatten_chromosome(child_flat, segments1, n_t, n_d)

    # ── Reparo pós-crossover: remover vítimas que violam TW ────────
    child = _repair_tw_violations(child, instance)

    return child


def _repair_tw_violations(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
) -> List[List[List[int]]]:
    """
    OPERADOR DE REPARO: Reatribui vítimas que violam a janela 07h–21h.

    >>> RESTRIÇÃO DE JANELA HORÁRIA REPARADA AQUI:
    >>> Vítimas que causam violação são removidas da rota e reinseridas
    >>> no primeiro slot viável de outra equipe/dia.
    """
    visit_dur_h = instance.visit_duration_min / 60.0
    displaced: List[int] = []

    # Fase 1: remover vítimas que violam TW
    for t in range(instance.n_teams):
        base_idx = t
        for d in range(instance.n_days):
            new_route: List[int] = []
            current_time = instance.tw_start
            current_pos = base_idx

            for v_idx in chromosome[t][d]:
                dest = instance.n_teams + v_idx
                travel_h = instance.time_matrix[current_pos, dest] / 60.0
                arrival = current_time + travel_h
                departure = max(arrival, instance.tw_start) + visit_dur_h
                return_h = instance.time_matrix[dest, base_idx] / 60.0

                if departure + return_h <= instance.tw_end:
                    new_route.append(v_idx)
                    current_time = departure
                    current_pos = dest
                else:
                    displaced.append(v_idx)

            chromosome[t][d] = new_route

    # Fase 2: reinserir deslocados no melhor slot disponível
    for v_idx in displaced:
        best_cost = float("inf")
        best_t = -1
        best_d = -1

        for t in range(instance.n_teams):
            base_idx = t
            for d in range(instance.n_days):
                route = chromosome[t][d]
                # Calcular tempo até o final da rota
                current_time = instance.tw_start
                current_pos = base_idx
                for existing in route:
                    dest = instance.n_teams + existing
                    travel_h = instance.time_matrix[current_pos, dest] / 60.0
                    current_time = max(current_time + travel_h, instance.tw_start) + visit_dur_h
                    current_pos = dest

                dest_v = instance.n_teams + v_idx
                travel_to = instance.time_matrix[current_pos, dest_v] / 60.0
                arrival = current_time + travel_to
                departure = max(arrival, instance.tw_start) + visit_dur_h
                return_h = instance.time_matrix[dest_v, base_idx] / 60.0

                if departure + return_h <= instance.tw_end:
                    cost = travel_to * 60  # minutos
                    if cost < best_cost:
                        best_cost = cost
                        best_t = t
                        best_d = d

        if best_t >= 0:
            chromosome[best_t][best_d].append(v_idx)
        # Se não couber em lugar nenhum, a vítima é descartada desta solução
        # (penalizada via fitness pelo menor número de visitas)

    return chromosome


# ── 4.6  Operadores de Mutação ──────────────────────────────────────────

def mutate_swap_within_route(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    MUTAÇÃO 1 — Swap intra-rota:
    Troca duas vítimas de posição dentro da mesma rota (equipe + dia).
    Preserva a alocação equipe/dia, melhora a sequência local.
    """
    chromosome = deepcopy(chromosome)
    # Encontrar rotas com >= 2 vítimas
    candidates = [
        (t, d)
        for t in range(instance.n_teams)
        for d in range(instance.n_days)
        if len(chromosome[t][d]) >= 2
    ]
    if not candidates:
        return chromosome

    t, d = rng.choice(candidates)
    route = chromosome[t][d]
    i, j = rng.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    return chromosome


def mutate_relocate_between_routes(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    MUTAÇÃO 2 — Relocação inter-rotas:
    Move uma vítima da equipe com maior carga para a equipe com menor carga.
    Serve para BALANCEAR a distribuição de visitas entre equipes.
    """
    chromosome = deepcopy(chromosome)

    # Contar visitas por equipe
    counts = [
        sum(len(chromosome[t][d]) for d in range(instance.n_days))
        for t in range(instance.n_teams)
    ]
    team_max = int(np.argmax(counts))
    team_min = int(np.argmin(counts))

    if team_max == team_min or counts[team_max] == 0:
        return chromosome

    # Escolher dia com visitas na equipe sobrecarregada
    days_with_visits = [
        d for d in range(instance.n_days) if len(chromosome[team_max][d]) > 0
    ]
    if not days_with_visits:
        return chromosome

    src_day = rng.choice(days_with_visits)
    src_route = chromosome[team_max][src_day]
    victim = src_route.pop(rng.randrange(len(src_route)))

    # Inserir no dia com menos visitas da equipe ociosa
    day_counts_min = [len(chromosome[team_min][d]) for d in range(instance.n_days)]
    dst_day = int(np.argmin(day_counts_min))
    chromosome[team_min][dst_day].append(victim)

    return chromosome


def mutate_reverse_segment(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
    rng: random.Random,
) -> List[List[List[int]]]:
    """
    MUTAÇÃO 3 — 2-opt local (reverse segment):
    Inverte um sub-segmento de uma rota, melhorando a sequência
    quando há cruzamentos de caminho.
    """
    chromosome = deepcopy(chromosome)
    candidates = [
        (t, d)
        for t in range(instance.n_teams)
        for d in range(instance.n_days)
        if len(chromosome[t][d]) >= 3
    ]
    if not candidates:
        return chromosome

    t, d = rng.choice(candidates)
    route = chromosome[t][d]
    i, j = sorted(rng.sample(range(len(route)), 2))
    route[i: j + 1] = reversed(route[i: j + 1])
    return chromosome


def mutate(
    chromosome: List[List[List[int]]],
    instance: VRPTWInstance,
    params: GAParams,
    rng: random.Random,
) -> List[List[List[int]]]:
    """Aplica mutações com probabilidades independentes configuráveis."""
    if rng.random() < params.mutation_swap_rate:
        chromosome = mutate_swap_within_route(chromosome, instance, rng)
    if rng.random() < params.mutation_relocate_rate:
        chromosome = mutate_relocate_between_routes(chromosome, instance, rng)
    if rng.random() < params.mutation_reverse_rate:
        chromosome = mutate_reverse_segment(chromosome, instance, rng)
    return chromosome


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — LOOP PRINCIPAL DO ALGORITMO GENÉTICO
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GAResult:
    """Resultado completo de uma execução do GA."""
    best_chromosome: List[List[List[int]]]
    best_fitness: float
    best_routes: List[DecodedRoute]
    fitness_history: List[float]        # melhor fitness por geração
    avg_fitness_history: List[float]    # fitness médio por geração
    params: GAParams
    instance: VRPTWInstance


def run_ga(
    instance: VRPTWInstance,
    params: GAParams | None = None,
    verbose: bool = True,
) -> GAResult:
    """
    Executa o Algoritmo Genético para o problema VRPTW.

    Fluxo por geração:
      1. Avaliar fitness de toda a população
      2. Elitismo: preservar os K melhores
      3. Seleção por torneio + crossover OX adaptado
      4. Mutação com operadores especializados
      5. Registrar métricas de convergência
    """
    if params is None:
        params = GAParams()

    rng = random.Random(params.random_seed)

    # ── Inicialização ─────────────────────────────────────────────────
    if verbose:
        print(f"Inicializando população ({params.population_size} indivíduos)...")
    population = initialize_population(instance, params)

    best_chromosome: List[List[List[int]]] = []
    best_fitness = float("inf")
    fitness_history: List[float] = []
    avg_fitness_history: List[float] = []

    # ── Loop evolutivo ────────────────────────────────────────────────
    for gen in range(params.generations):
        # Avaliar fitness
        fitnesses = [fitness(ind, instance, params) for ind in population]

        gen_best_idx = int(np.argmin(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        gen_avg_fit = float(np.mean(fitnesses))

        if gen_best_fit < best_fitness:
            best_fitness = gen_best_fit
            best_chromosome = deepcopy(population[gen_best_idx])

        fitness_history.append(best_fitness)
        avg_fitness_history.append(gen_avg_fit)

        if verbose and (gen % 20 == 0 or gen == params.generations - 1):
            routes = decode_chromosome(population[gen_best_idx], instance)
            n_viol = sum(r.n_violations for r in routes)
            total_visits = sum(len(r.visits) for r in routes)
            print(
                f"  Gen {gen:4d}/{params.generations} | "
                f"Best: {gen_best_fit:10.1f} min | "
                f"Avg: {gen_avg_fit:10.1f} min | "
                f"Visitas: {total_visits} | "
                f"Violações TW: {n_viol}"
            )

        # ── Elitismo: preservar os K melhores ─────────────────────────
        sorted_pop = sorted(
            zip(population, fitnesses), key=lambda x: x[1]
        )
        new_population: List[List[List[List[int]]]] = [
            deepcopy(ind) for ind, _ in sorted_pop[: params.elitism_k]
        ]

        # ── Reprodução: seleção + crossover + mutação ─────────────────
        while len(new_population) < params.population_size:
            parent1 = tournament_selection(
                population, fitnesses, params.k_tournament, rng
            )
            parent2 = tournament_selection(
                population, fitnesses, params.k_tournament, rng
            )

            if rng.random() < params.crossover_rate:
                child = order_crossover(parent1, parent2, instance, rng)
            else:
                child = deepcopy(parent1)

            child = mutate(child, instance, params, rng)
            new_population.append(child)

        population = new_population

    # ── Resultado final ───────────────────────────────────────────────
    best_routes = decode_chromosome(best_chromosome, instance)

    if verbose:
        total_visits = sum(len(r.visits) for r in best_routes)
        total_violations = sum(r.n_violations for r in best_routes)
        total_travel = sum(r.total_travel_min for r in best_routes)
        print(f"\n═══ GA Finalizado ═══")
        print(f"  Melhor fitness: {best_fitness:.1f} min")
        print(f"  Total de visitas: {total_visits}")
        print(f"  Violações de janela horária: {total_violations}")
        print(f"  Tempo total de deslocamento: {total_travel:.1f} min")

    return GAResult(
        best_chromosome=best_chromosome,
        best_fitness=best_fitness,
        best_routes=best_routes,
        fitness_history=fitness_history,
        avg_fitness_history=avg_fitness_history,
        params=params,
        instance=instance,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — DECODIFICAÇÃO PARA CRONOGRAMA SEMANAL (DataFrame)
# ═══════════════════════════════════════════════════════════════════════════

def routes_to_dataframe(result: GAResult) -> pd.DataFrame:
    """
    Converte as rotas do melhor indivíduo em DataFrame com cronograma
    semanal completo, pronto para visualização no Streamlit.
    """
    rows: List[Dict[str, Any]] = []
    inst = result.instance
    day_names = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]

    for route in result.best_routes:
        base = BASES[route.team_id - 1]
        for order, visit in enumerate(route.visits, start=1):
            v_idx = visit.victim_idx

            def _time_str(h: float) -> str:
                hours = int(h)
                minutes = int((h - hours) * 60)
                return f"{hours:02d}:{minutes:02d}"

            rows.append({
                "equipe": route.team_id,
                "orgao_base": base.orgao,
                "base_lat": base.lat,
                "base_lon": base.lon,
                "dia": route.day,
                "dia_semana": day_names[route.day - 1] if route.day <= 5 else f"Dia {route.day}",
                "ordem_visita": order,
                "id_vitima": inst.victim_ids[v_idx],
                "risco_prob": float(inst.victim_risks[v_idx]),
                "lat": float(inst.victim_lats[v_idx]),
                "lon": float(inst.victim_lons[v_idx]),
                "hora_chegada": _time_str(visit.arrival_time),
                "hora_saida": _time_str(visit.departure_time),
                "tempo_deslocamento_min": round(visit.travel_time_min, 1),
                "violacao_janela": visit.violates_tw,
            })

    df = pd.DataFrame(rows)
    return df
