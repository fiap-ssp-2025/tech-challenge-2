import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.fairness import evaluate_by_group

TARGET_COLUMN = "OUT_VEZES"
GROUP_COLUMN = "CS_RACA"
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "data" / "processed" / "df_preprocessed.parquet"

# Espaço de busca dos hiperparâmetros (representação dos genes)
HYPERPARAM_SPACE = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["sqrt", "log2", None],
}

# 3 experimentos com diferentes configurações do algoritmo genético
EXPERIMENTS = [
    {
        "name": "exp1_conservador",
        "population_size": 8,
        "generations": 5,
        "mutation_rate": 0.1,
        "elitism": 2,
        "tournament_size": 3,
    },
    {
        "name": "exp2_moderado",
        "population_size": 14,
        "generations": 10,
        "mutation_rate": 0.2,
        "elitism": 3,
        "tournament_size": 4,
    },
    {
        "name": "exp3_agressivo",
        "population_size": 20,
        "generations": 15,
        "mutation_rate": 0.35,
        "elitism": 4,
        "tournament_size": 5,
    },
]


# ── Dados ─────────────────────────────────────────────────────────────────────


def load_data():
    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_encoded = X.copy()
    for col in X_encoded.columns:
        col_data = X_encoded[col].astype(str).replace("nan", np.nan)
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(
            col_data.fillna(col_data.mode()[0] if len(col_data.mode()) > 0 else "0")
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )

    return df, X_encoded, y, X_train, X_test, y_train, y_test


# ── Métricas ──────────────────────────────────────────────────────────────────


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


def compute_fitness(metrics, recall_gap=0.0):
    """
    Score composto priorizando recall (doenças críticas) e
    penalizando desigualdade entre grupos demográficos.
    """
    fairness = 1.0 - recall_gap
    return (
        0.35 * metrics["recall"]
        + 0.25 * metrics["f1"]
        + 0.15 * metrics["specificity"]
        + 0.10 * metrics["roc_auc"]
        + 0.15 * fairness
    )


# ── Operadores do Algoritmo Genético ─────────────────────────────────────────


def create_individual():
    return {key: random.choice(options) for key, options in HYPERPARAM_SPACE.items()}


def select_parent(population, fitnesses, tournament_size=3):
    candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
    return deepcopy(max(candidates, key=lambda x: x[1])[0])


def crossover(parent1, parent2):
    return {key: random.choice([parent1[key], parent2[key]]) for key in HYPERPARAM_SPACE}


def mutate(individual, mutation_rate):
    mutant = deepcopy(individual)
    for key, options in HYPERPARAM_SPACE.items():
        if random.random() < mutation_rate:
            mutant[key] = random.choice(options)
    return mutant


# ── Avaliação de fitness de um indivíduo ──────────────────────────────────────


def evaluate_individual(individual, X_train, X_test, y_train, y_test, original_df):
    model = RandomForestClassifier(**individual, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    group_result = evaluate_by_group(model, X_test, y_test, original_df, GROUP_COLUMN)
    recall_gap = group_result.get("_summary", {}).get("recall_gap", 0.0)
    metrics["recall_gap"] = float(recall_gap)

    score = compute_fitness(metrics, recall_gap)
    return score, metrics


# ── Loop do Algoritmo Genético ────────────────────────────────────────────────


def run_genetic_algorithm(
    X_train,
    X_test,
    y_train,
    y_test,
    original_df,
    population_size,
    generations,
    mutation_rate,
    elitism,
    tournament_size,
):
    population = [create_individual() for _ in range(population_size)]

    best_individual = None
    best_fitness = float("-inf")
    best_metrics = None

    for gen in range(generations):
        scored = []
        for ind in population:
            score, metrics = evaluate_individual(
                ind, X_train, X_test, y_train, y_test, original_df
            )
            scored.append((ind, score, metrics))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_ind, top_score, top_metrics = scored[0]

        if top_score > best_fitness:
            best_individual = deepcopy(top_ind)
            best_fitness = top_score
            best_metrics = top_metrics

        print(f"  Geração {gen + 1}/{generations} — fitness: {top_score:.4f}")

        new_population = [deepcopy(s[0]) for s in scored[:elitism]]
        fit_values = [s[1] for s in scored]
        pop_list = [s[0] for s in scored]

        while len(new_population) < population_size:
            p1 = select_parent(pop_list, fit_values, tournament_size)
            p2 = select_parent(pop_list, fit_values, tournament_size)
            child = mutate(crossover(p1, p2), mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual, best_fitness, best_metrics


# ── Baseline ──────────────────────────────────────────────────────────────────


def train_baseline(X_train, X_test, y_train, y_test, original_df):
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    group_metrics = evaluate_by_group(model, X_test, y_test, original_df, GROUP_COLUMN)
    metrics["recall_gap"] = group_metrics.get("_summary", {}).get("recall_gap", 0.0)

    return model, metrics, group_metrics


# ── Pipeline principal ────────────────────────────────────────────────────────


def run_all_experiments():
    df, X, y, X_train, X_test, y_train, y_test = load_data()

    print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Target: {y.value_counts().to_dict()}\n")

    models_dir = BASE_PATH / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    features = X.columns.tolist()

    # ── Baseline ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("BASELINE (hiperparâmetros default do sklearn)")
    print("=" * 60)

    baseline_model, baseline_metrics, baseline_groups = train_baseline(
        X_train, X_test, y_train, y_test, df
    )
    print(f"Métricas: {baseline_metrics}\n")

    joblib.dump(
        {
            "model": baseline_model,
            "features": features,
            "params": "default",
            "fitness": None,
            "metrics": baseline_metrics,
            "search_metrics": None,
            "group_metrics": baseline_groups,
            "fitness_mode": "baseline",
        },
        models_dir / "random_forest_baseline.pkl",
    )

    all_results = [{"experiment": "baseline", **baseline_metrics}]

    # ── Experimentos GA ───────────────────────────────────────────────────
    for exp in EXPERIMENTS:
        name = exp["name"]
        config = {k: v for k, v in exp.items() if k != "name"}

        print("=" * 60)
        print(f"EXPERIMENTO: {name}")
        print(f"Config: {config}")
        print("=" * 60)

        best_params, best_fit, search_metrics = run_genetic_algorithm(
            X_train, X_test, y_train, y_test, df, **config
        )

        final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train)

        final_metrics = evaluate_model(final_model, X_test, y_test)
        group_metrics = evaluate_by_group(final_model, X_test, y_test, df, GROUP_COLUMN)
        final_metrics["recall_gap"] = group_metrics.get("_summary", {}).get(
            "recall_gap", 0.0
        )

        print(f"\nMelhor fitness: {best_fit:.4f}")
        print(f"Parâmetros: {best_params}")
        print(f"Métricas finais: {final_metrics}\n")

        joblib.dump(
            {
                "model": final_model,
                "features": features,
                "params": best_params,
                "fitness": float(best_fit),
                "metrics": final_metrics,
                "search_metrics": search_metrics,
                "group_metrics": group_metrics,
                "fitness_mode": name,
                "ga_config": config,
            },
            models_dir / f"random_forest_{name}.pkl",
        )

        all_results.append({"experiment": name, **final_metrics})

    # ── Comparação ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARAÇÃO: BASELINE vs EXPERIMENTOS")
    print("=" * 60)

    comparison_df = pd.DataFrame(all_results)
    print(comparison_df.to_string(index=False))

    results_dir = BASE_PATH / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir / "comparison.csv", index=False)

    with open(results_dir / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResultados salvos em {results_dir}")


if __name__ == "__main__":
    run_all_experiments()
