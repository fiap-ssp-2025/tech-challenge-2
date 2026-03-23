import random
from copy import deepcopy
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
)
from sklearn.model_selection import train_test_split
from src.evaluation.fairness import evaluate_by_group

TARGET_COLUMN = "OUT_VEZES"
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "data" / "processed" / "df_preprocessed.parquet"

print(f"Base path: {BASE_PATH.absolute()}")


def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Shape do dataset: {df.shape}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_encoded = X.copy()

    for col in X_encoded.columns:
        col_data = X_encoded[col].astype(str).replace("nan", np.nan)
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(
            col_data.fillna(col_data.mode()[0] if len(col_data.mode()) > 0 else "0")
        )

    print(f"\nFeatures preparadas: {X_encoded.shape}")
    print(f"Target: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )

    return df, X_encoded, y, X_train, X_test, y_train, y_test


N_ESTIMATORS_OPTIONS = [100, 200, 300, 400]
MAX_DEPTH_OPTIONS = [5, 10, 15, 20, None]
MIN_SAMPLES_SPLIT_OPTIONS = [2, 4, 6, 8]
MIN_SAMPLES_LEAF_OPTIONS = [1, 2, 3, 4]
MAX_FEATURES_OPTIONS = ["sqrt", "log2", None]

POPULATION_SIZE = 6
GENERATIONS = 5
ELITISM = 2
MUTATION_RATE = 0.2


def create_individual():
    return {
        "n_estimators": random.choice(N_ESTIMATORS_OPTIONS),
        "max_depth": random.choice(MAX_DEPTH_OPTIONS),
        "min_samples_split": random.choice(MIN_SAMPLES_SPLIT_OPTIONS),
        "min_samples_leaf": random.choice(MIN_SAMPLES_LEAF_OPTIONS),
        "max_features": random.choice(MAX_FEATURES_OPTIONS),
    }


def fitness(individual, X_train, X_test, y_train, y_test, fitness_mode="current"):
    model = RandomForestClassifier(
        n_estimators=individual["n_estimators"],
        max_depth=individual["max_depth"],
        min_samples_split=individual["min_samples_split"],
        min_samples_leaf=individual["min_samples_leaf"],
        max_features=individual["max_features"],
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if fitness_mode == "current":
        score = 0.5 * recall + 0.3 * f1 + 0.2 * roc_auc
    elif fitness_mode == "balanced":
        score = 0.4 * recall + 0.3 * f1 + 0.2 * specificity + 0.1 * roc_auc
    elif fitness_mode == "clinical":
        score = 0.6 * recall + 0.2 * f1 + 0.2 * specificity
    else:
        raise ValueError(f"fitness_mode inválido: {fitness_mode}")

    return score, {
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "specificity": float(specificity),
    }


def select_parent(population, fitnesses, tournament_size=3):
    candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return deepcopy(candidates[0][0])


def crossover(parent1, parent2):
    return {
        "n_estimators": random.choice(
            [parent1["n_estimators"], parent2["n_estimators"]]
        ),
        "max_depth": random.choice([parent1["max_depth"], parent2["max_depth"]]),
        "min_samples_split": random.choice(
            [parent1["min_samples_split"], parent2["min_samples_split"]]
        ),
        "min_samples_leaf": random.choice(
            [parent1["min_samples_leaf"], parent2["min_samples_leaf"]]
        ),
        "max_features": random.choice(
            [parent1["max_features"], parent2["max_features"]]
        ),
    }


def mutate(individual):
    mutant = deepcopy(individual)

    if random.random() < MUTATION_RATE:
        mutant["n_estimators"] = random.choice(N_ESTIMATORS_OPTIONS)

    if random.random() < MUTATION_RATE:
        mutant["max_depth"] = random.choice(MAX_DEPTH_OPTIONS)

    if random.random() < MUTATION_RATE:
        mutant["min_samples_split"] = random.choice(MIN_SAMPLES_SPLIT_OPTIONS)

    if random.random() < MUTATION_RATE:
        mutant["min_samples_leaf"] = random.choice(MIN_SAMPLES_LEAF_OPTIONS)

    if random.random() < MUTATION_RATE:
        mutant["max_features"] = random.choice(MAX_FEATURES_OPTIONS)

    return mutant


def run_genetic_algorithm(fitness_mode="balanced"):
    df, X, y, X_train, X_test, y_train, y_test = load_data()

    population = [create_individual() for _ in range(POPULATION_SIZE)]

    best_individual = None
    best_fitness = float("-inf")
    best_metrics = None

    for generation in range(GENERATIONS):
        scored_population = []

        for individual in population:
            score, metrics = fitness(
                individual, X_train, X_test, y_train, y_test, fitness_mode=fitness_mode
            )
            scored_population.append((individual, score, metrics))

        scored_population.sort(key=lambda x: x[1], reverse=True)
        generation_best = scored_population[0]

        if generation_best[1] > best_fitness:
            best_individual = deepcopy(generation_best[0])
            best_fitness = generation_best[1]
            best_metrics = generation_best[2]

        print(f"\nGeração {generation + 1} | fitness_mode={fitness_mode}")
        print(f"Melhor indivíduo: {generation_best[0]}")
        print(f"Fitness: {generation_best[1]:.4f}")
        print(f"Métricas durante busca: {generation_best[2]}")

        new_population = [deepcopy(item[0]) for item in scored_population[:ELITISM]]

        fitnesses = [item[1] for item in scored_population]
        current_population = [item[0] for item in scored_population]

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(current_population, fitnesses)
            parent2 = select_parent(current_population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    print("\n=== Melhor solução encontrada ===")
    print(best_individual)
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best metrics (durante busca): {best_metrics}")

    final_model = RandomForestClassifier(
        **best_individual,
        random_state=42,
        n_jobs=-1,
    )

    # modelo final treinado com todo o dataset
    final_model.fit(X, y)

    # métricas do modelo final salvo
    y_pred_final = final_model.predict(X_test)
    y_prob_final = final_model.predict_proba(X_test)[:, 1]

    recall_final = recall_score(y_test, y_pred_final)
    f1_final = f1_score(y_test, y_pred_final)
    roc_auc_final = roc_auc_score(y_test, y_prob_final)
    precision_final = precision_score(y_test, y_pred_final, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
    specificity_final = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    final_metrics = {
        "recall": float(recall_final),
        "f1": float(f1_final),
        "roc_auc": float(roc_auc_final),
        "precision": float(precision_final),
        "specificity": float(specificity_final),
    }

    print("\n=== Métricas do modelo final salvo ===")
    print(final_metrics)

    # avaliação por grupo
    group_metrics = evaluate_by_group(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        original_df=df,
        group_column="CS_RACA",
    )

    print("\n=== Avaliação por grupo ===")
    print(group_metrics)

    artifact = {
        "model": final_model,
        "features": X.columns.tolist(),
        "params": best_individual,
        "fitness": float(best_fitness),
        "metrics": final_metrics,
        "search_metrics": best_metrics,
        "group_metrics": group_metrics,
        "fitness_mode": fitness_mode,
    }

    model_path = BASE_PATH / "models" / f"random_forest_{fitness_mode}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_path)
    print(f"Modelo salvo em {model_path}")


if __name__ == "__main__":
    run_genetic_algorithm(fitness_mode="balanced")