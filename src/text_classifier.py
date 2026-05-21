"""Treino e inferencia local de risco textual.

Baseline forte e simples para classificacao de narrativas:
- TF-IDF com n-grams
- LogisticRegression com class_weight balanceado

O script treina usando os JSONL produzidos em data/processed/finetune e
tambem permite predicao interativa por texto livre.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


BASE_PATH = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = BASE_PATH / "data" / "processed" / "finetune" / "risk_train.jsonl"
DEFAULT_VAL_PATH = BASE_PATH / "data" / "processed" / "finetune" / "risk_val.jsonl"
DEFAULT_MODEL_PATH = BASE_PATH / "models" / "text_risk_classifier.pkl"
DEFAULT_REPORT_PATH = BASE_PATH / "models" / "text_risk_classifier_metrics.json"
CLASS_ORDER = ["BAIXO", "MEDIO", "ALTO"]

USER_TEXT_RE = re.compile(
    r"Relato anonimizado:\s*(.*?)\s*Classifique o risco",
    re.IGNORECASE | re.DOTALL,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def extract_training_text(user_message: str) -> str:
    """Extrai apenas o relato anonimizado do prompt de treino."""
    message = normalize_text(user_message)
    match = USER_TEXT_RE.search(message)
    if match:
        return normalize_text(match.group(1))
    return message


def load_jsonl_dataset(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user_message = next(
                item["content"]
                for item in record["messages"]
                if item["role"] == "user"
            )
            records.append(
                {
                    "text": extract_training_text(user_message),
                    "label": str(record["risk_label"]),
                    "row_id": record.get("row_id", -1),
                }
            )

    df = pd.DataFrame(records)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=40000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=None,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, x_val: Iterable[str], y_val: Iterable[str]) -> Dict[str, Any]:
    predictions = model.predict(list(x_val))

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average="macro", zero_division=0)
    recall = recall_score(y_val, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_val, predictions, average="macro", zero_division=0)
    report = classification_report(
        y_val,
        predictions,
        labels=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_val, predictions, labels=CLASS_ORDER).tolist()

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def train_model(
    train_path: Path,
    val_path: Path,
    model_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    train_df = load_jsonl_dataset(train_path)
    val_df = load_jsonl_dataset(val_path)

    model = build_pipeline()
    model.fit(train_df["text"], train_df["label"])

    metrics = evaluate(model, val_df["text"], val_df["label"])

    artifact = {
        "model": model,
        "class_order": CLASS_ORDER,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    report_payload = {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "model_path": str(model_path),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload


def load_artifact(model_path: Path) -> Dict[str, Any]:
    return joblib.load(model_path)


def predict_text(model: Pipeline, text: str) -> Dict[str, Any]:
    cleaned_text = normalize_text(text)
    if not cleaned_text:
        raise ValueError("Texto vazio nao pode ser classificado.")

    probabilities = model.predict_proba([cleaned_text])[0]
    classes = list(model.named_steps["clf"].classes_)

    best_index = int(max(range(len(probabilities)), key=lambda idx: probabilities[idx]))
    label = classes[best_index]
    confidence = float(probabilities[best_index])

    ranked = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "label": label,
        "confidence": confidence,
        "ranking": [(cls, float(prob)) for cls, prob in ranked],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino e inferencia local de classificacao textual de risco.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Treina o classificador local")
    train_parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    train_parser.add_argument("--val-path", default=str(DEFAULT_VAL_PATH))
    train_parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    train_parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))

    predict_parser = subparsers.add_parser("predict", help="Prediz uma narrativa")
    predict_parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    predict_parser.add_argument("--text", default=None, help="Texto para classificar")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        train_path = Path(args.train_path)
        val_path = Path(args.val_path)
        model_path = Path(args.model_path)
        report_path = Path(args.report_path)

        result = train_model(train_path, val_path, model_path, report_path)
        metrics = result["metrics"]

        print(f"[OK] Modelo salvo em: {model_path}")
        print(f"[OK] Relatorio salvo em: {report_path}")
        print(
            "[METRICAS] "
            f"accuracy={metrics['accuracy']:.4f} "
            f"precision_macro={metrics['precision_macro']:.4f} "
            f"recall_macro={metrics['recall_macro']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
        return

    if args.command == "predict":
        model_path = Path(args.model_path)
        artifact = load_artifact(model_path)
        model = artifact["model"]

        text = args.text
        if not text:
            print("Digite o texto para classificar e pressione Enter. Deixe vazio para sair.")
            while True:
                text = input("Texto> ").strip()
                if not text:
                    break
                result = predict_text(model, text)
                print(
                    f"Classe prevista: {result['label']} | "
                    f"Confianca: {result['confidence']:.2%}"
                )
                print("Ranking:")
                for label, prob in result["ranking"]:
                    print(f"  - {label}: {prob:.2%}")
                print()
            return

        result = predict_text(model, text)
        print(f"Classe prevista: {result['label']}")
        print(f"Confianca: {result['confidence']:.2%}")
        print("Ranking:")
        for label, prob in result["ranking"]:
            print(f"  - {label}: {prob:.2%}")
        return


if __name__ == "__main__":
    main()