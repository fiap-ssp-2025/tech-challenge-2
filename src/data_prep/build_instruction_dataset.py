import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


HIGH_NATURE_KEYWORDS = {
    "estupro",
    "estupro de vulnerável",
    "estupro de vulneravel",
    "lesão corporal",
    "lesao corporal",
    "tentativa de homicidio",
    "homicidio",
    "descumprir decisão judicial",
    "medidas protetivas",
}

MEDIUM_NATURE_KEYWORDS = {
    "ameaça",
    "ameaca",
    "vias de fato",
    "injuria",
    "injúria",
    "difamacao",
    "difamação",
    "perseguição",
    "perseguicao",
    "importunação sexual",
    "importunacao sexual",
    "assédio sexual",
    "assedio sexual",
}

HIGH_TEXT_SIGNALS = {
    "matar",
    "morte",
    "arma",
    "faca",
    "fogo",
    "estrangul",
    "quebrou medida",
    "descumpriu medida",
    "ameaçou matar",
    "chutes",
    "socos",
}

MEDIUM_TEXT_SIGNALS = {
    "ameaçou",
    "ameaçou",
    "xing",
    "humilhou",
    "perseg",
    "empurrou",
    "agred",
    "controla",
    "ciume",
    "ciúme",
}

SYSTEM_PROMPT = (
    "Voce e um assistente de triagem de risco de violencia domestica. "
    "Classifique o risco em BAIXO, MEDIO ou ALTO com justificativa curta e objetiva, "
    "sem aconselhamento juridico ou medico."
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def score_risk(natureza: str, relato: str) -> Tuple[str, str]:
    nature = normalize_text(natureza).lower()
    text = normalize_text(relato).lower()

    high_hits = []
    medium_hits = []

    for kw in HIGH_NATURE_KEYWORDS:
        if kw in nature:
            high_hits.append(f"natureza:{kw}")

    for kw in MEDIUM_NATURE_KEYWORDS:
        if kw in nature:
            medium_hits.append(f"natureza:{kw}")

    for kw in HIGH_TEXT_SIGNALS:
        if kw in text:
            high_hits.append(f"texto:{kw}")

    for kw in MEDIUM_TEXT_SIGNALS:
        if kw in text:
            medium_hits.append(f"texto:{kw}")

    if high_hits:
        label = "ALTO"
        reason = "Sinais graves detectados: " + ", ".join(high_hits[:3])
        return label, reason

    if medium_hits:
        label = "MEDIO"
        reason = "Sinais moderados detectados: " + ", ".join(medium_hits[:3])
        return label, reason

    label = "BAIXO"
    reason = "Sem sinais graves ou moderados nas regras iniciais."
    return label, reason


def to_example(row: pd.Series) -> Dict[str, object]:
    natureza = normalize_text(row.get("natureza", ""))
    relato = normalize_text(row.get("text_anonymized", ""))

    label, justification = score_risk(natureza, relato)

    user_content = (
        f"Natureza: {natureza}\n"
        f"Relato anonimizado: {relato}\n"
        "Classifique o risco em BAIXO, MEDIO ou ALTO e justifique em 1-2 frases."
    )

    assistant_content = f"Risco: {label}\nJustificativa: {justification}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "risk_label": label,
        "natureza": natureza,
        "row_id": int(row.get("row_id", -1)),
    }


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_records(records: List[Dict[str, object]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dataset de instrucoes e rotulos de risco para treino local.")
    parser.add_argument(
        "--input",
        default="data/processed/finetune/dataset_vitima_contexto.csv",
        help="Arquivo CSV preparado pela etapa de anonimização",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/finetune",
        help="Diretorio para salvar os arquivos train/val",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Proporcao de treino",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    examples = [to_example(row) for _, row in df.iterrows() if normalize_text(row.get("text_anonymized", ""))]

    train, val = split_records(examples, args.train_ratio, args.seed)

    train_path = output_dir / "risk_train.jsonl"
    val_path = output_dir / "risk_val.jsonl"
    stats_path = output_dir / "risk_label_stats.json"

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    label_counts: Dict[str, int] = {"ALTO": 0, "MEDIO": 0, "BAIXO": 0}
    for ex in examples:
        label_counts[ex["risk_label"]] += 1

    stats = {
        "total_examples": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "label_distribution": label_counts,
    }

    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)

    print(f"[OK] train jsonl: {train_path}")
    print(f"[OK] val jsonl: {val_path}")
    print(f"[OK] stats: {stats_path}")


if __name__ == "__main__":
    main()
