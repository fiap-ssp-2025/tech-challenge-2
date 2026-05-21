import argparse
from pathlib import Path

import pandas as pd


def normalize_relatos(df: pd.DataFrame, start_row_id: int) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "row_id": range(start_row_id, start_row_id + len(df)),
            "natureza": df["Classe"].fillna("").astype(str).str.strip(),
            "text_anonymized": df["Relato"].fillna("").astype(str).str.strip(),
            "segment_count": 1,
        }
    )
    normalized = normalized[normalized["text_anonymized"] != ""].reset_index(drop=True)
    return normalized


def merge_dataset(base_path: Path, relatos_df: pd.DataFrame, output_path: Path) -> int:
    base_df = pd.read_csv(base_path)
    max_row_id = int(pd.to_numeric(base_df["row_id"], errors="coerce").fillna(-1).max())
    normalized_relatos = normalize_relatos(relatos_df, max_row_id + 1)

    merged_df = pd.concat([base_df, normalized_relatos], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    return len(merged_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mescla relatos_treinamento.csv aos datasets vitima_only e vitima_contexto."
    )
    parser.add_argument(
        "--relatos-input",
        default="data/processed/finetune/relatos_treinamento.csv",
        help="CSV com colunas Classe, Relato e Target.",
    )
    parser.add_argument(
        "--vitima-only-input",
        default="data/processed/finetune/dataset_vitima_only.csv",
        help="Dataset base vitima_only.",
    )
    parser.add_argument(
        "--vitima-contexto-input",
        default="data/processed/finetune/dataset_vitima_contexto.csv",
        help="Dataset base vitima_contexto.",
    )
    parser.add_argument(
        "--vitima-only-output",
        default="data/processed/finetune/dataset_vitima_only_merged.csv",
        help="Saida do dataset vitima_only mesclado.",
    )
    parser.add_argument(
        "--vitima-contexto-output",
        default="data/processed/finetune/dataset_vitima_contexto_merged.csv",
        help="Saida do dataset vitima_contexto mesclado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    relatos_path = Path(args.relatos_input)
    vitima_only_input = Path(args.vitima_only_input)
    vitima_contexto_input = Path(args.vitima_contexto_input)
    vitima_only_output = Path(args.vitima_only_output)
    vitima_contexto_output = Path(args.vitima_contexto_output)

    for path in [relatos_path, vitima_only_input, vitima_contexto_input]:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    relatos_df = pd.read_csv(relatos_path)
    missing_cols = {"Classe", "Relato", "Target"} - set(relatos_df.columns)
    if missing_cols:
        raise ValueError(f"Colunas ausentes em relatos_treinamento.csv: {sorted(missing_cols)}")

    vitima_only_count = merge_dataset(vitima_only_input, relatos_df, vitima_only_output)
    vitima_contexto_count = merge_dataset(vitima_contexto_input, relatos_df, vitima_contexto_output)

    print(f"[OK] vitima_only merged: {vitima_only_output} ({vitima_only_count} linhas)")
    print(f"[OK] vitima_contexto merged: {vitima_contexto_output} ({vitima_contexto_count} linhas)")


if __name__ == "__main__":
    main()