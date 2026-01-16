from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.project_root import find_project_root


@dataclass(frozen=True)
class AprioriConfig:
    columns: list[str]
    min_support: float = 0.02
    min_confidence: float = 0.3
    min_lift: float = 1.0
    max_len: int = 3
    max_consequent_len: int = 1
    sample_n: int | None = None
    random_state: int = 42
    ignore_values: tuple[str, ...] = ("", " ", "nan", "None", "NaN", "9", "99", "88")


def _as_items_frame(
    df: pd.DataFrame, columns: list[str], ignore_values: tuple[str, ...]
) -> pd.DataFrame:
    items: dict[str, pd.Series] = {}
    ignore_set = set(v.strip() for v in ignore_values)

    for col in columns:
        s = df[col]
        # normalize to string labels (keep original codes)
        s = s.astype("string").str.strip()
        s = s.where(~s.isna(), pd.NA)
        s = s.where(~s.isin(ignore_set), pd.NA)
        items[col] = (col + "=" + s).astype("string")

    return pd.DataFrame(items)


def one_hot_items(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-column item labels (like 'COL=VAL') into a boolean one-hot matrix.
    """
    onehots: list[pd.DataFrame] = []
    for col in items_df.columns:
        d = pd.get_dummies(items_df[col], prefix="", prefix_sep="", dtype=bool)
        onehots.append(d)
    if not onehots:
        return pd.DataFrame()
    X = pd.concat(onehots, axis=1).fillna(False)
    # defensive: de-duplicate columns if any
    X = X.loc[:, ~X.columns.duplicated()]
    return X


def _support_from_arrays(
    item_arrays: dict[str, np.ndarray], items: Iterable[str]
) -> float:
    it = iter(items)
    first = next(it)
    mask = item_arrays[first].copy()
    for k in it:
        mask &= item_arrays[k]
    return float(mask.mean())


def apriori_frequent_itemsets(
    X: pd.DataFrame, min_support: float, max_len: int
) -> dict[frozenset[str], float]:
    """
    Simple Apriori producing a mapping {itemset -> support}.
    X: boolean DataFrame, columns are items.
    """
    if X.empty:
        return {}

    item_arrays = {c: X[c].to_numpy(dtype=bool) for c in X.columns}
    n_rows = len(X)
    if n_rows == 0:
        return {}

    supports: dict[frozenset[str], float] = {}

    # 1-itemsets
    s1 = X.mean(axis=0)
    L_prev = [frozenset([c]) for c, sup in s1.items() if sup >= min_support]
    for fs in L_prev:
        supports[fs] = float(s1[tuple(fs)[0]])

    k = 2
    while L_prev and k <= max_len:
        # candidates by join step
        prev_list = sorted([tuple(sorted(fs)) for fs in L_prev])
        prev_sets = set(L_prev)
        candidates: set[frozenset[str]] = set()

        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                a, b = prev_list[i], prev_list[j]
                if a[: k - 2] != b[: k - 2]:
                    break
                cand = frozenset(a) | frozenset(b)
                if len(cand) != k:
                    continue
                # prune: all (k-1)-subsets must be frequent
                all_subsets_frequent = True
                for sub in itertools.combinations(sorted(cand), k - 1):
                    if frozenset(sub) not in prev_sets:
                        all_subsets_frequent = False
                        break
                if all_subsets_frequent:
                    candidates.add(cand)

        Lk: list[frozenset[str]] = []
        for cand in candidates:
            sup = _support_from_arrays(item_arrays, sorted(cand))
            if sup >= min_support:
                supports[cand] = sup
                Lk.append(cand)

        L_prev = Lk
        k += 1

    return supports


def association_rules(
    supports: dict[frozenset[str], float],
    min_confidence: float,
    min_lift: float,
    max_consequent_len: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for itemset, sup_xy in supports.items():
        if len(itemset) < 2:
            continue

        items_sorted = tuple(sorted(itemset))
        for r in range(1, len(items_sorted)):
            for antecedent_tuple in itertools.combinations(items_sorted, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = frozenset(itemset - antecedent)
                if not consequent or len(consequent) > max_consequent_len:
                    continue

                sup_x = supports.get(antecedent)
                sup_y = supports.get(consequent)
                if not sup_x or not sup_y:
                    continue

                confidence = sup_xy / sup_x
                lift = confidence / sup_y
                if confidence < min_confidence or lift < min_lift:
                    continue

                leverage = sup_xy - (sup_x * sup_y)
                conviction = (
                    (1 - sup_y) / (1 - confidence) if confidence < 1 else np.inf
                )

                rows.append(
                    {
                        "antecedent": " & ".join(sorted(antecedent)),
                        "consequent": " & ".join(sorted(consequent)),
                        "support": sup_xy,
                        "confidence": confidence,
                        "lift": lift,
                        "antecedent_support": sup_x,
                        "consequent_support": sup_y,
                        "leverage": leverage,
                        "conviction": conviction,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)


def run_apriori(config: AprioriConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = find_project_root()
    data_path = root / "data" / "processed" / "df_cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Não encontrei {data_path}. Rode o pré-processamento e gere o parquet primeiro."
        )

    df = pd.read_parquet(data_path)
    missing = [c for c in config.columns if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas não encontradas no df_cleaned.parquet: {missing}")

    df = df[config.columns].copy()
    if config.sample_n:
        df = df.sample(n=config.sample_n, random_state=config.random_state)

    items_df = _as_items_frame(df, config.columns, config.ignore_values)
    X = one_hot_items(items_df)
    # remove rows with no items set
    X = X.loc[X.any(axis=1)]

    supports = apriori_frequent_itemsets(
        X, min_support=config.min_support, max_len=config.max_len
    )
    frequent = (
        pd.DataFrame(
            {
                "itemset": [" & ".join(sorted(k)) for k in supports.keys()],
                "length": [len(k) for k in supports.keys()],
                "support": list(supports.values()),
            }
        )
        .sort_values(["length", "support"], ascending=[True, False])
        .reset_index(drop=True)
    )

    rules = association_rules(
        supports,
        min_confidence=config.min_confidence,
        min_lift=config.min_lift,
        max_consequent_len=config.max_consequent_len,
    )
    return frequent, rules


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apriori + regras de associação (sem mlxtend)."
    )
    p.add_argument(
        "--columns",
        nargs="+",
        required=False,
        default=[
            "AG_AMEACA",
            "AUTOR_SEXO",
            "CS_ESCOL_N",
            "CS_RACA",
            "ORIENT_SEX",
            "OUT_VEZES",
            "REDE_SAU",
            "SG_UF",
            "SIT_CONJUG",
        ],
        help="Colunas a usar como itens (COL=VAL).",
    )
    p.add_argument("--min-support", type=float, default=0.02)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--min-lift", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=3)
    p.add_argument("--max-consequent-len", type=int, default=1)
    p.add_argument("--sample-n", type=int, default=None)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Diretório de saída (default: data/processed/).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = AprioriConfig(
        columns=list(args.columns),
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_len,
        max_consequent_len=args.max_consequent_len,
        sample_n=args.sample_n,
    )

    frequent, rules = run_apriori(cfg)
    root = find_project_root()
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (root / "data" / "processed")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frequent_path = out_dir / "apriori_frequent_itemsets.csv"
    rules_path = out_dir / "apriori_rules.csv"
    frequent.to_csv(frequent_path, index=False)
    rules.to_csv(rules_path, index=False)

    print(f"✅ Itemsets frequentes: {len(frequent)} -> {frequent_path}")
    print(f"✅ Regras: {len(rules)} -> {rules_path}")
    print("\nTop 20 regras (lift/confidence/support):")
    if rules.empty:
        print("(nenhuma regra com esses thresholds)")
    else:
        print(rules.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
