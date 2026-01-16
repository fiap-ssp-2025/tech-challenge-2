# Tech-Challenge-1

## Associação (Apriori)

Roda Apriori + regras de associação em cima do `data/processed/df_cleaned.parquet`, criando itens do tipo `COL=VAL`.

Exemplo (com as colunas pedidas):

```bash
python -m src.association.apriori_rules \
  --columns AG_AMEACA AUTOR_SEXO CS_ESCOL_N CS_RACA ORIENT_SEX OUT_VEZES REDE_SAU SG_UF SIT_CONJUG \
  --min-support 0.02 --min-confidence 0.3 --min-lift 1.0 --max-len 3
```

Saídas (default): `data/processed/apriori_frequent_itemsets.csv` e `data/processed/apriori_rules.csv`.
