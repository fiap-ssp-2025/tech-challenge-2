# Tech Challenge 2 — Otimização do modelo preditor de recorrência em casos de violência contra a mulher

Nesta fase o grupo aplicou os conceitos da disciplina ao **mesmo conjunto de dados da Fase 1**, evoluindo o trabalho anterior. O foco foi usar **algoritmo genético** para otimizar hiperparâmetros de um **Random Forest** classificador (algoritmo com as melhores métricas dentre os usados na Fase 1), com ênfase na **revocação (recall)** do rótulo de recorrência, priorizando a redução de **falsos negativos**.

## Setup rápido

Requisitos: **Python 3.11+** (recomendado).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit         # necessário para o app em app/streamlit_app.py
```

Variáveis de ambiente (opcional): para o módulo `src/llm/llm_explainer.py`, configure `OPENAI_API_KEY` no arquivo `.env` na raiz do projeto (o cliente usa `python-dotenv`).

**Dados:** o script `src/optimize_ga.py` espera o arquivo `data/processed/df_preprocessed.parquet`. Este dataset foi gerado na Fase 1 após o preprocessamento dos dados oriúndos do DATASUS.

## Estrutura do repositório

| Caminho                         | Descrição                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `app/streamlit_app.py`          | Interface Streamlit para carregar modelos, métricas, matriz de confusão e explicações SHAP |
| `notebooks/`                    | Notebooks de análise (ex.: SHAP com Random Forest)                                         |
| `src/optimize_ga.py`            | Otimização por algoritmo genético dos hiperparâmetros do Random Forest                     |
| `src/evaluation/fairness.py`    | Avaliação de métricas por grupo                                                            |
| `src/explain/shap_explainer.py` | Cálculo de valores SHAP para explicabilidade                                               |
| `src/llm/llm_explainer.py`      | Geração de explicações em linguagem natural via API OpenAI                                 |
| `models/`                       | Artefatos serializados (`joblib`) e utilitário `load_model.py`                             |
| `docs/`                         | Documentação complementar do desafio                                                       |

## Dataset

**SINAN/SUS — Violência interpessoal/autoprovocada referente ao ano de 2024**

Para mais informações sobre o dataset usado, acesse o [repositório](https://github.com/fiap-ssp-2025/tech-challenge-1).

## Uso

### Otimização com algoritmo genético

Na raiz do repositório (com o ambiente virtual ativo e `df_preprocessed.parquet` disponível):

```bash
python src/optimize_ga.py
```

### Aplicação Streamlit

```bash
streamlit run app/streamlit_app.py
```

O app permite escolher entre variantes de modelo em `models/` (conforme arquivos `.pkl` gerados pela execução do `src/optimize_ga.py`).

### Preparacao local para fine-tuning com dados sigilosos

Foi adicionada uma etapa inicial de preparo de corpus para uso 100% local:

1. Anonimizacao e separacao de narrativas relevantes (remove segmentos administrativos):

```bash
python src/data_prep/prepare_finetune_corpus.py \
	--input data/Dados_Treinamento/DEAM_2026_preprocessadoII.xlsx \
	--output-dir data/processed/finetune
```

O script extrai exclusivamente o conteudo do bloco `Oitiva(s)` e elimina secoes administrativas como `Das Providencias`, `Despacho` e `Aditamento`.
Caso o arquivo com underscore nao exista, o script usa automaticamente `data/Dados_Treinamento/DEAM2026_preprocessadoII.xlsx`.

Saidas principais em `data/processed/finetune`:
- `dataset_vitima_only.csv` (apenas segmentos classificados como fala da vitima)
- `dataset_vitima_contexto.csv` (vitima + contexto nao administrativo)
- `preparation_report.json` (metricas de extração)

2. Geracao de dataset de instrucoes com rotulo inicial de risco (heuristico):

```bash
python src/data_prep/build_instruction_dataset.py \
	--input data/processed/finetune/dataset_vitima_contexto.csv \
	--output-dir data/processed/finetune
```

Saidas principais:
- `risk_train.jsonl`
- `risk_val.jsonl`
- `risk_label_stats.json`

Observacao: os rotulos de risco gerados sao baseline por regras e devem ser validados por especialistas antes de qualquer uso operacional.

### Notebooks

Abra e execute `notebooks/shap_random_forest_violencia_mulher.ipynb` no Jupyter ou VS Code (kernel apontando para o `.venv`).

### Modelo textual local para narrativas

Para treinar um classificador local de narrativas com `TF-IDF + LogisticRegression` e testar textos livres:

```bash
python src/text_classifier.py train
python src/text_classifier.py predict --text "texto da narrativa aqui"
```

Saidas do treino:
- `models/text_risk_classifier.pkl`
- `models/text_risk_classifier_metrics.json`

Se quiser testar de forma interativa, rode apenas:

```bash
python src/text_classifier.py predict
```

O modelo de texto usa os arquivos `risk_train.jsonl` e `risk_val.jsonl` gerados na etapa de preparacao.

## Integrantes

| Nome                           |
| ------------------------------ |
| Marcelo Arruda de Siqueira     |
| Leonardo Barbosa Nogueira      |
| Jose Flavio Neto               |
| Pedro Matias dos Santos        |
| Wellington Oliveira de Andrade |
