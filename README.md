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

### Notebooks

Abra e execute `notebooks/shap_random_forest_violencia_mulher.ipynb` no Jupyter ou VS Code (kernel apontando para o `.venv`).

## Integrantes

| Nome                        |
| --------------------------- |
| Marcelo Arruda de Siqueira  |
| Leonardo Barbosa Nogueira   |
| Jose Flavio Neto            |
| Pedro Matias dos Santos     |
| Rodrigo Oliveira de Andrade |
