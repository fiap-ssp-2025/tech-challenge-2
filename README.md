# Tech Challenge 1 - Sistema de IA para Diagnóstico de Recorrência em casos de Agressão/Violência contra a mulher.

Neste repositório nós analisamos o dataset do SUS referente aos casos registrados de Violência Interpessoal contra a mulher no ano de 2024 (registro mais atual encontrado na base do DATASUS).

O trabalho foi feito inteiramente em notebooks do Jupyter para facilitar a colaboração entre os autores e o entendimento das conceitos aplicados.

## 🚀 Setup Rápido

```bash
# 1. Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar Jupyter
python -m ipykernel install --user --name tech-challenge

# 4. Iniciar Jupyter
jupyter lab
```

## 📁 Estrutura

```
tech-challenge-1/
├── data/
│   ├── raw/              # Dados brutos (DBC)
│   └── processed/        # Dados processados (parquet)
├── src/
│   ├── notebooks/        # Notebooks Jupyter (executar em ordem)
│   ├── metadata/         # Dicionários e mapeamentos
│   └── utils/           # Utilitários Python
└── docs/                 # Documentação
```

## 📊 Dataset

**SINAN/SUS - Violência Interpessoal/Autoprovocada**

- Fonte: [DATASUS](https://datasus.saude.gov.br/transferencia-de-arquivos/#)
- Seleção: `SINAN > DADOS > Violência doméstica, sexual e outras violências > |Ano| > BR`

## 🔧 Uso

### Notebooks

Execute os notebooks em ordem numérica:
1. `01_extract_sinan_data.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_correlation.ipynb`
4. `04_logistic_regression.ipynb`
5. `05_knn.ipynb`
6. `06_random_forest.ipynb`

## 👥 Integrantes

| Nome |
|------|
| Marcelo Arruda de Siqueira |
| Leonardo Barbosa Nogueira |
| Jose Flavio Neto |
| Pedro Matias dos Santos |
| Rodrigo Oliveira de Andrade |
