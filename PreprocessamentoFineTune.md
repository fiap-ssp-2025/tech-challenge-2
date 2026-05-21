# Preparacao de Dados para Fine-Tuning

## 1. Protecao de dados sensiveis

Os dados utilizados nas pastas de treinamento contêm registros reais de ocorrencia. Por esse motivo, o projeto adota medidas para reduzir risco de exposicao:

- Segredos e configuracoes locais devem ficar no arquivo [.env](.env), que nao e versionado.
- O arquivo [.gitignore](.gitignore) bloqueia dados sensiveis de treinamento, incluindo a pasta de dados brutos ([Dados_Treinamento](Dados_Treinamento)).
- As etapas de preparacao foram desenhadas para anonimizar e filtrar o conteudo antes da geracao dos artefatos finais de treino.

## 2. Abordagem tecnica de preprocessamento

No pipeline atual, a preparacao textual foi feita com **regex + heuristicas de regex**, sem uso de IA generativa para extracao/anonimizacao dos registros DEAM.

Em especial, sao aplicadas regras para:

- extrair apenas blocos de interesse (oitivas),
- descartar blocos administrativos,
- anonimizar padroes sensiveis (Nome, CPF, telefone, e-mail, endereco etc.),
- limpar tokens residuais antes da escrita dos datasets de treino.

## 3. Ordem de execucao da pasta src/data_prep

### 3.1 [src/data_prep/pre_processamento.py](src/data_prep/pre_processamento.py)

Etapa de tratamento tabular inicial do Excel bruto:

- remove linhas indesejadas (ex.: `Historico = BLOQUEADO`, `Natureza Padronizada = Em apuracao`),
- elimina duplicidades por chave de ocorrencia,
- remove colunas administrativas/auxiliares,
- gera o arquivo preprocessado:

`data/Dados_Treinamento/DEAM2026_preprocessadoII.xlsx`

### 3.2 [src/data_prep/prepare_finetune_corpus.py](src/data_prep/prepare_finetune_corpus.py)

Etapa de extracao textual e anonimização:

- le o preprocessado,
- extrai apenas trechos de `Oitiva(s)`,
- remove trechos de providencias/aditamentos e outros blocos administrativos,
- aplica anonimização por regex/heuristica,
- remove registros sem oitiva valida ou sem texto util,
- gera:

`data/processed/finetune/dataset_vitima_only.csv`  
`data/processed/finetune/dataset_vitima_contexto.csv`  
`data/processed/finetune/preparation_report.json`

### 3.3 [src/data_prep/relatos_sem_violência.py](src/data_prep/relatos_sem_violência.py)

Este arquivo foi originalmente executado no **Google Colab** para gerar relatos sinteticos de **SEM_VIOLENCIA** (conflitos relacionais comuns, sem violencia psicologica, fisica, patrimonial ou ameaca).

Objetivo dessa etapa:

- criar exemplos negativos de linguagem cotidiana,
- mesclar esses relatos com dados de ocorrencia,
- reduzir viés de selecao do dataset,
- mitigar risco de **falsos positivos** no fine-tuning.

Saida gerada por esse script:

`relatos_treinamento.csv`

### 3.4 [src/data_prep/merge_relato_training_sets.py](src/data_prep/merge_relato_training_sets.py)

Etapa de mescla do arquivo de relatos com os datasets processados:

- mapeia colunas de `relatos_treinamento.csv` para o schema padrao (`row_id`, `natureza`, `text_anonymized`, `segment_count`),
- concatena com:
	- `dataset_vitima_only.csv`
	- `dataset_vitima_contexto.csv`
- gera arquivos mesclados para treino:

`data/processed/finetune/dataset_vitima_only_merged.csv`  
`data/processed/finetune/dataset_vitima_contexto_merged.csv`

### 3.5 [src/data_prep/build_instruction_dataset.py](src/data_prep/build_instruction_dataset.py)

Etapa final de formatacao para fine-tuning:

- le o CSV final escolhido (normalmente o de contexto, puro ou mesclado),
- aplica rotulagem heuristica de risco (`BAIXO`, `MEDIO`, `ALTO`),
- embaralha e divide treino/validacao,
- gera artefatos finais:

`data/processed/finetune/risk_train.jsonl`  
`data/processed/finetune/risk_val.jsonl`  
`data/processed/finetune/risk_label_stats.json`

## 4. Arquivo final usado no fine-tuning

Os arquivos efetivamente usados para fine-tuning sao os JSONL:

- `risk_train.jsonl` (treino)
- `risk_val.jsonl` (validacao)

Eles sao produzidos a partir do dataset de entrada definido no `build_instruction_dataset.py` (por exemplo, `dataset_vitima_contexto.csv` ou `dataset_vitima_contexto_merged.csv`).
