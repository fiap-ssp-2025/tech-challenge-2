# Relatório Técnico — Otimização do Modelo Preditor de Recorrência de Violência contra a Mulher

[Link do Repositório no Github](https://github.com/fiap-ssp-2025/tech-challenge-2.git)

[Link do Vídeo Explicatório](https://youtu.be/ypwzWRyV6jU)

**Projeto 1 — Otimização de Modelos para Saúde da Mulher**
FIAP — Pós-graduação em IA para Devs · Tech Challenge Fase 2
Grupo SSPDF
| Integrante |
|---|
| Marcelo Arruda de Siqueira |
| Leonardo Barbosa Nogueira |
| Jose Flavio Neto |
| Pedro Matias dos Santos |
| Wellington Oliveira de Andrade |

---

## 1. Contexto e objetivo

Na Fase 1, o grupo desenvolveu modelos de classificação sobre o dataset **SINAN/SUS — Violência interpessoal/autoprovocada (2024)**, obtido via DATASUS. O dataset contém 57.022 registros de notificações de violência contra a mulher, com 37 features após pré-processamento (tipos de agressão, tipos de violência, relação com o agressor, dados demográficos, contexto da ocorrência). A variável-alvo é `OUT_VEZES` (recorrência da violência).

Na Fase 2, o objetivo foi evoluir o trabalho anterior aplicando:

1. **Algoritmo genético** para otimização de hiperparâmetros do Random Forest (melhor modelo da Fase 1).
2. **Integração com LLM** (GPT-4o-mini via API OpenAI) para gerar explicações em linguagem natural dos resultados, com prompt engineering sensível ao gênero.
3. **Análise de equidade** por grupo demográfico (raça), incorporada diretamente na função fitness.
4. **Explicabilidade** via SHAP (TreeExplainer) para identificar os fatores de risco mais relevantes.
5. **Persistência de dados para fine-tuning** em MongoDB, preparando a base para a Fase 3.

---

## 2. Implementação do algoritmo genético

### 2.1 Representação dos genes

Cada indivíduo do algoritmo genético é um dicionário de hiperparâmetros do `RandomForestClassifier` (scikit-learn):

| Gene                | Valores possíveis   |
| ------------------- | ------------------- |
| `n_estimators`      | 100, 200, 300, 400  |
| `max_depth`         | 5, 10, 15, 20, None |
| `min_samples_split` | 2, 4, 6, 8          |
| `min_samples_leaf`  | 1, 2, 3, 4          |
| `max_features`      | sqrt, log2, None    |

Total de combinações possíveis: 4 × 5 × 4 × 4 × 3 = 960.

### 2.2 Operadores genéticos

| Operador       | Implementação                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Criação**    | Indivíduo gerado por seleção aleatória uniforme de cada gene (`create_individual`)                                            |
| **Seleção**    | Torneio: amostra `tournament_size` indivíduos e seleciona o de maior fitness (`select_parent`)                                |
| **Cruzamento** | Uniforme gene-a-gene: para cada hiperparâmetro, escolhe aleatoriamente o valor de um dos dois pais (`crossover`)              |
| **Mutação**    | Por gene: cada gene tem probabilidade `mutation_rate` de ser substituído por um valor aleatório do espaço de busca (`mutate`) |
| **Elitismo**   | Os `elitism` melhores indivíduos são preservados intactos na próxima geração                                                  |

### 2.3 Função fitness

A função fitness combina cinco componentes em um score composto, priorizando recall e incorporando equidade entre grupos demográficos:

```
fitness = 0.35 × recall + 0.25 × F1 + 0.15 × especificidade + 0.10 × ROC-AUC + 0.15 × (1 - recall_gap)
```

| Componente                | Peso | Justificativa                                                       |
| ------------------------- | ---- | ------------------------------------------------------------------- |
| Recall                    | 0.35 | Prioridade máxima: minimizar falsos negativos em casos de violência |
| F1-score                  | 0.25 | Equilíbrio entre precisão e sensibilidade                           |
| Especificidade            | 0.15 | Reduzir falsos positivos em triagem                                 |
| ROC-AUC                   | 0.10 | Capacidade discriminativa geral                                     |
| Fairness (1 - recall_gap) | 0.15 | Penaliza desigualdade de desempenho entre grupos raciais            |

O `recall_gap` é calculado como a diferença entre o maior e o menor recall entre os grupos da coluna `CS_RACA` (raça), avaliado a cada indivíduo durante a evolução — não apenas ao final do processo.

### 2.4 Configurações dos experimentos

Foram executados 3 experimentos com configurações distintas do algoritmo genético, além do baseline:

| Experimento        | População | Gerações | Taxa de mutação | Elitismo | Torneio |
| ------------------ | --------- | -------- | --------------- | -------- | ------- |
| `exp1_conservador` | 8         | 5        | 0.10            | 2        | 3       |
| `exp2_moderado`    | 14        | 10       | 0.20            | 3        | 4       |
| `exp3_agressivo`   | 20        | 15       | 0.35            | 4        | 5       |

### 2.5 Resultados da otimização

| Experimento            | Recall     | F1         | ROC-AUC    | Precisão | Especificidade | Recall Gap |
| ---------------------- | ---------- | ---------- | ---------- | -------- | -------------- | ---------- |
| **Baseline**           | 0.7453     | 0.7261     | 0.7615     | 0.7079   | 0.6518         | 0.1658     |
| **Exp1 (conservador)** | 0.8244     | 0.7534     | 0.7767     | 0.6936   | 0.5877         | 0.0735     |
| **Exp2 (moderado)**    | 0.8227     | 0.7531     | 0.7767     | 0.6943   | 0.5900         | **0.0595** |
| **Exp3 (agressivo)**   | **0.8280** | **0.7546** | **0.7782** | 0.6931   | 0.5849         | 0.0735     |

---

## 3. Análise de métricas prioritárias: sensibilidade vs especificidade

### 3.1 Trade-off observado

Os três modelos otimizados apresentaram ganho significativo em **recall** (de 0.745 para ~0.825, aumento de ~10.5 pontos percentuais) com redução moderada em **especificidade** (de 0.652 para ~0.588, queda de ~6.4 pp).

Esse trade-off é aceitável no contexto do projeto: a consequência de um falso negativo (não identificar uma mulher em risco de recorrência) é significativamente mais grave do que a de um falso positivo (acionar acompanhamento para uma mulher que não teria recorrência). A função fitness reflete essa prioridade ao atribuir peso 0.35 ao recall contra 0.15 à especificidade.

### 3.2 Precisão e F1

A precisão caiu marginalmente (de 0.708 para ~0.694), enquanto o F1 subiu (de 0.726 para ~0.754). O aumento do F1 confirma que o ganho em recall superou a perda de precisão em termos de equilíbrio geral.

---

## 4. Comparativo: baseline vs modelos otimizados

O baseline é um `RandomForestClassifier` com hiperparâmetros default do scikit-learn (100 árvores, sem limite de profundidade, `min_samples_split=2`, `min_samples_leaf=1`).

### 4.1 Ganhos consolidados (Exp3 vs Baseline)

| Métrica        | Baseline | Exp3 (melhor) | Variação   |
| -------------- | -------- | ------------- | ---------- |
| Recall         | 0.7453   | 0.8280        | **+11.1%** |
| F1             | 0.7261   | 0.7546        | +3.9%      |
| ROC-AUC        | 0.7615   | 0.7782        | +2.2%      |
| Precisão       | 0.7079   | 0.6931        | −2.1%      |
| Especificidade | 0.6518   | 0.5849        | −10.3%     |
| Recall Gap     | 0.1658   | 0.0735        | **−55.7%** |

Os modelos otimizados detectam mais casos reais de recorrência e apresentam desempenho mais equilibrado entre grupos raciais, ao custo de uma taxa ligeiramente maior de falsos positivos.

### 4.2 Convergência entre experimentos

Os três experimentos convergiram para métricas próximas, indicando robustez da solução. A principal diferença foi no recall gap: o `exp2_moderado` obteve o menor gap (0.0595), sugerindo que configurações com mutação moderada e população intermediária favorecem a equidade.

---

## 5. Integração com LLMs

### 5.1 Modelo utilizado

**GPT-4o-mini** via API OpenAI, escolhido pelo equilíbrio entre custo, latência e qualidade para o contexto de geração de texto orientativo. Parâmetros de geração: `temperature=0.2`, `max_tokens=200` (resumo) e `max_tokens=400` (detalhamento).

### 5.2 Arquitetura de prompts

A integração segue uma abordagem de duas chamadas separadas por simulação de risco:

1. **Resumo** (`_generate_summary`): gera um parágrafo de no máximo 3 frases em tom de alerta profissional, citando indicadores presentes, motivo do nível de risco e orientação ao atendente.
2. **Detalhamento** (`_generate_details`): explica cada fator SHAP individualmente, indicando se aumenta ou reduz o risco e o que significa na prática para o atendimento.

Ambas as chamadas compartilham um **system prompt base** (`_SYSTEM_BASE`) estruturado em três eixos obrigatórios:

#### 5.2.1 Contexto médico feminino

O prompt instrui a LLM a:

- Considerar fatores de saúde específicos da mulher (trauma, saúde reprodutiva, impacto psicológico da violência).
- Relacionar fatores de risco a vulnerabilidades concretas (gestação, puerpério, dependência econômica).
- Orientar encaminhamentos compatíveis: acolhimento psicossocial, serviços de saúde da mulher, rede de proteção.

#### 5.2.2 Sensibilidade a questões de gênero

- Proibição explícita de linguagem que culpabilize a vítima.
- Reconhecimento da violência como fenômeno estrutural, não falha individual.
- Terminologia respeitosa: "mulher em situação de violência", nunca de forma redutora.
- Reforço da autonomia da mulher nas orientações ao atendente.

#### 5.2.3 Privacidade e confidencialidade

- Proibição de reproduzir dados pessoais, nomes, endereços ou informação identificável.
- Tratamento sigiloso conforme protocolos de atendimento.
- Lembrete ao atendente, quando pertinente, sobre sigilo e segurança da mulher.

#### 5.2.4 Restrições adicionais

- Não explicar SHAP ou o funcionamento interno do modelo.
- Não fazer introduções genéricas.
- Linguagem direta e profissional.

### 5.3 Fluxo de dados para a LLM

```
Simulação no Streamlit
  → Modelo Random Forest gera probabilidade de recorrência
  → SHAP TreeExplainer calcula valores SHAP para o caso
  → Top 5 features por |SHAP| são extraídas
  → LLM recebe: probabilidade, classificação de risco (baixo/moderado/alto), lista de features com impacto SHAP
  → LLM retorna: resumo + detalhamento
  → Resposta exibida no Streamlit
  → Par (prompt, resposta) salvo no MongoDB para fine-tuning futuro
```

### 5.4 Avaliação da qualidade

A qualidade das respostas foi avaliada qualitativamente nos seguintes critérios:

| Critério                      | Avaliação                                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Precisão**                  | As respostas refletem corretamente a direção dos fatores SHAP (positivo = aumenta risco, negativo = reduz)     |
| **Sensibilidade cultural**    | Terminologia adequada, sem culpabilização, reconhecimento da violência como fenômeno estrutural                |
| **Adequação ao público-alvo** | Linguagem voltada para servidores/atendentes da assistência social, com orientações práticas de encaminhamento |
| **Concisão**                  | Resumo limitado a 3 frases; detalhamento em formato de lista por fator                                         |

### 5.5 Preparação para fine-tuning (Fase 3)

Cada interação com a LLM é salva automaticamente no MongoDB (collection `tech_challenge.finetuning`) no formato:

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "created_at": "2026-03-24T..."
}
```

Esse formato é compatível com o formato JSONL exigido pela OpenAI para fine-tuning. Na Fase 3, basta exportar a collection e submeter ao processo de fine-tuning. O MongoDB foi escolhido por oferecer escalabilidade para cenários de produção com múltiplas instâncias, ao custo mínimo de um container Docker.

---

## 6. Análise de equidade

### 6.1 Métrica utilizada

A equidade é medida pelo **recall gap**: diferença entre o maior e o menor recall entre os grupos da coluna `CS_RACA` (raça). A avaliação é feita por grupo no módulo `src/evaluation/fairness.py`, que calcula recall, especificidade, precisão e F1 por subgrupo.

### 6.2 Resultado

| Modelo             | Recall Gap |
| ------------------ | ---------- |
| Baseline           | 0.1658     |
| Exp1 (conservador) | 0.0735     |
| Exp2 (moderado)    | **0.0595** |
| Exp3 (agressivo)   | 0.0735     |

O baseline apresentava um recall gap de 16.6%, indicando que o grupo demográfico com pior recall tinha detecção ~17 pontos percentuais inferior ao grupo com melhor recall. Após a otimização com a componente de fairness na fitness, o gap caiu para 5.9–7.4%, uma redução de 55–64%.

### 6.3 Mecanismo

A incorporação de equidade acontece dentro do loop evolutivo do GA: a cada avaliação de indivíduo, o modelo candidato é avaliado por grupo demográfico, o recall gap é calculado e entra na fitness como penalidade (`1 - recall_gap`). Indivíduos que geram modelos desiguais recebem scores mais baixos e têm menor probabilidade de serem selecionados como pais.

### 6.4 Visualização

A interface Streamlit exibe, na seção "Avaliação e Equidade", gráficos de barras com recall e especificidade por grupo racial, além da tabela detalhada com tamanho de cada grupo, TP, FP, FN e TN.

---

## 7. Explicabilidade (SHAP)

A explicabilidade do modelo foi implementada via **SHAP (SHapley Additive exPlanations)** usando `TreeExplainer`, que calcula contribuições exatas para modelos baseados em árvore.

### 7.1 Importância global (Top 10)

Extraída do notebook `notebooks/shap_random_forest_violencia_mulher.ipynb`:

| Pos. | Feature               | Mean   |
| ---- | --------------------- | ------ |
| 1    | `REL_PARCEIRO_INTIMO` | 0.0781 |
| 2    | `REL_CONHECIDO`       | 0.0747 |
| 3    | `AG_AMEACA`           | 0.0428 |
| 4    | `VIOL_PSICO`          | 0.0392 |
| 5    | `REL_FAMILIAR`        | 0.0263 |
| 6    | `VIOL_FISIC`          | 0.0173 |
| 7    | `ESCOLARIDADE`        | 0.0167 |
| 8    | `AUTOR_ALCO`          | 0.0137 |
| 9    | `AUTOR_SEXO`          | 0.0134 |
| 10   | `DEF_TRANS`           | 0.0127 |

O tipo de relação com o agressor (parceiro íntimo, conhecido, familiar) domina o ranking, seguido por meios de agressão (ameaça) e tipo de violência (psicológica). Esses achados são consistentes com a literatura sobre fatores de risco para recorrência de violência doméstica.

### 7.2 Uso na interface

Na simulação individual (Streamlit), os 5 fatores com maior |SHAP| para aquele caso são exibidos em tabela e repassados à LLM para gerar a orientação contextualizada. O formulário expõe as 20 features de maior importância global, permitindo que o atendente informe as variáveis mais relevantes para a predição.

---

## 8. Desafios enfrentados e soluções implementadas

| Desafio                                                                                                                | Solução                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Data leakage na versão inicial**: o modelo final era treinado em todo o dataset (incluindo teste), inflando métricas | Corrigido: modelo final treinado apenas em `X_train`, avaliado em `X_test` (split 70/30 estratificado)                        |
| **Equidade avaliada apenas pós-GA**: a fairness era calculada depois da otimização, sem influenciar a busca            | Incorporada na função fitness com peso 0.15, avaliada a cada indivíduo durante a evolução                                     |
| **Configuração única do GA**: apenas uma execução com parâmetros fixos                                                 | Parametrização completa do GA; 3 experimentos com configurações progressivamente mais exploratórias                           |
| **Ausência de baseline formal**: sem referência para medir o ganho da otimização                                       | Função `train_baseline()` treina RF com parâmetros default e salva artefato; tabela comparativa gerada automaticamente        |
| **LLM gerava respostas longas e genéricas**: texto não adequado para o público-alvo (atendentes)                       | Prompt reestruturado em duas chamadas separadas (resumo + detalhamento), com limites rígidos de tokens e instruções de estilo |
| **Hot-reload do Streamlit não reimportava módulos atualizados**: cache Python mantinha versão antiga da LLM            | Adicionado `importlib.reload` para forçar reimportação a cada execução                                                        |
| **Formulário expunha poucas features**: apenas 3-5 campos, desperdiçando capacidade preditiva                          | Expandido para as 20 features de maior importância SHAP, com mapeamento correto para os códigos do LabelEncoder               |

---

## 9. Considerações éticas

### 9.1 Privacidade

- O dataset utilizado (SINAN/SUS) é de acesso público e não contém informações identificáveis (nomes, CPF, endereços).
- O system prompt da LLM proíbe explicitamente a reprodução de dados pessoais nas respostas.
- As respostas salvas para fine-tuning no MongoDB contêm apenas os prompts e respostas da LLM, sem dados identificáveis da vítima.
- O `.gitignore` exclui o diretório `data/` e arquivos `.env` do versionamento.

### 9.2 Bias e equidade

- O modelo baseline apresentava recall gap de 16.6% entre grupos raciais, indicando viés de desempenho.
- A incorporação de fairness na função fitness reduziu esse gap para 5.9–7.4%, demonstrando que é possível otimizar desempenho e equidade simultaneamente.
- O trade-off observado: a melhoria em equidade teve custo marginal em especificidade (~6 pp), compensado pelo ganho em recall (~10 pp).
- O módulo `fairness.py` permite monitorar métricas desagregadas por grupo, possibilitando auditoria contínua do modelo.

### 9.3 Impacto social

- O sistema tem como público-alvo servidores e atendentes da assistência social, não a vítima diretamente.
- As orientações da LLM reforçam a autonomia da mulher e evitam culpabilização.
- O modelo é uma ferramenta de apoio à decisão, não um sistema autônomo de classificação de risco. A decisão final permanece com o profissional de atendimento.
- O prompt engineering incorpora três eixos (contexto médico feminino, sensibilidade de gênero, privacidade) para garantir que as orientações sejam adequadas ao contexto.

### 9.4 Limitações reconhecidas

- O dataset é referente a 2024 e restrito a notificações no SINAN, o que implica subnotificação significativa.
- A variável-alvo (`OUT_VEZES`) representa recorrência registrada, não recorrência real — casos que não geraram nova notificação não são capturados.
- O modelo não deve ser usado isoladamente para decisões de alto impacto sem validação por profissional qualificado.

---

## 10. Arquitetura da solução

### 10.1 Visão geral

A solução opera localmente com os seguintes componentes:

```
┌─────────────────────────────────────────────────────┐
│                    Usuário (Atendente)               │
│                          │                           │
│                    ┌─────▼─────┐                     │
│                    │ Streamlit │                      │
│                    │   (UI)    │                      │
│                    └─────┬─────┘                     │
│                          │                           │
│          ┌───────────────┼───────────────┐           │
│          ▼               ▼               ▼           │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│   │   Random   │  │    SHAP    │  │  LLM (API  │    │
│   │   Forest   │  │ Explainer  │  │  OpenAI)   │    │
│   │  (.pkl)    │  │            │  │            │    │
│   └────────────┘  └────────────┘  └──────┬─────┘    │
│                                          │           │
│                                   ┌──────▼─────┐    │
│                                   │  MongoDB   │    │
│                                   │ (Docker)   │    │
│                                   └────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 10.2 Stack tecnológica

| Componente      | Tecnologia               | Versão          |
| --------------- | ------------------------ | --------------- |
| Linguagem       | Python                   | 3.11+           |
| ML              | scikit-learn             | ≥ 1.3.0         |
| Explicabilidade | SHAP                     | ≥ 0.43.0        |
| LLM             | OpenAI API (GPT-4o-mini) | ≥ 1.0.0         |
| Interface       | Streamlit                | última          |
| Serialização    | joblib                   | ≥ 1.2.0         |
| Persistência    | MongoDB 7 (via Docker)   | pymongo ≥ 4.6.0 |
| Ambiente        | venv                     | nativo          |

### 10.3 Estrutura do repositório

```
tech-challenge-2/
├── app/
│   └── streamlit_app.py          # Interface: simulação, detalhes, equidade, comparação
├── src/
│   ├── optimize_ga.py            # Algoritmo genético + baseline + pipeline de experimentos
│   ├── db.py                     # Conexão MongoDB (singleton)
│   ├── evaluation/
│   │   └── fairness.py           # Métricas por grupo demográfico
│   ├── explain/
│   │   └── shap_explainer.py     # SHAP TreeExplainer
│   └── llm/
│       └── llm_explainer.py      # Prompt engineering + chamadas à API OpenAI
├── models/                       # Artefatos serializados (.pkl)
├── results/                      # comparison.csv, comparison.json
├── notebooks/                    # Análise SHAP exploratória
├── data/processed/               # df_preprocessed.parquet
├── docs/                         # Documentação e enunciado
├── docker-compose.yml            # MongoDB 7
├── requirements.txt
├── .env.example
└── README.md
```

### 10.4 Artefatos do modelo

Cada modelo é salvo como um dicionário serializado com `joblib`, contendo:

- `model`: instância treinada do `RandomForestClassifier`
- `features`: lista de nomes das features esperadas
- `params`: hiperparâmetros utilizados (ou "default" para o baseline)
- `fitness`: score fitness do melhor indivíduo
- `metrics`: recall, F1, ROC-AUC, precisão, especificidade
- `search_metrics`: métricas durante a busca do GA
- `group_metrics`: métricas por grupo racial + recall gap
- `ga_config`: configuração do experimento (população, gerações, mutação, etc.)

### 10.5 Docker Compose

O `docker-compose.yml` contém apenas o MongoDB. O Streamlit roda localmente via `streamlit run`. Essa separação simplifica o desenvolvimento e não exige containerização da aplicação Python.

```yaml
services:
  mongo:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
```

---

## 11. Ferramentas utilizadas e justificativas

| Ferramenta        | Justificativa                                                                                                                                                  |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Random Forest** | Melhor modelo da Fase 1 para o dataset; robusto a overfitting com os hiperparâmetros corretos; compatível com SHAP TreeExplainer                               |
| **SHAP**          | Explicações por contribuição de cada feature, com garantias teóricas (valores de Shapley); TreeExplainer é exato para modelos de árvore                        |
| **GPT-4o-mini**   | Custo baixo (~$0.15/1M tokens), latência aceitável, qualidade suficiente para geração de texto orientativo; suporta instruções complexas de prompt engineering |
| **MongoDB**       | Armazenamento de documentos JSON sem esquema fixo; escalável para produção; free tier disponível (Atlas); formato compatível com pipeline de fine-tuning       |
| **Streamlit**     | Prototipagem rápida de interfaces; adequado para dashboards internos; sem necessidade de frontend separado                                                     |
