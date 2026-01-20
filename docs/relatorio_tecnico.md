# Análise Preditiva da Recorrência da Violência contra a Mulher  
## Um Estudo com Dados do SINAN

---

## 1. Introdução

A violência contra a mulher constitui um problema complexo de saúde pública e segurança, marcado por múltiplos fatores sociais, relacionais e institucionais. Um dos aspectos mais críticos desse fenômeno é a **recorrência da violência**, isto é, a repetição de episódios envolvendo a mesma vítima ao longo do tempo.

Neste trabalho, investiga-se a seguinte questão de pesquisa:

> **É possível prever a recorrência da violência contra a mulher a partir das características da vítima, do agressor, do vínculo relacional e do tipo de violência notificada?**

Para isso, foram utilizados dados públicos do **Sistema de Informação de Agravos de Notificação (SINAN)**, com foco em notificações de violência interpessoal contra mulheres no ano de 2024.

---

## 2. Base de Dados e Análise Exploratória (EDA)

### 2.1 Descrição geral do dataset

O dataset analisado contém **662.803 registros** iniciais referentes a notificações de violência interpessoal e autoprovocada do SINAN para o ano de 2024, com **160 variáveis** disponíveis.

Após a aplicação dos filtros para delimitar o escopo do estudo:
- **Sexo feminino** (`CS_SEXO = 'F'`)
- **Lesões não autoprovocadas** (`LES_AUTOP = '2'`)

O dataset foi reduzido para **294.393 registros**, representando aproximadamente 44% do conjunto original.

As variáveis disponíveis abrangem:
- **Perfil da vítima**: sexo, raça/cor, escolaridade, deficiência, identidade de gênero, orientação sexual, situação conjugal, ciclo de vida
- **Perfil do agressor**: sexo, uso de álcool
- **Relação entre vítima e agressor**: múltiplas categorias (parceiro íntimo, familiar, conhecido, institucional, outros)
- **Tipos de violência**: física, psicológica, sexual, financeira, negligência, tortura, tráfico, infantil, legal, outras
- **Meios de agressão**: força física, enforcamento, objeto, corte, quente, envenenamento, fogo, ameaça, outros
- **Encaminhamentos institucionais**: saúde, tutela, vara, abrigo, sentinela, DEAM, DPCA, delegacia, MPU, atendimento à mulher, CREAS, IML, outros
- **Contexto do evento**: circunstância da lesão (CID-10)

A variável alvo definida foi:

- **`OUT_VEZES`**: indica se a violência ocorreu **outras vezes** (proxy de recorrência), codificada como binária (1 = Sim, 0 = Não).

---

### 2.2 Distribuição e padrões iniciais

A análise exploratória revelou:

- Grande predominância de **variáveis categóricas e binárias**
- Presença frequente de códigos especiais como `8` (não se aplica), `9` (ignorado), valores vazios e textos livres, representando informações ausentes ou preenchimento inconsistente
- Forte concentração de episódios associados a **parceiros íntimos, familiares ou conhecidos**, evidenciando o caráter relacional da violência
- Alta incidência de **violência psicológica e física**, frequentemente coexistentes
- Distribuição geográfica concentrada nos estados de São Paulo, Rio de Janeiro, Paraná e Minas Gerais

Esses achados já sugerem que a recorrência da violência pode estar associada a **vínculos sociais próximos**, hipótese posteriormente testada estatisticamente.

---

## 3. Estratégias de Pré-processamento

### 3.1 Seleção de variáveis relevantes

Dado o grande número de variáveis disponíveis (160) e a ausência de documentação completa para todas elas, foi criado um **dicionário de dados próprio** (`dicionario_campos_sinan.json`) que classificou cada campo como relevante ou não para o estudo. Esta estratégia permitiu:

- Focar em variáveis com potencial preditivo. Isso nos fez excluir boa partes dos campos que se referiam informações relacionadas ao pós-eventos (encaminhamentos, evolução etc.).
- Eliminar campos redundantes ou de baixa qualidade
- Reduzir a dimensionalidade do problema

---

### 3.2 Limpeza e tratamento de valores inconsistentes

Foram adotadas as seguintes estratégias:

- **Remoção de registros com valores ausentes**: após o pré-processamento completo, registros com qualquer valor ausente foram removidos, resultando em **57.022 registros** finais para modelagem
- **Conversão de códigos binários**: valores do tipo `1` (Sim) e `2` (Não) foram convertidos para formato binário numérico (1/0)
- **Tratamento de códigos especiais**: valores como `8` (não se aplica), `9` (ignorado), strings vazias e `NaN` foram tratados como ausentes, exceto quando semanticamente relevantes
- **Exclusão de campos textuais livres**: campos com alta cardinalidade e baixo valor preditivo foram descartados

---

### 3.3 Binarização e engenharia de atributos

#### 3.3.1 Tipos de violência

As variáveis de tipo de violência (`VIOL_*`) foram convertidas para formato binário, permitindo a identificação clara da ocorrência de cada tipo:
- `VIOL_FISIC`, `VIOL_PSICO`, `VIOL_SEXU`, `VIOL_FINAN`, `VIOL_NEGLI`, `VIOL_TORT`, `VIOL_TRAF`, `VIOL_INFAN`, `VIOL_LEGAL`, `VIOL_OUTR`

#### 3.3.2 Agregação semântica de relações

As múltiplas variáveis de relação com o agressor foram agrupadas em categorias interpretáveis:

- **`REL_PARCEIRO_INTIMO`**: conjugue, ex-conjugue, namorado, ex-namorado
- **`REL_FAMILIAR`**: pai, mãe, padrasto, madrasta, irmão, filho
- **`REL_CONHECIDO`**: conhecido, cuidador, desconhecido
- **`REL_INSTITUCIONAL`**: patrão, trabalho, institucional, policial, proprietário
- **`REL_OUTROS_FLAG`**: outros

Esta transformação reduziu a dimensionalidade e aumentou a interpretabilidade.

#### 3.3.3 Circunstância da lesão (CID-10)

O campo `CIRC_LESAO` foi agrupado por **família CID** (ex.: Y04, Y05, Y09, T74), extraindo o código de três caracteres (letra + dois dígitos) para reduzir a esparsidade e criar categorias mais interpretáveis.

#### 3.3.4 Escolaridade

A variável `CS_ESCOL_N` foi reagrupada em categorias sucintas:
- `FUNDAMENTAL_INCOMPLETO`: analfabeto, 1ª a 4ª série incompleta, 5ª a 8ª série incompleta
- `ENSINO_MEDIO_INCOMPLETO`: ensino fundamental completo, ensino médio incompleto
- `ENSINO_MEDIO_COMPLETO`: ensino médio completo
- `ENSINO_SUPERIOR`: educação superior (incompleta ou completa)

#### 3.3.5 Variáveis temporais

A partir da data de ocorrência (`DT_OCOR`), foram criadas:
- **`DIA_SEMANA_OCOR`**: dia da semana (SEG, TER, QUA, QUI, SEX, SAB, DOM)
- **`MES_OCOR`**: mês do ano (JAN a DEZ)

#### 3.3.6 Outras transformações

- **Estados (UF)**: códigos numéricos do IBGE foram convertidos para siglas (SP, RJ, PR, etc.)
- **Variáveis categóricas**: mapeamento de códigos para valores legíveis (ex.: raça, identidade de gênero, situação conjugal, ciclo de vida)

---

### 3.4 Dataset final

Após todo o pré-processamento, o dataset final para modelagem contém:
- **57.022 registros**
- **37 features** (variáveis preditoras)
- **1 target** (variável alvo: `OUT_VEZES`)

A distribuição da variável alvo é aproximadamente balanceada:
- **Recorrência (1)**: 30.277 casos (53,1%)
- **Não recorrência (0)**: 26.745 casos (46,9%)

---

## 4. Análise de Associação Estatística

Antes da modelagem, foi aplicado o **teste qui-quadrado de independência** entre cada variável explicativa e a variável alvo (`OUT_VEZES`). O teste foi aplicado apenas a variáveis com **10 ou menos categorias** para garantir adequação estatística.

### 4.1 Principais associações identificadas

Variáveis com associação estatisticamente significativa (p < 0,001), ordenadas por estatística qui-quadrado:

| Rank | Variável | Descrição | χ² | p-value | Graus de Liberdade |
|------|----------|-----------|----|---------|-------------------|
| 1 | `REL_CONHECIDO` | Relação: Conhecido | 6.368,51 | < 0,001 | 1 |
| 2 | `REL_PARCEIRO_INTIMO` | Relação: Parceiro Íntimo | 5.040,73 | < 0,001 | 1 |
| 3 | `AG_AMEACA` | Meio de Agressão: Ameaça | 2.364,86 | < 0,001 | 1 |
| 4 | `VIOL_PSICO` | Tipo de Violência: Psicológica | 2.219,61 | < 0,001 | 1 |
| 5 | `AUTOR_SEXO` | Sexo do Agressor | 1.636,84 | < 0,001 | 3 |
| 6 | `SIT_CONJUG` | Situação Conjugal | 723,67 | < 0,001 | 4 |
| 7 | `VIOL_FINAN` | Tipo de Violência: Financeira | 707,55 | < 0,001 | 1 |
| 8 | `REL_INSTITUCIONAL` | Relação: Institucional | 655,34 | < 0,001 | 1 |
| 9 | `AG_ENFOR` | Meio de Agressão: Enforcamento | 461,56 | < 0,001 | 1 |
| 10 | `CICL_VID` | Ciclo de Vida | 453,96 | < 0,001 | 5 |

*Nota: Todas as análises foram realizadas com n = 57.022 observações.*

### 4.2 Interpretação dos resultados

Os resultados indicam que:

- **Relação com o agressor** apresenta as associações mais fortes, especialmente parceiro íntimo e conhecido
- **Tipos de violência** (psicológica e financeira) estão fortemente associados à recorrência
- **Perfil do agressor** (sexo) e **perfil da vítima** (situação conjugal, ciclo de vida) também apresentam associações significativas
- **Meios de agressão** (ameaça, enforcamento) estão relacionados à recorrência

Dado o grande tamanho amostral (n = 57.022), reconhece-se que o p-valor tende a ser muito pequeno; portanto, os resultados foram utilizados como **filtro informativo** para identificar variáveis relevantes, e não como evidência causal.

---

## 5. Modelagem Preditiva

### 5.1 Preparação dos dados

#### 5.1.1 Encoding e normalização

- **Encoding categórico**: todas as variáveis categóricas foram codificadas usando `LabelEncoder` do scikit-learn
- **Normalização**: aplicação de `StandardScaler` para padronizar as features, essencial para modelos sensíveis à escala (Regressão Logística e KNN)
- **Divisão treino/teste**: 70% para treino (39.915 amostras) e 30% para teste (17.107 amostras), com estratificação para manter a proporção da variável alvo

---

### 5.2 Modelos avaliados

Foram testados **três modelos** de classificação binária:

#### a) Regressão Logística
- Modelo linear, altamente interpretável
- Adequado para variáveis binárias e categóricas
- Permite análise direta do impacto das variáveis através dos coeficientes
- Permite o balanceamento para viés em não-recorrente e recorrente.
- **Otimização**: busca aleatória de hiperparâmetros (C, penalty, class_weight) via `RandomizedSearchCV`

#### b) K-Nearest Neighbors (KNN)
- Modelo baseado em distância, não paramétrico
- Utilizado como comparação de desempenho
- **Otimização**: seleção do número ótimo de vizinhos (k) através de validação cruzada 5-fold, testando valores de k de 1 a 29 (ímpares)

#### c) Random Forest
- Modelo baseado em árvores de decisão
- Segundo a literatura, robusto a overfitting e capaz de capturar interações não lineares
- **Otimização**: busca aleatória de hiperparâmetros (n_estimators, max_depth, min_samples_split, min_samples_leaf) via `RandomizedSearchCV`

---

### 5.3 Resultados obtidos

A tabela abaixo apresenta as métricas de desempenho para cada modelo no conjunto de teste:

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|-------|----------|----------|--------|----------|---------|
| **Regressão Logística** | 70,06% | 70,03% | 70,06% | 69,95% | **76,00%** |
| **KNN** | 70,16% | 70,26% | 70,16% | 69,92% | 75,58% |
| **Random Forest** | **71,87%** | **72,22%** | **71,87%** | **71,51%** | **78,50%** |

#### 5.3.1 Análise comparativa

- **Random Forest** apresentou o melhor desempenho geral em todas as métricas
- **Regressão Logística** e **KNN** apresentaram desempenhos muito similares, com diferenças inferiores a 0,2 pontos percentuais
- Todos os modelos superaram significativamente o desempenho aleatório (50% de acurácia)
- A métrica **ROC-AUC** indica capacidade discriminativa moderada a boa (0,76-0,79), superior à acurácia, sugerindo que os modelos conseguem distinguir melhor entre classes do que a acurácia isolada indicaria

---

### 5.4 Escolha do modelo final

Embora o **Random Forest** tenha apresentado o melhor desempenho absoluto, a **Regressão Logística** foi escolhida como modelo principal devido a:

1. **Interpretabilidade**: permite análise direta dos coeficientes e odds ratios, essencial para compreensão do fenômeno em contexto de saúde pública
2. **Transparência**: facilita a explicação dos resultados para stakeholders não técnicos
3. **Menor custo computacional**: treinamento e inferência mais rápidos
4. **Estabilidade**: menor variância nas predições, importante para aplicações práticas
5. **Desempenho adequado**: diferença de apenas 1,8 pontos percentuais em relação ao Random Forest, compensada pelos benefícios de interpretabilidade

O **KNN** foi mantido como **modelo comparativo**, reforçando a robustez dos achados. O **Random Forest** foi documentado como alternativa para cenários onde o desempenho máximo é prioritário sobre a interpretabilidade.

---

## 6. Interpretação dos Resultados

### 6.1 Capacidade preditiva

Os resultados indicam que é **possível prever a recorrência da violência com desempenho moderado**, significativamente superior ao acaso (50%). A acurácia de aproximadamente **70%** e ROC-AUC de **76%** sugerem que o modelo identifica padrões relevantes, embora exista espaço para melhorias.

### 6.2 Fatores associados à recorrência

Com base na análise de associação estatística e na modelagem, os fatores mais associados à recorrência incluem:

1. **Vínculo próximo com o agressor**
   - Parceiro íntimo (associação mais forte)
   - Conhecido
   - Familiar

2. **Tipos de violência**
   - Violência psicológica (forte associação)
   - Violência financeira
   - Negligência

3. **Perfil do agressor**
   - Sexo do agressor
   - Uso de álcool

4. **Perfil da vítima**
   - Escolaridade
   - Raça/cor
   - Situação conjugal
   - Identidade de gênero
   - Ciclo de vida

5. **Contexto e atendimento**
   - Atendimento à mulher
   - Assistência social
   - Meios de agressão (ameaça, enforcamento)

### 6.3 Implicações práticas

Esses achados são coerentes com a literatura sobre violência doméstica e reforçam a importância de:

- **Intervenções precoces**, especialmente em casos envolvendo relações íntimas
- **Atenção especial** a violências não físicas (psicológica, financeira), que podem ser subnotificadas mas estão fortemente associadas à recorrência
- **Políticas públicas** que considerem o perfil da vítima e do agressor na identificação de casos de alto risco
- **Integração de serviços** (saúde, assistência social, justiça) para atendimento integral

---

## 7. Limitações e Trabalhos Futuros

### 7.1 Limitações do estudo

1. **Variável alvo**: `OUT_VEZES` representa **recorrência autorreferida** no momento da notificação, não permitindo previsão de eventos futuros no tempo. Trata-se de uma medida retrospectiva, não prospectiva.

2. **Subnotificação**: dados do SINAN dependem de notificação, podendo haver viés de subnotificação, especialmente em casos de violência psicológica ou financeira.

3. **Viés institucional**: os dados refletem apenas casos que chegaram ao sistema de saúde, possivelmente sub-representando certos perfis ou contextos.

4. **Ausência de variáveis temporais longitudinais**: não há informações sobre histórico prévio de notificações da mesma vítima, limitando a análise temporal.

5. **Perda de dados**: a remoção de registros com valores ausentes reduziu o dataset de 294.393 para 57.022 registros (aproximadamente 80% de redução), podendo introduzir viés de seleção.

6. **Desempenho moderado**: acurácia de 70% indica que o modelo ainda comete erros significativos, limitando sua aplicação direta em contextos de alta criticidade.

### 7.2 Trabalhos futuros

Como trabalhos futuros, sugere-se:

1. **Modelos temporais**: uso de modelos de séries temporais ou sequenciais para análise de recorrência ao longo do tempo
2. **Análise de importância de variáveis**: exploração mais detalhada da importância das features em modelos ensemble (Random Forest, XGBoost)
3. **Integração com dados externos**: incorporação de dados socioeconômicos (IBGE), dados de segurança pública e indicadores de vulnerabilidade social
4. **Técnicas de imputação**: aplicação de métodos avançados de imputação de dados para reduzir a perda de registros
5. **Modelos mais complexos**: experimentação com redes neurais, gradient boosting (XGBoost, LightGBM) e modelos de ensemble
6. **Análise de subgrupos**: modelagem específica para diferentes perfis de vítimas ou contextos (urbano vs. rural, diferentes faixas etárias)
7. **Validação externa**: teste dos modelos em dados de outros anos ou outras fontes para avaliar generalização
8. **Sistema de alerta**: desenvolvimento de sistema de alerta precoce baseado nos modelos para identificação de casos de alto risco

---

## 8. Conclusão

Este estudo demonstrou que técnicas de aprendizado de máquina, aliadas a um pré-processamento cuidadoso e análise estatística rigorosa, permitem identificar padrões relevantes associados à recorrência da violência contra a mulher.

Embora os modelos não sejam determinísticos e apresentem limitações importantes, os resultados oferecem subsídios valiosos para:

- **Compreensão do fenômeno**: identificação de fatores de risco associados à recorrência
- **Formulação de políticas públicas**: evidências para direcionamento de recursos e intervenções
- **Apoio à tomada de decisão**: ferramentas auxiliares para profissionais de saúde e assistência social na identificação de casos de alto risco
- **Prevenção**: base para desenvolvimento de estratégias de prevenção primária e secundária

A escolha da Regressão Logística como modelo principal, apesar do desempenho ligeiramente inferior ao Random Forest, reflete a importância da **interpretabilidade** em contextos de saúde pública, onde a transparência e a capacidade de explicar decisões são fundamentais.

Os achados reforçam a importância de **intervenções precoces**, especialmente em casos envolvendo relações íntimas e violência não física, e destacam a necessidade de uma abordagem integrada e multissetorial para o enfrentamento da violência contra a mulher.

---

## Referências

- Ministério da Saúde. Sistema de Informação de Agravos de Notificação (SINAN). Disponível em: http://portalsinan.saude.gov.br/
- DATASUS. Transferência de Arquivos. Disponível em: https://datasus.saude.gov.br/transferencia-de-arquivos/
- Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1996). A simulation study of the number of events per variable in logistic regression analysis. *Journal of Clinical Epidemiology*, 49(12), 1373-1379.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

---

**Nota**: Este relatório foi elaborado com base na análise de dados públicos do SINAN para o ano de 2024. Todos os procedimentos metodológicos, códigos e resultados estão documentados nos notebooks Jupyter disponíveis no repositório do projeto.
