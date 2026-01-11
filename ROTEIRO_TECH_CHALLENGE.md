# ROTEIRO TECH CHALLENGE - Sistema de IA para Diagnóstico Médico

## CHECKLIST PRINCIPAL

### 1. Configuração do Ambiente e Projeto

### 2. Seleção e Download do Dataset

### 3. Exploração de Dados (EDA)

### 4. Pré-processamento de Dados

### 5. Análise de Correlação

### 6. Modelagem - Criação dos Modelos

### 7. Treinamento e Avaliação dos Modelos

### 8. Interpretação dos Resultados

### 9. Organização do Código e Documentação

### 10. Dockerfile e Configuração de Ambiente

### 11. Relatório Técnico

### 12. Vídeo de Demonstração

### 13. Preparação dos Entregáveis Finais

---

## DETALHAMENTO DOS ITENS

### 1. Configuração do Ambiente e Projeto

**Objetivo:** Estruturar o projeto inicial e configurar o ambiente de desenvolvimento.

**Ações:**

- Criar estrutura de pastas do projeto (data/, notebooks/, src/, models/, results/)
- Configurar ambiente virtual Python (venv ou conda)
- Criar arquivo `requirements.txt` com todas as dependências
- Inicializar repositório Git
- Criar `.gitignore` apropriado
- Criar estrutura básica do `README.md`

**Dependências sugeridas:**

- pandas, numpy, matplotlib, seaborn
- scikit-learn
- jupyter
- shap (para interpretabilidade)
- docker (para containerização)

---

### 2. Seleção e Download do Dataset

**Objetivo:** Escolher e obter o dataset médico para análise.

**Ações:**

- Escolher um dataset médico público (ex: Câncer de Mama, Diabetes)
- Fazer download do dataset
- Salvar na pasta `data/` do projeto
- Documentar a fonte e características do dataset escolhido
- Definir claramente o problema a ser resolvido (ex: "Classificar se paciente tem câncer de mama maligno ou benigno")

**Datasets sugeridos:**

- Câncer de Mama: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- Diabetes: https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data

---

### 3. Exploração de Dados (EDA)

**Objetivo:** Entender as características e padrões dos dados.

**Ações:**

- Carregar o dataset no notebook Jupyter
- Verificar dimensões (número de linhas e colunas)
- Verificar tipos de dados de cada coluna
- Identificar valores ausentes (missing values)
- Calcular estatísticas descritivas (média, mediana, desvio padrão, quartis)
- Visualizar distribuições das variáveis (histogramas, boxplots)
- Analisar distribuição da variável target (classes balanceadas ou não?)
- Criar visualizações relevantes (gráficos de barras, scatter plots, etc.)
- Documentar insights iniciais no notebook

---

### 4. Pré-processamento de Dados

**Objetivo:** Preparar os dados para modelagem, garantindo qualidade e formato adequado.

**Ações:**

- **Limpeza:**
  - Tratar valores ausentes (remover, imputar ou estratégia adequada)
  - Identificar e tratar outliers
  - Remover duplicatas (se houver)
- **Transformação:**
  - Converter variáveis categóricas em numéricas (Label Encoding ou One-Hot Encoding)
  - Normalizar/padronizar variáveis numéricas (se necessário)
  - Criar pipeline de pré-processamento em Python (usando sklearn Pipeline)
- **Separação:**
  - Separar features (X) e target (y)
  - Dividir dados em: treino (70%), validação (15%) e teste (15%)
  - Usar `train_test_split` do sklearn

---

### 5. Análise de Correlação

**Objetivo:** Identificar relações entre variáveis e features mais relevantes.

**Ações:**

- Calcular matriz de correlação entre variáveis numéricas
- Visualizar matriz de correlação (heatmap)
- Identificar features altamente correlacionadas
- Analisar correlação entre features e variável target
- Documentar features mais relevantes para o diagnóstico

---

### 6. Modelagem - Criação dos Modelos

**Objetivo:** Implementar pelo menos 2 algoritmos de classificação diferentes.

**Ações:**

- Escolher pelo menos 2 algoritmos (exemplos):
  - Regressão Logística
  - Árvore de Decisão
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - SVM
  - XGBoost
- Implementar cada modelo usando scikit-learn
- Configurar hiperparâmetros iniciais
- Garantir que todos os modelos usem os mesmos dados de treino/validação/teste

---

### 7. Treinamento e Avaliação dos Modelos

**Objetivo:** Treinar os modelos e avaliar seu desempenho com métricas adequadas.

**Ações:**

- **Treinamento:**
  - Treinar cada modelo com conjunto de treino
  - Ajustar hiperparâmetros usando conjunto de validação (opcional: GridSearchCV)
- **Avaliação:**
  - Fazer predições no conjunto de teste
  - Calcular métricas:
    - Accuracy (Acurácia)
    - Precision (Precisão)
    - Recall (Sensibilidade)
    - F1-Score
  - Gerar matriz de confusão para cada modelo
  - Comparar desempenho dos modelos
  - Discutir qual métrica é mais importante para o problema médico (ex: Recall pode ser mais importante para não perder casos positivos)

---

### 8. Interpretação dos Resultados

**Objetivo:** Entender como o modelo toma decisões e quais features são mais importantes.

**Ações:**

- **Feature Importance:**
  - Extrair importância das features (para modelos que suportam, como Random Forest)
  - Visualizar features mais importantes (gráfico de barras)
- **SHAP Values:**
  - Calcular SHAP values para interpretabilidade
  - Criar visualizações SHAP (summary plot, waterfall plot)
  - Explicar contribuição de cada feature para predições específicas
- **Análise Crítica:**
  - Discutir se o modelo pode ser usado na prática
  - Explicar limitações do modelo
  - Enfatizar que o médico sempre tem a palavra final
  - Sugerir como o modelo poderia ser integrado ao fluxo clínico

---

### 9. Organização do Código e Documentação

**Objetivo:** Estruturar o código de forma profissional e documentada.

**Ações:**

- Organizar código em notebooks Jupyter ou scripts Python
- Adicionar comentários explicativos no código
- Criar funções reutilizáveis quando apropriado
- Documentar cada etapa do processo
- Incluir markdown cells no notebook explicando o que está sendo feito
- Salvar modelos treinados (usando pickle ou joblib)
- Salvar gráficos e visualizações na pasta `results/`

---

### 10. Dockerfile e Configuração de Ambiente

**Objetivo:** Criar ambiente containerizado para facilitar execução e reprodução.

**Ações:**

- Criar `Dockerfile` com:
  - Imagem base Python
  - Instalação de dependências do `requirements.txt`
  - Configuração do ambiente Jupyter (se aplicável)
  - Comandos para executar o projeto
- Atualizar `README.md` com:
  - Descrição do projeto
  - Instruções de instalação (com e sem Docker)
  - Como executar o código
  - Estrutura do projeto
  - Link para download do dataset
  - Informações sobre os modelos e resultados

---

### 11. Relatório Técnico

**Objetivo:** Documentar todo o processo, decisões e resultados de forma técnica.

**Ações:**

- Criar documento PDF com:
  - **Introdução:** Contexto do problema e objetivo
  - **Dataset:** Descrição do dataset escolhido e problema a resolver
  - **Metodologia:**
    - Estratégias de pré-processamento aplicadas
    - Modelos escolhidos e justificativa
    - Métricas de avaliação e por que foram escolhidas
  - **Resultados:**
    - Tabelas com métricas de cada modelo
    - Gráficos e visualizações (matriz de confusão, feature importance, SHAP)
    - Comparação entre modelos
  - **Interpretação:**
    - Análise dos resultados
    - Features mais importantes
    - Limitações e considerações práticas
  - **Conclusão:**
    - Resumo dos achados
    - Próximos passos sugeridos
  - **Link do repositório Git**

---

### 12. Vídeo de Demonstração

**Objetivo:** Apresentar o sistema em funcionamento de forma clara e objetiva.

**Ações:**

- Gravar vídeo de até 15 minutos
- **Conteúdo do vídeo:**
  - Apresentação breve do problema
  - Demonstração do código em execução
  - Explicação do fluxo do projeto
  - Mostrar resultados principais (gráficos, métricas)
  - Interpretação dos resultados
  - Conclusão
- Fazer upload no YouTube ou Vimeo
- Configurar como "público" ou "não listado"
- Incluir link do vídeo no README e relatório

---

### 13. Preparação dos Entregáveis Finais

**Objetivo:** Consolidar todos os materiais para entrega.

**Ações:**

- **Verificar repositório Git:**
  - Código-fonte completo commitado
  - README.md atualizado
  - Dockerfile presente
  - Dataset ou link para download
  - Resultados (gráficos, análises) salvos
- **Verificar relatório PDF:**
  - Todas as seções preenchidas
  - Link do repositório incluído
  - Gráficos e tabelas de qualidade
  - Formatação profissional
- **Verificar vídeo:**
  - Vídeo publicado e acessível
  - Link funcionando
  - Duração dentro do limite (15 min)
- **Checklist final:**
  - [ ] Repositório Git completo e organizado
  - [ ] README.md com instruções claras
  - [ ] Dockerfile funcional
  - [ ] Relatório PDF completo
  - [ ] Vídeo publicado e link disponível
  - [ ] Todos os códigos documentados
  - [ ] Resultados salvos e organizados

---

## TAREFA EXTRA (OPCIONAL - Pode aumentar a nota)

### 14. Classificação com Dados de Imagem (CNN)

**Objetivo:** Implementar diagnóstico usando redes neurais convolucionais para imagens médicas.

**Ações:**

- Escolher dataset de imagens médicas (ex: Pneumonia em Radiografias)
- Carregar e explorar imagens
- Pré-processar imagens (redimensionar, normalizar)
- Criar modelo CNN usando TensorFlow/Keras
- Treinar e avaliar modelo de imagem
- Comparar resultados com modelo de dados estruturados
- Documentar no relatório e vídeo

**Datasets sugeridos:**

- Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Câncer de Mama (imagens): https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data

---

## DICAS IMPORTANTES

- **Comece cedo:** Este trabalho requer várias etapas, não deixe para a última hora
- **Versionamento:** Use Git desde o início, faça commits frequentes
- **Documentação:** Documente enquanto desenvolve, não deixe para o final
- **Testes:** Teste cada etapa antes de avançar para a próxima
- **Backup:** Mantenha backups do trabalho
- **Colaboração:** Se for trabalho em grupo, use branches do Git e organize bem as tarefas
