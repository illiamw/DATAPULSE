# DATAPULSE

DATAPULSE é um projeto de MLOps para prever falhas operacionais em um ambiente industrial usando um conjunto de dados sintético, processamento automatizado de dados, treinamento de modelos com MLflow e uma camada de serviço em FastAPI.

O projeto foi criado para simular um pipeline completo de ciência de dados e produção: geração de dados, validação, pré-processamento, engenharia de features, comparação de modelos, rastreamento de experimentos e predição em tempo real. Ele segue uma estrutura prática de MLOps, com o código organizado por dados, features, treinamento e serving.

## Visão geral

Este projeto constrói e expõe um modelo de classificação para prever se um ativo industrial falhará dentro das próximas 24 horas (`falha_24h`). A API recebe dados de sensores operacionais e retorna um resultado binário em linguagem natural:

- `Falha`
- `Sem Falha`

O pipeline de dados inclui inconsistências realistas que exigem limpeza, como:

- formatos de data misturados
- aliases categóricos e valores inválidos
- incompatibilidade de unidades (`C`/`F`, `bar`/`psi`)
- valores ausentes
- valores físicos impossíveis
- outliers
- variáveis redundantes ou irrelevantes

## Principais funcionalidades

- Gerador de dataset industrial sintético
- Pipeline de validação e limpeza de qualidade de dados
- Padronização de unidades e timestamps
- Engenharia de features e transformações
- Treinamento e comparação de modelos com múltiplos algoritmos
- Rastreamento de experimentos com MLflow e integração ao registry de modelos
- Serviço de inferência com FastAPI
- Suporte a Docker e Docker Compose para implantação local
- Endpoints para listagem de modelos e execução do pipeline

## Arquitetura do projeto

O repositório segue esta estrutura:

```text
DATAPULSE/
├── data/
│   ├── raw/
│   ├── silver/
│   └── serving/
├── scripts/
│   ├── run_data_transform.py
│   └── run_pipeline.py
├── src/
│   └── app_mlops/
│       ├── data/
│       │   ├── gerate_data.py
│       │   ├── load_data.py
│       │   ├── preprocess_data.py
│       │   └── __pycache__/
│       ├── features/
│       │   └── build_features.py
│       ├── models/
│       │   └── train.py
│       ├── serving/
│       │   └── inference.py
│       └── main.py
├── tests/
│   └── test_server_predict.json
├── Dockerfile
├── docker-compose.yml
├── init_server.sh
├── requirements.txt
├── README.md
├── mlflow.db
├── mlruns/
└── mlartifacts/
```

## Fluxo de dados

O pipeline de dados do projeto funciona da seguinte forma:

1. Geração de dados a partir de registros industriais sintéticos
2. Carregamento dos dados brutos em CSV
3. Pré-processamento e normalização
4. Engenharia de features e transformação
5. Treinamento do modelo
6. Rastreamento de métricas e artefatos via MLflow
7. Exposição de predições por meio do FastAPI

### Geração de dados

O dataset sintético é gerado em `src/app_mlops/data/gerate_data.py` e contém variáveis como:

- `registro_id`
- `data_registro`
- `linha_producao`
- `turno`
- `maquina`
- `idade_maquina_anos`
- `temperatura_valor`
- `unidade_temperatura`
- `pressao_valor`
- `unidade_pressao`
- `vibracao_motor_mm_s`
- `velocidade_esteira_m_min`
- `umidade_pct`
- `tamanho_lote`
- `tempo_setup_min`
- `paradas_nao_planejadas`
- `taxa_defeitos_pct`
- `energia_sensor_b_kwh`
- `codigo_campanha`
- `ruido_aleatorio`
- `consumo_energia_kwh`
- `falha_24h`

Os dados gerados introduzem intencionalmente problemas reais de qualidade de dados industriais para exercitar a lógica de limpeza e pré-processamento.

### Pré-processamento

`src/app_mlops/data/preprocess_data.py` é responsável por:

- padronização de formatos de data
- conversão de unidades de temperatura e pressão
- normalização de campos categóricos
- substituição de aliases textuais inválidos
- tratamento de valores ausentes
- remoção de linhas duplicadas e inconsistentes
- limpeza de valores impossíveis
- clipping de outliers por limites IQR
- remoção de colunas redundantes
- codificação one-hot para variáveis categóricas
- escalonamento de valores numéricos

### Engenharia de features

`src/app_mlops/features/build_features.py` realiza:

- tratamento de outliers
- limpeza de multicolinearidade
- transformação simbólica para numérica
- escalonamento numérico com transformações log e padronização

O conjunto final de features é armazenado no diretório de dados de serving para treinamento e previsão.

## Treinamento de modelos

A camada de modelagem é implementada em `src/app_mlops/models/train.py`.

### Tipos de modelos suportados

O registry de treinamento inclui:

- `GaussianNB`
- `KNeighborsClassifier`
- `SVC`
- `GradientBoostingClassifier`
- `RandomForestClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `XGBClassifier`

Lógica de treinamento:

- divide o dataset em treino e teste com `train_test_split`
- configura um experimento no MLflow: `PrevisaoFalha24h`
- registra parâmetros, tags, nomes das features e métricas
- calcula métricas de classificação:
  - accuracy
  - precision
  - recall
  - f1-score
  - AUC quando disponível
- registra o artefato do modelo no MLflow
- disponibiliza o modelo para inferência em produção

O alvo padrão de treinamento é `falha_24h`.

## Serviço de inferência

A API é implementada em `src/app_mlops/main.py` e usa FastAPI.

### Verificação de saúde

```http
GET /
```

Retorna:

```json
{"status": "ok_v05"}
```

### Endpoint de predição

```http
POST /predict
```

Este endpoint:

1. carrega os dados brutos
2. acrescenta o novo registro ao conjunto de dados
3. executa o pipeline de transformação
4. mantém apenas a observação mais recente
5. chama o modelo de previsão
6. retorna o resultado binário

Exemplo de corpo da requisição:

```json
{
  "model": {
    "model_run": "model_name"
  },
  "data": {
    "registro_id": "REG-00000",
    "data_registro": "2025-01-01 00:00:00",
    "linha_producao": "L-C",
    "turno": "Tarde",
    "maquina": "M05",
    "idade_maquina_anos": 10,
    "temperatura_valor": 764.04,
    "unidade_temperatura": "C",
    "pressao_valor": 5.925,
    "unidade_pressao": "bar",
    "vibracao_motor_mm_s": 2.826,
    "velocidade_esteira_m_min": 42.87,
    "umidade_pct": 60.76,
    "tamanho_lote": 1958,
    "tempo_setup_min": 34.68,
    "paradas_nao_planejadas": 3,
    "taxa_defeitos_pct": 2.044,
    "energia_sensor_b_kwh": 550.54,
    "codigo_campanha": "C99",
    "ruido_aleatorio": -1.0691204379907002,
    "consumo_energia_kwh": 540.04,
    "falha_24h": 0
  }
}
```

Exemplo de resposta:

```json
{
  "prediction": "Sem Falha"
}
```

### Endpoint de modelos ativos

```http
GET /models
```

Retorna os modelos ativos rastreados no MLflow com:

- `model_id`
- `model_name`
- `run_id`
- `run_name`
- `status`
- `accuracy`
- `precision`
- `f1`

### Endpoints de pipeline

```http
POST /pipeline
```

Executa o pipeline de ETL e preparação de dados.

```http
POST /experiment
```

Executa uma configuração específica de experimento.

```http
POST /allexperiments
```

Executa a suíte completa de experimentos padrão.

## Integração com MLflow

O projeto usa MLflow para rastrear experimentos e artefatos de modelos. A configuração principal está em `docker-compose.yml` e nos módulos Python.

### URL padrão de rastreamento

- Desenvolvimento local: `http://localhost:5000`
- Ambiente Docker: `http://mlflow:5000`

Variável de ambiente:

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
```

O projeto armazena resultados dos experimentos e artefatos dos modelos nos diretórios `mlruns/` e `mlartifacts/`.

## Scripts e automação

### `scripts/run_pipeline.py`

Este script gera os dados brutos, carrega-os, faz o pré-processamento, salva o dataset silver e grava o dataset de serving.

### `scripts/run_data_transform.py`

Este script é usado para transformar o pipeline de dados e prepará-lo para modelagem.

### `init_server.sh`

Instala o ambiente base necessário para o projeto, incluindo:

- dependências do sistema
- Docker
- Docker Compose

## Requisitos

As dependências do projeto estão listadas em `requirements.txt`:

- numpy
- pandas
- seaborn
- scikit-learn
- mlflow
- xgboost
- fastapi[standard]
- pydantic

## Configuração local

### Pré-requisitos

- Python 3.11+
- pip
- Docker e Docker Compose (recomendado)

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Executar o pipeline de dados manualmente

```bash
python scripts/run_pipeline.py
```

### Iniciar a API localmente

```bash
uvicorn src.app_mlops.main:app --host 0.0.0.0 --port 8000 --reload
```

Em seguida, acesse:

- `http://localhost:8000/`
- `http://localhost:8000/docs`

## Implantação com Docker

O projeto inclui uma configuração do Docker Compose que inicia:

- banco PostgreSQL
- servidor de rastreamento MLflow
- aplicação FastAPI

### Iniciar serviços

```bash
docker compose up --build
```

### Parar serviços

```bash
docker compose down
```

### Reconstruir sem cache

```bash
docker compose up --build --force-recreate
```

## Observações importantes

- O projeto usa um dataset sintético e não um banco industrial de produção.
- A API de inferência espera um nome de modelo do MLflow e usa esse modelo para gerar uma previsão a partir de dados transformados.
- Para uma implantação real em produção, recomenda-se substituir as configurações de CORS wildcard e configurar um ambiente seguro.
- A aplicação foi desenhada para uso em um ambiente local de MLOps e pode ser expandida para fluxos de CI/CD, monitoramento e re-treinamento de modelos.

## Exemplo de fluxo de desenvolvimento

```bash
# 1. instalar dependências
pip install -r requirements.txt

# 2. gerar e preprocessar dados
python scripts/run_pipeline.py

# 3. treinar modelos
python -c "from src.app_mlops.models.train import run_experiment; run_experiment()"

# 4. iniciar API
uvicorn src.app_mlops.main:app --host 0.0.0.0 --port 8000 --reload
```

## Licença

Este projeto foi desenvolvido para uso educacional e experimental em MLOps. Ajuste os termos de licença conforme a sua organização ou ambiente de implantação.
