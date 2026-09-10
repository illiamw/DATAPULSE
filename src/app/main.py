from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "industrial_data_raw.csv"
SERVING_DATA_PATH = PROJECT_ROOT / "data" / "serving" / "industrial_data_serving.csv"

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"RAW_DATA_PATH: {RAW_DATA_PATH}")


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from serving.inference import predict  # Core ML inference logic
from data.load_data import load_data  # Core ML inference logic


Script_ROOT = Path(__file__).resolve().parents[2]
if str(Script_ROOT) not in sys.path:
    sys.path.insert(0, str(Script_ROOT))

from scripts.run_data_transform import main as transf_pipeline  # Pipeline execution for data processing

# Initialize FastAPI application
app = FastAPI(
    title="DataPulse - Industrial Prediction API",
    description="ML API for predicting failures in the industrial sector",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    registro_id: str
    data_registro: str
    linha_producao: str
    turno: str
    maquina: str
    idade_maquina_anos: int
    temperatura_valor: float
    unidade_temperatura: str
    pressao_valor: float
    unidade_pressao: str
    vibracao_motor_mm_s: float
    velocidade_esteira_m_min: float
    umidade_pct: float
    tamanho_lote: int
    tempo_setup_min: float
    paradas_nao_planejadas: int
    taxa_defeitos_pct: float
    energia_sensor_b_kwh: float
    codigo_campanha: str
    ruido_aleatorio: float
    consumo_energia_kwh: float

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Carrega os dados históricos
        raw_data = load_data(
            RAW_DATA_PATH
        )
    
        # Adiciona o registro de teste
        raw_data = pd.concat(
            [
                raw_data,
                pd.DataFrame([data.dict()])  # Convert Pydantic model to DataFrame
            ],
            ignore_index=True
        )
    
        # Aplica o pipeline de transformação
        transformed_data = transf_pipeline(
            df=raw_data
        )
    
        print("###########################################################")
    
        # Mantém a última linha como DataFrame
        transformed_data = transformed_data.iloc[[-1]]
    
        print("Dados transformados:")
        print(transformed_data)  # Optional: For debugging in development
        result = predict(
                df=transformed_data
            )
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 