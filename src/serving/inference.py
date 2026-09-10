"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "/app/model"

try:
    # Load the trained XGBoost model in MLflow pyfunc format
    # This ensures compatibility regardless of the underlying ML library
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    # Fallback for local development (OPTIONAL)
    try:
        # Try loading the latest local MLflow model, regardless of the cwd.
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        mlruns_dir = project_root / "mlruns"
        local_model_paths = [
            model_path.parent
            for model_path in mlruns_dir.rglob("MLmodel")
        ] if mlruns_dir.is_dir() else []

        print(f"Local model paths found: {local_model_paths}")
        if not local_model_paths:
            raise FileNotFoundError(
                f"No MLflow model found under {mlruns_dir}"
            )

        latest_model = max(
            local_model_paths,
            key=lambda path: path.stat().st_mtime
        )
        print(f"Attempting to load latest local model from {latest_model}")
        model = mlflow.pyfunc.load_model(str(latest_model))
        MODEL_DIR = str(latest_model)
        print(f"✅ Fallback: Loaded model from {latest_model}")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    if os.path.isfile(feature_file):
        with open(feature_file, encoding="utf-8") as f:
            FEATURE_COLS = [line.strip() for line in f if line.strip()]
    else:
        input_schema = model.metadata.get_input_schema()
        FEATURE_COLS = (
            [column.name for column in input_schema.inputs]
            if input_schema is not None and input_schema.inputs is not None
            else []
        )

        if not FEATURE_COLS:
            sklearn_model = getattr(
                getattr(model, "_model_impl", None),
                "sklearn_model",
                None
            )
            FEATURE_COLS = list(
                getattr(sklearn_model, "feature_names_in_", [])
            )

        if not FEATURE_COLS:
            project_root = Path(__file__).resolve().parents[2]
            serving_file = project_root / "data" / "serving" / "industrial_data_serving.csv"
            if serving_file.is_file():
                FEATURE_COLS = [
                    column
                    for column in pd.read_csv(serving_file, nrows=0).columns
                    if column not in {"falha_24h", "data_registro_pad"}
                ]

    if not FEATURE_COLS:
        raise FileNotFoundError(
            "feature_columns.txt was not found and the MLflow model has "
            "no input schema or feature_names_in_. Retrain the model with "
            "the current training code."
        )

    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
    print(f"Feature columns: {FEATURE_COLS}")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance


IGNORE_COLS = ["falha_24h", "data_registro_pad"]  # Columns to ignore during prediction


def predict(df = None, input_dict: dict = None) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
    """
    if df is None:
        # === STEP 1: Convert Input to DataFrame ===
        # Create single-row DataFrame for pandas transformations
        df = pd.DataFrame([input_dict])

    df = df.drop(columns=IGNORE_COLS, errors="ignore")
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    print(f"Input DataFrame for prediction:\n{df}")
    print(f"Expected feature columns: {FEATURE_COLS}")
    print(f"Schema: {df.dtypes}")
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = model.predict(df)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Falha"      # High risk - needs intervention
    else:
        return "Sem Falha"  # Low risk - maintain normal service