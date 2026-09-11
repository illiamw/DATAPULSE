"""
INFERENCE PIPELINE - Production ML Model Serving
================================================

DataPulse - Industrial Failure Prediction

This module provides the core inference functionality for the
industrial failure prediction model.

It ensures consistency between training-time and serving-time
features and loads the model from local MLflow artifacts during
development or from /app/model inside Docker.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import mlflow


IGNORE_COLS = [
    "falha_24h",
    "data_registro_pad"
]


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model_path=None,
    df=None,
    input_dict: dict = None
) -> str:

    """
    Performs industrial failure prediction.

    Args:
        df:
            DataFrame containing transformed features.

        input_dict:
            Dictionary containing raw input data.

    Returns:
        "Falha" or "Sem Falha"
    """

    # --------------------------------------------------------
    # 1. Convert dictionary to DataFrame
    # --------------------------------------------------------

    if df is None:

        if input_dict is None:

            raise ValueError(
                "Either 'df' or 'input_dict' must be provided."
            )

        df = pd.DataFrame(
            [input_dict]
        )

    # --------------------------------------------------------
    # 2. Remove columns that should not enter the model
    # --------------------------------------------------------

    df = df.drop(
        columns=IGNORE_COLS,
        errors="ignore"
    )

    # --------------------------------------------------------
    # 3. Guarantee exact training feature order
    # --------------------------------------------------------

    FEATURE_COLS = df.columns.tolist()
    df = df.reindex(
        columns=FEATURE_COLS,
        fill_value=0
    )

    print("=" * 70)
    print("Input DataFrame for prediction:")
    print(df)

    print(
        f"Expected feature columns:\n"
        f"{FEATURE_COLS}"
    )

    print(
        f"Schema:\n"
        f"{df.dtypes}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 4. Model prediction
    # --------------------------------------------------------

    try:
        model = load_model(model_path)
        preds = model.predict(df)

        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if (
            isinstance(preds, (list, tuple))
            and len(preds) == 1
        ):

            result = preds[0]

        else:

            result = preds

    except Exception as e:

        raise Exception(
            f"Model prediction failed: {e}"
        )

    # --------------------------------------------------------
    # 5. Business output
    # --------------------------------------------------------

    if result == 1:

        return "Falha"

    else:

        return "Sem Falha"


import mlflow


def load_model(model_id: str):

    tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    model_uri = f"models:/{model_id}"

    print(f"Model URI: {model_uri}")

    model_info = mlflow.get_logged_model(model_id)

    print("MODEL ID:", model_info.model_id)
    print("MODEL NAME:", model_info.name)
    print("MODEL URI:", model_info.model_uri)
    print("ARTIFACT LOCATION:", model_info.artifact_location)
    print("RUN ID:", model_info.source_run_id)
    print("STATUS:", model_info.status)

    model = mlflow.pyfunc.load_model(model_uri)

    return model

    