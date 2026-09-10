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


# ============================================================
# PROJECT PATHS
# ============================================================

# inference.py:
#
# MLOPS_DATAPULSE/
# └── src/
#     └── app_mlops/
#         └── serving/
#             └── inference.py
#
# parents[0] -> serving
# parents[1] -> app_mlops
# parents[2] -> src
# parents[3] -> MLOPS_DATAPULSE

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_ROOT = PROJECT_ROOT / "src"

DATA_ROOT = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_ROOT / "raw"

SERVING_DATA_DIR = DATA_ROOT / "serving"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# ============================================================
# DATA PATHS
# ============================================================

RAW_DATA_PATH = (
    RAW_DATA_DIR
    / "industrial_data_raw.csv"
)

SERVING_DATA_PATH = (
    SERVING_DATA_DIR
    / "industrial_data_serving.csv"
)


# ============================================================
# MODEL PATHS
# ============================================================

# Docker / production
DOCKER_MODEL_DIR = Path("/app/model")


# Local development
LOCAL_MLRUNS_DIR = MLRUNS_DIR


# ============================================================
# DEBUG
# ============================================================

print("=" * 70)
print("DataPulse - Inference Configuration")
print("=" * 70)

print(f"PROJECT_ROOT       : {PROJECT_ROOT}")
print(f"SRC_ROOT           : {SRC_ROOT}")
print(f"DATA_ROOT          : {DATA_ROOT}")
print(f"RAW_DATA_PATH      : {RAW_DATA_PATH}")
print(f"SERVING_DATA_PATH  : {SERVING_DATA_PATH}")
print(f"MLRUNS_DIR         : {MLRUNS_DIR}")
print(f"DOCKER_MODEL_DIR   : {DOCKER_MODEL_DIR}")

print("=" * 70)


# ============================================================
# PYTHON PATH
# ============================================================

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IGNORE_COLS = [
    "falha_24h",
    "data_registro_pad"
]


# ============================================================
# PREDICTION
# ============================================================

def predict(
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