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


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_DIR = str(DOCKER_MODEL_DIR)

model = None


# ------------------------------------------------------------
# 1. Try Docker model
# ------------------------------------------------------------

if DOCKER_MODEL_DIR.exists():

    try:

        print(f"Attempting to load model from:")
        print(f"  {DOCKER_MODEL_DIR}")

        model = mlflow.pyfunc.load_model(
            str(DOCKER_MODEL_DIR)
        )

        print(
            f"✅ Model loaded successfully from "
            f"{DOCKER_MODEL_DIR}"
        )

    except Exception as e:

        print(
            f"⚠️ Failed to load Docker model: {e}"
        )


# ------------------------------------------------------------
# 2. Fallback to local MLflow artifacts
# ------------------------------------------------------------

if model is None:

    print("Searching local MLflow artifacts...")

    if not LOCAL_MLRUNS_DIR.exists():

        raise FileNotFoundError(
            f"MLflow directory not found:\n"
            f"{LOCAL_MLRUNS_DIR}"
        )

    # Find all MLmodel files
    local_model_paths = [
        model_path.parent
        for model_path in LOCAL_MLRUNS_DIR.rglob("MLmodel")
    ]

    print(
        f"Local MLflow models found: "
        f"{len(local_model_paths)}"
    )

    if not local_model_paths:

        raise FileNotFoundError(
            f"No MLflow model found under:\n"
            f"{LOCAL_MLRUNS_DIR}"
        )

    # Select most recently modified model
    latest_model = max(
        local_model_paths,
        key=lambda path: path.stat().st_mtime
    )

    print(
        f"Attempting to load latest local model:\n"
        f"{latest_model}"
    )

    try:

        model = mlflow.pyfunc.load_model(
            str(latest_model)
        )

        MODEL_DIR = str(latest_model)

        print(
            f"✅ Local MLflow model loaded successfully:\n"
            f"{latest_model}"
        )

    except Exception as e:

        raise Exception(
            f"Failed to load local MLflow model:\n"
            f"{latest_model}\n\n"
            f"Error: {e}"
        )


# ============================================================
# FEATURE SCHEMA
# ============================================================

try:

    feature_file = os.path.join(
        MODEL_DIR,
        "feature_columns.txt"
    )

    # --------------------------------------------------------
    # Option 1: feature_columns.txt
    # --------------------------------------------------------

    if os.path.isfile(feature_file):

        print(
            f"Loading feature columns from:\n"
            f"{feature_file}"
        )

        with open(
            feature_file,
            encoding="utf-8"
        ) as f:

            FEATURE_COLS = [
                line.strip()
                for line in f
                if line.strip()
            ]

    else:

        FEATURE_COLS = []

        # ----------------------------------------------------
        # Option 2: MLflow input schema
        # ----------------------------------------------------

        try:

            input_schema = (
                model.metadata.get_input_schema()
            )

            if (
                input_schema is not None
                and input_schema.inputs is not None
            ):

                FEATURE_COLS = [
                    column.name
                    for column in input_schema.inputs
                ]

        except Exception as schema_error:

            print(
                f"⚠️ Could not load MLflow input schema: "
                f"{schema_error}"
            )

        # ----------------------------------------------------
        # Option 3: sklearn feature_names_in_
        # ----------------------------------------------------

        if not FEATURE_COLS:

            sklearn_model = getattr(
                getattr(
                    model,
                    "_model_impl",
                    None
                ),
                "sklearn_model",
                None
            )

            FEATURE_COLS = list(
                getattr(
                    sklearn_model,
                    "feature_names_in_",
                    []
                )
            )

        # ----------------------------------------------------
        # Option 4: serving dataset
        # ----------------------------------------------------

        if not FEATURE_COLS:

            if SERVING_DATA_PATH.is_file():

                print(
                    f"Loading feature columns from:\n"
                    f"{SERVING_DATA_PATH}"
                )

                FEATURE_COLS = [
                    column
                    for column in pd.read_csv(
                        SERVING_DATA_PATH,
                        nrows=0
                    ).columns
                    if column not in {
                        "falha_24h",
                        "data_registro_pad"
                    }
                ]


    # --------------------------------------------------------
    # Validate feature columns
    # --------------------------------------------------------

    if not FEATURE_COLS:

        raise FileNotFoundError(
            "Could not determine model feature columns.\n"
            "Expected one of:\n"
            "- feature_columns.txt\n"
            "- MLflow input schema\n"
            "- sklearn feature_names_in_\n"
            "- industrial_data_serving.csv"
        )


    print(
        f"✅ Loaded {len(FEATURE_COLS)} feature columns"
    )

    print(
        f"Feature columns:\n{FEATURE_COLS}"
    )


except Exception as e:

    raise Exception(
        f"Failed to load feature columns: {e}"
    )


# ============================================================
# COLUMNS TO IGNORE
# ============================================================

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