import mlflow
import pandas as pd
import sys
from pathlib import Path
import mlflow.xgboost
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_mlops.data.load_data import load_data


mlflow.set_experiment("PrevisaoFalha24h")

mlflow.set_tracking_uri("http://localhost:5000")

def data_split(
        df: pd.DataFrame | None = None,
        target_col: str = "falha_24h"):
    """
    Splits the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.

    Returns:
        tuple: Split datasets (X_train, X_test, y_train, y_test).
    """
    # Exemplo de uso das funções
    if df is None:
        file_path = "data/serving/industrial_data_serving.csv"
        df = load_data(file_path)

    X = df.drop(columns=[target_col, "data_registro_pad"])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):
    
    

    with mlflow.start_run(run_name=model_name):
        print(f"\n{'=' * 40}")
        print(f"iNINCIANDO Modelo: {model_name}")
        print(f"{'=' * 40}")

        # ==========================
        # Tags
        # ==========================

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("problem_type", "classification")
        mlflow.set_tag("target", "falha_24h")

        # ==========================
        # Parâmetros
        # ==========================

        mlflow.log_params(model.get_params())

        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_features", X_train.shape[1])
        mlflow.log_text(
            "\n".join(str(column) for column in X_train.columns),
            "feature_columns.txt"
        )

        # ==========================
        # Training
        # ==========================

        model.fit(X_train, y_train)

        # ==========================
        # Prediction
        # ==========================

        y_pred = model.predict(X_test)

        y_proba = None

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]

        # ==========================
        # Metrics
        # ==========================

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            "f1": f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        }

        if y_proba is not None:
            metrics["auc"] = roc_auc_score(
                y_test,
                y_proba
            )

        mlflow.log_metrics(metrics)

        # ==========================
        # Model
        # ==========================

        if isinstance(model, XGBClassifier):

            mlflow.xgboost.log_model(
                model,
                name="model"
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model"
            )

        # ==========================
        # Output
        # ==========================

        print(f"\n{'=' * 40}")
        print(f"Modelo: {model_name}")
        print(f"{'=' * 40}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

def run_experiment(model : dict= None):

    file_path = "data/serving/industrial_data_serving.csv"
    
    df = load_data(file_path)

    print(f"Schema of the loaded DataFrame:\n{df.dtypes}\n")

    X_train, X_test, y_train, y_test = data_split(df)

    if model is None:
        model = {
                
                "KNN": KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance"
                )
                }

    for model_name, model in models.items():
    
            train_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

def main():
    file_path = "data/serving/industrial_data_serving.csv"
    
    df = load_data(file_path)

    print(f"Schema of the loaded DataFrame:\n{df.dtypes}\n")

    X_train, X_test, y_train, y_test = data_split(df)

    models = {
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ),
        "SVM": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=10000,
            C=1.0
        ),

        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        )
    }

    for model_name, model in models.items():

        train_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

if __name__ == "__main__":

    main()