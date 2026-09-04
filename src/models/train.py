"""
Entrenamiento baseline y logging de métricas en MLflow (ticket #10).

Entrena 3 algoritmos (Logistic Regression, Random Forest, XGBoost) sobre
data/features/features.csv para predecir `neurodev_alteration`, y loguea
hiperparámetros, métricas y el pipeline entrenado (.joblib) en MLflow.

Nota sobre imputación y escalado (ver nota en src/features/build_features.py):
se ajustan aquí, DESPUÉS del split train/test, dentro de un sklearn.Pipeline
por modelo. El Pipeline solo se ajusta (`.fit`) sobre X_train, nunca sobre
X_test, para evitar data leakage.

Decisiones de preprocesamiento:
- Imputación: solo `congenital.anomaly` tiene nulos genuinos (ver
  data/README.md y tests/test_features.py::test_no_unexpected_nulls). Es una
  variable binaria 0/1, por lo que se imputa con la moda (most_frequent), no
  con media/mediana.
- Escalado: RobustScaler (mediana/IQR) sobre las variables continuas. Se
  eligió sobre StandardScaler/MinMaxScaler porque, revisando la distribución
  real, `duration.O2` tiene skew fuerte (+2.34) y ~12% de outliers por IQR
  (11/89) — RobustScaler es el menos sensible a esos extremos en un dataset
  de solo 89 filas, donde unos pocos outliers distorsionan bastante
  media/std.

Nota sobre validación: con n=89 (~71 train / 18 test tras el split), un
único holdout deja muy pocas filas para medir métricas de forma confiable.
Por eso se reporta también StratifiedKFold(5) sobre train, además del
holdout final.
"""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.config import settings
from src.data.loader import TARGET
from src.features.build_features import FEATURES_OUTPUT_PATH

RANDOM_STATE = 42

CONTINUOUS_VARS = [
    "PregnancyAge",
    "BirthWeight",
    "apgar1",
    "apgar5",
    "duration.hopitalization",
    "duration.O2",
    "RoundHeadAtBirth",
]
# Única columna con nulos genuinos tras build_features() (ver docstring).
IMPUTE_VARS = ["congenital.anomaly"]

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_dataset(path: Path = FEATURES_OUTPUT_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Carga data/features/features.csv y separa predictores del target."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Imputer (moda) + RobustScaler, ajustados solo sobre train dentro de
    cada Pipeline. El resto de columnas (ya binarias/dummies desde
    build_features.py) pasan sin transformar."""
    impute_cols = [c for c in IMPUTE_VARS if c in X.columns]
    scale_cols = [c for c in CONTINUOUS_VARS if c in X.columns]
    return ColumnTransformer(
        transformers=[
            ("impute", SimpleImputer(strategy="most_frequent"), impute_cols),
            ("scale", RobustScaler(), scale_cols),
        ],
        remainder="passthrough",
    )


def build_models() -> dict[str, ClassifierMixin]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
    }


def cross_validate_scores(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    """StratifiedKFold(5) sobre train. cross_validate re-ajusta el Pipeline
    completo (imputer+scaler+modelo) en cada fold usando solo esa porción
    de train, así que tampoco hay leakage aquí."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"f1": "f1", "roc_auc": "roc_auc", "recall": "recall"},
    )
    return {
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_recall_mean": float(scores["test_recall"].mean()),
    }


def evaluate_holdout(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_recall": float(recall_score(y_test, y_pred)),
    }


def train_and_log_model(
    name: str,
    model: ClassifierMixin,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Entrena un modelo dentro de un Pipeline, loguea hiperparámetros,
    métricas de CV y holdout, y el pipeline completo como artefacto
    .joblib en un run de MLflow. Retorna las métricas de holdout."""
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    with mlflow.start_run(run_name=name):
        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})

        cv_metrics = cross_validate_scores(pipeline, X_train, y_train)
        mlflow.log_metrics(cv_metrics)

        pipeline.fit(X_train, y_train)
        holdout_metrics = evaluate_holdout(pipeline, X_test, y_test)
        mlflow.log_metrics(holdout_metrics)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, artifact_path)
        mlflow.log_artifact(str(artifact_path))

    return holdout_metrics


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)
    results: dict[str, dict[str, float]] = {}
    for name, model in build_models().items():
        results[name] = train_and_log_model(
            name, model, preprocessor, X_train, y_train, X_test, y_test
        )

    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
