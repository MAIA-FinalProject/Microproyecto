from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import src.models.train as train_module
from src.features.build_features import FEATURES_OUTPUT_PATH
from src.models.train import (
    CONTINUOUS_VARS,
    IMPUTE_VARS,
    build_models,
    build_preprocessor,
    cross_validate_scores,
    evaluate_holdout,
    load_dataset,
    train_and_log_model,
)


@pytest.fixture()
def synthetic_features() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético del tamaño y forma de data/features/features.csv
    (post build_features(): todo numérico, un puñado de continuas, una
    columna con nulos genuinos en congenital.anomaly). No depende del
    dataset real, por lo que corre igual en CI que en local. 40 filas,
    balanceadas, para que StratifiedKFold(5) tenga suficientes muestras
    por clase en cada fold."""
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(
        {
            "PregnancyAge": rng.normal(32, 2, n),
            "BirthWeight": rng.normal(1600, 400, n),
            "apgar1": rng.integers(1, 10, n).astype(float),
            "apgar5": rng.integers(3, 11, n).astype(float),
            "duration.hopitalization": rng.exponential(15, n),
            "duration.O2": rng.exponential(6, n),
            "RoundHeadAtBirth": rng.normal(29, 3, n),
            "Sex": rng.integers(0, 2, n),
            "DM": rng.integers(0, 2, n),
            "congenital.anomaly": [np.nan if i % 10 == 0 else 0.0 for i in range(n)],
        }
    )
    # Alternado (no en bloques) para que cualquier corte contiguo (p.ej.
    # X.iloc[30:] en los tests de abajo) tenga ambas clases representadas.
    target = pd.Series([i % 2 for i in range(n)], name="neurodev_alteration")
    return df, target


def test_build_models_returns_three_algorithms() -> None:
    models = build_models()
    assert set(models.keys()) == {"logistic_regression", "random_forest", "xgboost"}


def test_preprocessor_only_imputes_declared_vars(
    synthetic_features: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = synthetic_features
    preprocessor = build_preprocessor(X)
    impute_cols = next(
        cols for (name, _, cols) in preprocessor.transformers if name == "impute"
    )
    assert impute_cols == IMPUTE_VARS


def test_no_leakage_scaler_fit_only_on_train(
    synthetic_features: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Ajusta el preprocesador solo con X_train y confirma que la mediana
    aprendida por RobustScaler coincide con la mediana calculada a mano
    sobre X_train (no sobre X_train+X_test). Si el código accidentalmente
    ajustara el scaler con el dataset completo, esta aserción fallaría."""
    X, _ = synthetic_features
    X_train = X.iloc[:30]

    preprocessor = build_preprocessor(X)
    preprocessor.fit(X_train)

    scaler = dict(preprocessor.named_transformers_)["scale"]
    expected_center = X_train[CONTINUOUS_VARS].median().to_numpy()
    np.testing.assert_allclose(scaler.center_, expected_center)


def test_cross_validate_scores_in_valid_range(
    synthetic_features: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_features
    preprocessor = build_preprocessor(X)
    model = build_models()["logistic_regression"]
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    scores = cross_validate_scores(pipeline, X, y)
    for value in scores.values():
        assert 0.0 <= value <= 1.0


def test_evaluate_holdout_metrics_in_valid_range(
    synthetic_features: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = synthetic_features
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train, y_test = y.iloc[:30], y.iloc[30:]

    preprocessor = build_preprocessor(X)
    model = build_models()["random_forest"]
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_holdout(pipeline, X_test, y_test)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_train_and_log_model_saves_reloadable_joblib(
    synthetic_features: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """Corre el flujo completo para un modelo contra un tracking store de
    MLflow temporal (no toca ./mlruns real ni requiere un server local), y
    confirma que el .joblib guardado se puede recargar y que sus
    predicciones reproducen el mismo F1 que quedó logueado."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-experiment")

    X, y = synthetic_features
    X_train, X_test = X.iloc[:30], X.iloc[30:]
    y_train, y_test = y.iloc[:30], y.iloc[30:]

    preprocessor = build_preprocessor(X)
    model = build_models()["logistic_regression"]

    original_models_dir = train_module.MODELS_DIR
    train_module.MODELS_DIR = tmp_path / "models"
    try:
        metrics = train_and_log_model(
            "logistic_regression",
            model,
            preprocessor,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    finally:
        train_module.MODELS_DIR = original_models_dir

    assert set(metrics.keys()) == {"test_f1", "test_roc_auc", "test_recall"}

    artifact_path = tmp_path / "models" / "logistic_regression.joblib"
    assert artifact_path.exists()

    reloaded = joblib.load(artifact_path)
    reloaded_pred = reloaded.predict(X_test)
    assert set(reloaded_pred).issubset({0, 1})
    assert f1_score(y_test, reloaded_pred) == pytest.approx(metrics["test_f1"])


@pytest.mark.skipif(
    not Path(FEATURES_OUTPUT_PATH).exists(),
    reason="Requiere data/features/features.csv (DVC), no disponible en CI",
)
class TestRealDataIntegration:
    """Validación end-to-end con el dataset real, se salta automáticamente
    si no está disponible localmente (igual que en test_features.py)."""

    def test_load_dataset_shape(self) -> None:
        X, y = load_dataset()
        assert X.shape[0] == 89
        assert y.shape[0] == 89
