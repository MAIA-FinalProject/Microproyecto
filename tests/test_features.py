import pandas as pd
import pytest

from src.data.loader import TARGET
from src.features.build_features import build_features


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    """Construye la matriz de features una sola vez y la reutiliza en
    todos los tests del módulo (el dataset es pequeño, pero evita
    recalcular innecesariamente)."""
    return build_features()


def test_output_shape(features_df: pd.DataFrame) -> None:
    """El dataset tiene 89 registros; el número de columnas puede variar
    si se agregan/quitan variables, pero las filas nunca deberían."""
    assert features_df.shape[0] == 89


def test_no_nulls_in_output(features_df: pd.DataFrame) -> None:
    """La matriz final debe estar completamente imputada, sin nulos."""
    assert features_df.isna().sum().sum() == 0


def test_all_columns_numeric(features_df: pd.DataFrame) -> None:
    """Ninguna columna debe quedar como texto tras el encoding (esto es
    justamente lo que falló silenciosamente con csf.culture antes del
    fix de case-sensitivity)."""
    non_numeric = [
        col
        for col in features_df.columns
        if not pd.api.types.is_numeric_dtype(features_df[col])
    ]
    assert non_numeric == [], f"Columnas no numéricas encontradas: {non_numeric}"


def test_no_bool_dtype_columns(features_df: pd.DataFrame) -> None:
    """Las columnas dummy de pd.get_dummies deben quedar como 0/1 (int),
    no como bool, para un CSV de salida consistente."""
    bool_cols = [c for c in features_df.columns if features_df[c].dtype == bool]
    assert bool_cols == []


class TestDecision1AgeExclusion:
    """correctedage y Age se excluyen: representan edad al momento de la
    evaluación Bayley, no una variable disponible al nacer."""

    def test_age_not_in_output(self, features_df: pd.DataFrame) -> None:
        assert "Age" not in features_df.columns

    def test_correctedage_not_in_output(self, features_df: pd.DataFrame) -> None:
        assert "correctedage" not in features_df.columns


class TestDecision2CongenitalAnomalyFix:
    """El valor espurio 34.0 (duplicado de PregnancyAge en la misma fila)
    debe recodificarse como nulo y luego imputarse, no interpretarse como
    una categoría real."""

    def test_no_value_34_survives_as_raw_category(
        self, features_df: pd.DataFrame
    ) -> None:
        # Tras encoding binario (0/1) + imputación, 34.0 nunca debe
        # aparecer como valor en esta columna.
        assert 34.0 not in features_df["congenital.anomaly"].unique()

    def test_congenital_anomaly_is_binary(self, features_df: pd.DataFrame) -> None:
        assert set(features_df["congenital.anomaly"].unique()).issubset({0, 1})


class TestDecision3RespiratoryIntervention:
    """surfactant y aggressive.ventilation se combinan en una sola
    variable derivada; las originales no deben sobrevivir por separado."""

    def test_combined_feature_exists(self, features_df: pd.DataFrame) -> None:
        assert "intervencion_respiratoria_agresiva" in features_df.columns

    def test_original_columns_removed(self, features_df: pd.DataFrame) -> None:
        assert "surfactant" not in features_df.columns
        assert "aggressive.ventilation" not in features_df.columns

    def test_combined_feature_is_binary(self, features_df: pd.DataFrame) -> None:
        values = set(features_df["intervencion_respiratoria_agresiva"].unique())
        assert values.issubset({0, 1})


class TestDecision4PregnancyComplicationRetained:
    """pregnancycomplication se mantiene junto a sus componentes, ya que
    12.8% de los casos positivos no tienen ningún componente individual
    marcado (aporta señal propia)."""

    def test_pregnancycomplication_present(self, features_df: pd.DataFrame) -> None:
        assert "pregnancycomplication" in features_df.columns

    def test_individual_components_still_present(
        self, features_df: pd.DataFrame
    ) -> None:
        for col in ["DM", "preeclampsia", "hypothyroid", "PROM", "IUGR"]:
            assert col in features_df.columns


class TestDomainLeakagePrevention:
    """Las variables de dominio de neurodesarrollo (fuente directa del
    target compuesto) no deben aparecer como predictores."""

    def test_no_domain_score_columns(self, features_df: pd.DataFrame) -> None:
        leakage_vars = [
            "CognitiveDomain",
            "LanguageDomain",
            "PerceptualDomain",
            "finemotor",
            "coarsemotor",
            "cog.cat",
            "lang.cat",
            "percep.cat",
            "fine.cat",
            "coarse.cat",
        ]
        for var in leakage_vars:
            assert var not in features_df.columns


class TestRegressionBugs:
    """Casos específicos que fallaron silenciosamente en la primera
    versión del pipeline, para que no se vuelvan a romper sin avisar."""

    def test_csf_culture_not_fully_null_after_encoding(
        self, features_df: pd.DataFrame
    ) -> None:
        """Bug original: el mapping usaba 'positive'/'negative' en
        minúsculas, pero el codebook trae 'Positive'/'Negatiev'
        (capitalizado, con typo), dejando la columna 100% nula."""
        assert "csf.culture" in features_df.columns
        assert features_df["csf.culture"].notna().all()

    def test_notaggressive_ventilation_is_encoded(
        self, features_df: pd.DataFrame
    ) -> None:
        """Bug original: esta variable no estaba en BINARY_YES_NO_VARS y
        pasaba sin codificar (string 'yes'/'no' en vez de 0/1)."""
        assert "notaggressive.ventilation" in features_df.columns
        assert pd.api.types.is_numeric_dtype(features_df["notaggressive.ventilation"])


class TestTarget:
    def test_target_column_present(self, features_df: pd.DataFrame) -> None:
        assert TARGET in features_df.columns

    def test_target_is_binary(self, features_df: pd.DataFrame) -> None:
        assert set(features_df[TARGET].unique()) == {0, 1}

    def test_target_distribution_matches_eda(self, features_df: pd.DataFrame) -> None:
        """Confirma que la distribución coincide con lo documentado en el
        EDA (47 abnormal / 42 normal, 52.8%/47.2%)."""
        counts = features_df[TARGET].value_counts()
        assert counts[1] == 47
        assert counts[0] == 42
