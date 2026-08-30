from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import RAW_DATA_PATH, TARGET, load_labeled
from src.features.build_features import build_features


def _make_synthetic_row(**overrides: object) -> dict[str, object]:
    """Fila base con valores 'normales' representativos del dataset real
    (post load_labeled: strings etiquetados, no códigos numéricos). Los
    tests sobreescriben solo los campos relevantes para cada caso."""
    base = {
        "number": 1.0,
        "Sex": "male",
        "DM": "no",
        "preeclampsia": "no",
        "hypothyroid": "no",
        "PROM": "no",
        "IUGR": "NO",
        "pregnancycomplication": "no",
        "pneumothorax": "no",
        "NEC": "no",
        "sepsis": "no",
        "PDA": "no",
        "icter": "yes",
        "meningitis": "no",
        "IVH": "no",
        "seizure": "no",
        "BPD": "no",
        "apgar1": 7.0,
        "apgar5": 9.0,
        "B.C": "negative",
        "PregnancyAge": 32.0,
        "drug.mother": "no",
        "duration.hopitalization": 14.0,
        "laborType": "NVD",
        "type.of.ressucitation": np.nan,
        "ehya.badve.tavallod": "NO",
        "BirthWeight": 1650.0,
        "RoundHeadAtBirth": 30.0,
        "surfactant": "no",
        "aggressive.ventilation": "no",
        "notaggressive.ventilation": "yes",
        "csf.culture": "Negatiev",
        "congenital.anomaly": "no",
        "duration.O2": 1.0,
        "SepsisnegativeCulture": "no",
        "mother.sonogarphy.result": "normal",
        "CognitiveDomain": "n",
        "LanguageDomain": "n",
        "PerceptualDomain": "n",
        "finemotor": "n",
        "coarsemotor": "n",
        "correctedage": 400.0,
        "Age": 400.0,
        "lang.recode": "Normal",
        "percep.recode": "Normal",
        "fine.recode": "Normal",
        "coarse.recode": "Normal",
        "cog.recode": "Normal",
        "lang.cat": "normal",
        "percep.cat": "normal",
        "fine.cat": "normal",
        "cog.cat": "normal",
        "coarse.cat": "normal",
        TARGET: "normal",
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """Dataset sintético pequeño y determinístico que cubre, a propósito,
    los casos límite relevantes para cada decisión de preprocesamiento.
    No depende del archivo .sav real, por lo que corre igual en CI que
    en local."""
    rows = [
        _make_synthetic_row(number=1.0),
        _make_synthetic_row(number=2.0, **{"coarse.cat": "abnormal"}),
        # Caso decisión 2: congenital.anomaly con el valor espurio real (34.0)
        _make_synthetic_row(number=3.0, **{"congenital.anomaly": "34.0"}),
        # Caso decisión 3: surfactant=yes, aggressive.ventilation=no (discordante)
        _make_synthetic_row(
            number=4.0, surfactant="yes", **{"aggressive.ventilation": "no"}
        ),
        # Caso decisión 3: ambas=yes (concordante)
        _make_synthetic_row(
            number=5.0, surfactant="yes", **{"aggressive.ventilation": "yes"}
        ),
        # Caso decisión 4: pregnancycomplication=yes SIN ningún componente individual
        _make_synthetic_row(
            number=6.0,
            pregnancycomplication="yes",
            DM="no",
            preeclampsia="no",
            hypothyroid="no",
            PROM="no",
            IUGR="NO",
        ),
        # Caso regresión: csf.culture con el typo real del codebook ("Negatiev")
        _make_synthetic_row(number=7.0, **{"csf.culture": "Negatiev"}),
        _make_synthetic_row(number=8.0, **{"csf.culture": "Positive"}),
        # Caso: type.of.ressucitation nulo (66% de los casos reales)
        _make_synthetic_row(number=9.0, **{"type.of.ressucitation": np.nan}),
        _make_synthetic_row(number=10.0, **{"type.of.ressucitation": "ppv"}),
    ]
    df = pd.DataFrame(rows)
    return df


@pytest.fixture(scope="module")
def features_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    return build_features(synthetic_df)


def test_no_nulls_in_output(features_df: pd.DataFrame) -> None:
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
    bool_cols = [c for c in features_df.columns if features_df[c].dtype == bool]
    assert bool_cols == []


class TestDecision1AgeExclusion:
    def test_age_not_in_output(self, features_df: pd.DataFrame) -> None:
        assert "Age" not in features_df.columns

    def test_correctedage_not_in_output(self, features_df: pd.DataFrame) -> None:
        assert "correctedage" not in features_df.columns


class TestDecision2CongenitalAnomalyFix:
    def test_no_value_34_survives(self, features_df: pd.DataFrame) -> None:
        assert 34.0 not in features_df["congenital.anomaly"].unique()

    def test_congenital_anomaly_is_binary(self, features_df: pd.DataFrame) -> None:
        assert set(features_df["congenital.anomaly"].unique()).issubset({0, 1})


class TestDecision3RespiratoryIntervention:
    def test_combined_feature_exists(self, features_df: pd.DataFrame) -> None:
        assert "intervencion_respiratoria_agresiva" in features_df.columns

    def test_original_columns_removed(self, features_df: pd.DataFrame) -> None:
        assert "surfactant" not in features_df.columns
        assert "aggressive.ventilation" not in features_df.columns

    def test_combined_is_or_of_originals(self, synthetic_df: pd.DataFrame) -> None:
        """Fila 4 (surfactant=yes, ventilation=no) y fila 5 (ambas=yes)
        deben dar 1 en la variable combinada; la fila base (ambas=no)
        debe dar 0. build_features() elimina la columna 'number' (no es
        un predictor), por lo que se referencia por posición: el orden
        de las filas se preserva, no se filtran registros."""
        result = build_features(synthetic_df).reset_index(drop=True)
        assert result.loc[3, "intervencion_respiratoria_agresiva"] == 1  # number=4.0
        assert result.loc[4, "intervencion_respiratoria_agresiva"] == 1  # number=5.0
        assert result.loc[0, "intervencion_respiratoria_agresiva"] == 0  # number=1.0


class TestDecision4PregnancyComplicationRetained:
    def test_pregnancycomplication_present(self, features_df: pd.DataFrame) -> None:
        assert "pregnancycomplication" in features_df.columns

    def test_individual_components_still_present(
        self, features_df: pd.DataFrame
    ) -> None:
        for col in ["DM", "preeclampsia", "hypothyroid", "PROM", "IUGR"]:
            assert col in features_df.columns


class TestDomainLeakagePrevention:
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

    def test_csf_culture_negatiev_maps_to_zero(
        self, synthetic_df: pd.DataFrame
    ) -> None:
        result = build_features(synthetic_df).reset_index(drop=True)
        assert result.loc[6, "csf.culture"] == 0  # number=7.0

    def test_csf_culture_positive_maps_to_one(self, synthetic_df: pd.DataFrame) -> None:
        result = build_features(synthetic_df).reset_index(drop=True)
        assert result.loc[7, "csf.culture"] == 1  # number=8.0

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
        assert set(features_df[TARGET].unique()).issubset({0, 1})


@pytest.mark.skipif(
    not Path(RAW_DATA_PATH).exists(),
    reason=(
        "Requiere data/raw/preterm_infant.sav (DVC), no disponible en "
        "CI hasta que se resuelva PR #8"
    ),
)
class TestRealDataIntegration:
    """Validación end-to-end contra el dataset real. Se salta
    automáticamente en entornos sin el archivo .sav (p.ej. CI hasta que
    el remoto DVC esté configurado), pero corre en local donde el
    archivo sí está presente."""

    def test_output_shape(self) -> None:
        result = build_features(load_labeled())
        assert result.shape[0] == 89

    def test_target_distribution_matches_eda(self) -> None:
        """Confirma que la distribución coincide con lo documentado en el
        EDA (47 abnormal / 42 normal, 52.8%/47.2%)."""
        result = build_features(load_labeled())
        counts = result[TARGET].value_counts()
        assert counts[1] == 47
        assert counts[0] == 42
