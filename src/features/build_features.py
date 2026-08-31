"""
Feature engineering y preprocesamiento para NeuroRisk (ticket #9).

Toma el DataFrame etiquetado producido por src/data/loader.py y lo
transforma en una matriz con encoding de variables categóricas y
variables derivadas, lista para que un pipeline de modelado (ticket #10)
aplique imputación y normalización DESPUÉS del train/test split.

Nota sobre imputación y normalización (ver PR #15, comentario de
@mhgualdron): estos pasos se excluyen deliberadamente de este script.
Calcular mediana/media/std/moda sobre el dataset completo antes de
separar train/test filtraría información del set de prueba hacia el
entrenamiento (data leakage). Esas estadísticas deben ajustarse
únicamente sobre X_train (p.ej. mediante sklearn.Pipeline con
SimpleImputer + StandardScaler) y aplicarse a X_test ya ajustadas. Por
eso los nulos genuinos (no los recodificados por error de captura, ver
decisión 2) se preservan como NaN en la salida de este script.

Decisiones de preprocesamiento (ver justificación completa en el PR /
data/README.md):

1. correctedage y Age se excluyen como predictores. Su rango (130-1190
   días) corresponde a edad en el momento de la evaluación Bayley, no a
   una variable temprana disponible al momento del nacimiento. Usarlas
   como predictor entra en conflicto con la pregunta de negocio (predecir
   a partir de condiciones de nacimiento, para intervención temprana).

2. congenital.anomaly = 34.0 se recodifica como nulo. Se verificó que ese
   valor coincide exactamente con PregnancyAge de la misma fila (34.0),
   consistente con un error de captura (duplicación de columna adyacente)
   y no con una categoría real. Esto es limpieza de datos (corregir un
   error conocido), no imputación estadística, por lo que se hace aquí
   y no en el pipeline de modelado.

3. surfactant y aggressive.ventilation se combinan en una sola variable
   derivada (intervencion_respiratoria_agresiva = OR de ambas). Coinciden
   en 95.5% de los casos; los 4 casos discordantes no muestran una
   dirección consistente, por lo que mantenerlas por separado añadiría
   redundancia sin aportar señal adicional clara.

4. pregnancycomplication se conserva junto a sus componentes individuales
   (DM, preeclampsia, hypothyroid, PROM, IUGR). Aunque está fuertemente
   asociada con ellos, 6 de 47 casos positivos (12.8%) no tienen ningún
   componente individual marcado, lo que indica que captura señal propia.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import TARGET, load_labeled

FEATURES_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "features" / "features.csv"
)

# Variables excluidas como predictores, con su justificación documentada
# arriba. Se excluyen explícitamente en vez de simplemente "no usarlas",
# para que quede claro en el código que la omisión es deliberada.
EXCLUDED_VARS = ["correctedage", "Age", "number"]

# Variables de dominio de neurodesarrollo: se excluyen como predictores
# porque son la fuente directa del target compuesto (usarlas causaría
# fuga de información trivial).
DOMAIN_LEAKAGE_VARS = [
    "CognitiveDomain",
    "LanguageDomain",
    "PerceptualDomain",
    "finemotor",
    "coarsemotor",
    "lang.recode",
    "percep.recode",
    "fine.recode",
    "coarse.recode",
    "cog.recode",
    "lang.cat",
    "percep.cat",
    "fine.cat",
    "cog.cat",
    "coarse.cat",
]

BINARY_YES_NO_VARS = [
    "Sex",
    "DM",
    "preeclampsia",
    "hypothyroid",
    "PROM",
    "IUGR",
    "pregnancycomplication",
    "pneumothorax",
    "NEC",
    "sepsis",
    "PDA",
    "icter",
    "meningitis",
    "IVH",
    "seizure",
    "BPD",
    "B.C",
    "drug.mother",
    "ehya.badve.tavallod",
    "csf.culture",
    "congenital.anomaly",
    "SepsisnegativeCulture",
    "mother.sonogarphy.result",
    "notaggressive.ventilation",
]

MULTI_CATEGORY_VARS = ["laborType", "type.of.ressucitation"]


def _fix_congenital_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Decisión 2: recodifica el valor espurio 34.0 como nulo."""
    df = df.copy()
    df.loc[df["congenital.anomaly"] == "34.0", "congenital.anomaly"] = np.nan
    # También cubre el caso en que el valor llegue como float sin castear
    df["congenital.anomaly"] = df["congenital.anomaly"].replace(34.0, np.nan)
    return df


def _combine_respiratory_intervention(df: pd.DataFrame) -> pd.DataFrame:
    """Decisión 3: combina surfactant y aggressive.ventilation en una sola
    variable derivada (OR lógico), y elimina las dos originales."""
    df = df.copy()
    df["intervencion_respiratoria_agresiva"] = (
        (df["surfactant"] == "yes") | (df["aggressive.ventilation"] == "yes")
    ).map({True: "yes", False: "no"})
    df = df.drop(columns=["surfactant", "aggressive.ventilation"])
    return df


def _encode_binary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Codifica variables yes/no (y variantes YES/NO, positive/negative,
    male/female) a 0/1. Comparación insensible a mayúsculas/minúsculas, e
    incluye el typo heredado del codebook fuente ("Negatiev" en
    csf.culture, ver data/README.md). Los nulos se preservan para
    imputarse después."""
    df = df.copy()
    mapping = {
        "yes": 1,
        "no": 0,
        "positive": 1,
        "negative": 0,
        "negatiev": 0,  # typo del codebook original
        "male": 1,
        "female": 0,
        "abnormal": 1,
        "normal": 0,
    }
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: mapping.get(v.lower(), np.nan) if isinstance(v, str) else v
            )
    return df


def _impute_and_flag_resuscitation(df: pd.DataFrame) -> pd.DataFrame:
    """type.of.ressucitation tiene 66.3% de nulos, probablemente no
    aleatorios (solo aplica si hubo reanimación). Se trata como categoría
    propia en vez de imputarse estadísticamente."""
    df = df.copy()
    df["type.of.ressucitation"] = df["type.of.ressucitation"].fillna("no_resuscitation")
    return df


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pipeline completo de feature engineering.

    Parameters
    ----------
    df : DataFrame etiquetado (salida de load_labeled()). Si es None, se
        carga automáticamente.

    Returns
    -------
    DataFrame con encoding aplicado y la columna TARGET codificada como
    0/1. Los nulos genuinos (no los errores de captura corregidos en la
    decisión 2) se preservan como NaN: la imputación y normalización se
    delegan al pipeline de modelado (#10), que debe ajustarlas solo
    sobre X_train para evitar data leakage (ver nota al inicio del
    módulo).
    """
    if df is None:
        df = load_labeled()

    df = df.copy()

    # Decisión 1: excluir correctedage, Age y variables de dominio
    # (evita fuga de información hacia el target compuesto).
    cols_to_drop = [c for c in EXCLUDED_VARS + DOMAIN_LEAKAGE_VARS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Decisión 2
    df = _fix_congenital_anomaly(df)

    # Decisión 3
    df = _combine_respiratory_intervention(df)

    # Decisión 4: pregnancycomplication no requiere transformación especial,
    # se mantiene y se codifica junto con el resto de binarias.

    binary_cols = [c for c in BINARY_YES_NO_VARS if c in df.columns] + [
        "intervencion_respiratoria_agresiva"
    ]
    df = _encode_binary(df, binary_cols)

    # Chequeo de seguridad: si una columna quedó 100% nula tras el
    # encoding, es casi seguro un bug en el mapping (ver el caso real de
    # csf.culture con "Negatiev"/"Positive" capitalizados), no un patrón
    # legítimo de datos faltantes. Falla ruidosamente en vez de dejarlo
    # pasar silenciosamente.
    for col in binary_cols:
        if col in df.columns and df[col].isna().all():
            raise ValueError(
                f"La columna '{col}' quedó completamente vacía tras el "
                "encoding. Revisar el mapping en _encode_binary."
            )

    df = _impute_and_flag_resuscitation(df)
    df = pd.get_dummies(
        df,
        columns=[c for c in MULTI_CATEGORY_VARS if c in df.columns],
        prefix=MULTI_CATEGORY_VARS,
        dummy_na=False,
    )

    # Cast dummy columns (bool tras get_dummies) a 0/1 explícito, para que
    # el CSV de salida sea limpio y no dependa de interpretar True/False.
    dummy_cols = [c for c in df.columns if df[c].dtype == bool]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # Los nulos genuinos restantes (p.ej. congenital.anomaly tras la
    # decisión 2) se preservan deliberadamente. Imputarlos aquí con la
    # moda del dataset completo sería el mismo problema de leakage que
    # las funciones de imputación continua que se removieron (ver nota
    # al inicio del módulo). Corresponde a la etapa de modelado (#10)
    # ajustar un imputer únicamente sobre X_train.

    # Target: abnormal/normal -> 1/0
    df[TARGET] = df[TARGET].map({"abnormal": 1, "normal": 0})

    return df


def save_features(df: pd.DataFrame, path: Path = FEATURES_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    features_df = build_features()
    save_features(features_df)
    print(f"Features shape: {features_df.shape}")
    print(f"Guardado en: {FEATURES_OUTPUT_PATH}")
