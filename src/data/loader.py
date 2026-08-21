"""
Carga y etiquetado del dataset NeoRisk (preterm_infant.sav).

Fuente: Darabi, A., Faramarzi, R., Boskabadi, H., Maamouri, G., Rezvani, R. (2024).
Dataset on neonatal and maternal factors influencing neurodevelopmental
outcomes in preterm infants. Mendeley Data.
https://data.mendeley.com/datasets/h464gsf77t/2

Este módulo centraliza la carga de datos crudos (.sav) y la aplicación de las
etiquetas de valor definidas en el codebook, de forma que todos los bloques
del proyecto (EDA, correlaciones, features, modelado) partan de la misma
representación de los datos.
"""

from pathlib import Path

import pandas as pd
import pyreadstat

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "preterm_infant.sav"

# Columnas conocidas por bloque temático (según codebook). Útil para EDA
# segmentado y para features.py más adelante.
MATERNAL_VARS = [
    "DM",
    "preeclampsia",
    "hypothyroid",
    "PROM",
    "IUGR",
    "pregnancycomplication",
    "drug.mother",
    "mother.sonogarphy.result",
]

NEONATAL_COMPLICATION_VARS = [
    "pneumothorax",
    "NEC",
    "sepsis",
    "PDA",
    "icter",
    "meningitis",
    "IVH",
    "seizure",
    "BPD",
    "congenital.anomaly",
]

INTERVENTION_VARS = [
    "surfactant",
    "aggressive.ventilation",
    "notaggressive.ventilation",
    "type.of.ressucitation",
    "laborType",
]

CONTINUOUS_VARS = [
    "PregnancyAge",
    "BirthWeight",
    "apgar1",
    "apgar5",
    "duration.hopitalization",
    "duration.O2",
    "RoundHeadAtBirth",
]

NEURODEV_DOMAIN_VARS = [
    "CognitiveDomain",
    "LanguageDomain",
    "PerceptualDomain",
    "finemotor",
    "coarsemotor",
]

# Variables categóricas normal/abnormal por dominio (Escalas Bayley), usadas
# para construir la variable objetivo compuesta. Ver Sección 1.2 del reporte:
# sepsis fue descartada como target por desbalance severo (5.6% positivos);
# esta variable compuesta resultó ser la más balanceada (52.8%).
DOMAIN_CAT_VARS = [
    "cog.cat",
    "lang.cat",
    "percep.cat",
    "fine.cat",
    "coarse.cat",
]

TARGET = "neurodev_alteration"

# Variables con inconsistencia conocida entre el codebook (escala ordinal
# 1-3) y los valores reales observados (enteros grandes, p.ej. 204, 820),
# que sugieren edad en días. Se documentan aquí para no perder el hallazgo
# y decidir en equipo cómo tratarlas antes de usarlas en modelado.
FLAGGED_INCONSISTENT_VARS = ["correctedage", "Age"]


def load_raw(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Lee el .sav crudo y retorna el DataFrame junto con los metadatos
    (incluye las etiquetas de valor originales de SPSS)."""
    df, meta = pyreadstat.read_sav(str(path))
    return df, meta.variable_value_labels


def load_labeled(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Retorna el dataset con las variables categóricas mapeadas a sus
    etiquetas (p.ej. 1.0/2.0 -> 'yes'/'no') en lugar de códigos numéricos.
    Las variables continuas y las de dominio neurodesarrollo se dejan tal
    cual, salvo que su codebook indique lo contrario.
    """
    df, value_labels = load_raw(path)
    df_labeled = df.copy()

    for col, mapping in value_labels.items():
        if col in df_labeled.columns:
            df_labeled[col] = df_labeled[col].map(mapping).fillna(df_labeled[col])

    # Variable objetivo compuesta: "abnormal" si al menos uno de los 5
    # dominios de neurodesarrollo (Escalas Bayley) es anormal, "normal" en
    # caso contrario. Ver Sección 1.2 / 6.5 del reporte para la justificación.
    is_abnormal = (df_labeled[DOMAIN_CAT_VARS] == "abnormal").any(axis=1)
    df_labeled[TARGET] = is_abnormal.map({True: "abnormal", False: "normal"})

    return df_labeled


if __name__ == "__main__":
    df = load_labeled()
    print(f"Shape: {df.shape}")
    print(df.head())
