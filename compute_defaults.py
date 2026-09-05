"""
Calcula valores por defecto (moda para binarias, mediana para continuas)
para las variables del modelo que el dashboard NO pregunta directamente.
Corre esto parado en la raíz del proyecto (Entrega 2).
"""

import pandas as pd

FREE_FIELDS = {"DM", "preeclampsia", "PregnancyAge", "BirthWeight", "apgar5"}
TARGET = "neurodev_alteration"

df = pd.read_csv("data/features/features.csv")
if TARGET in df.columns:
    df = df.drop(columns=[TARGET])

defaults = {}
for col in df.columns:
    if col in FREE_FIELDS:
        continue
    unique_vals = set(df[col].dropna().unique())
    if unique_vals <= {0, 1}:
        value = int(df[col].mode(dropna=True).iloc[0])
    else:
        value = float(df[col].median())
    defaults[col] = value

print("DEFAULT_VALUES: dict[str, float | int] = {")
for col, value in defaults.items():
    print(f'    "{col}": {value!r},')
print("}")
