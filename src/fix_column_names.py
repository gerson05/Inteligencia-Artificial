# src/fix_column_names.py

import pandas as pd

df = pd.read_csv("data/annotated_data/preprocessed_data.csv")

# Generar nombres: x0, y0, x1, y1, ..., x65, y65 (132 columnas) + "label"
column_names = [f'x{i//2}' if i % 2 == 0 else f'y{i//2}' for i in range(132)]
column_names.append("label")

df.columns = column_names

# Guardar con nombres corregidos
df.to_csv("data/annotated_data/preprocessed_data_fixed.csv", index=False)
print("✅ Archivo corregido guardado como: preprocessed_data_fixed.csv")
