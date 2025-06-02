# src/feature_extraction.py

import numpy as np
import pandas as pd

def calculate_angle(a, b, c):
    """
    Calcula el ángulo entre tres puntos (a, b, c) usando producto punto.
    Los puntos deben ser vectores de 2 o 3 dimensiones (x, y) o (x, y, z).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_features_from_landmarks(df):
    """
    Extrae ángulos y otras características útiles desde un DataFrame con landmarks.
    Se espera que el DataFrame tenga columnas nombradas como: x0, y0, x1, y1, ..., xN, yN, label
    """
    # Número de puntos = número total de columnas (menos 'label') dividido por 2
    landmark_cols = [col for col in df.columns if col != "label"]
    num_points = len(landmark_cols) // 2

    features = []

    for idx, row in df.iterrows():
        row_features = {}

        # Ejemplo: calcular ángulo entre hombro-hiproderecha-piernaderecha
        try:
            shoulder = (row["x11"], row["y11"])
            hip = (row["x23"], row["y23"])
            knee = (row["x25"], row["y25"])
            angle_hip = calculate_angle(shoulder, hip, knee)
            row_features["angle_hip"] = angle_hip
        except:
            row_features["angle_hip"] = np.nan

        # Velocidad (diferencia entre puntos consecutivos, si hay frame_id)
        # Esto es opcional, dependiendo del dataset
        
        # Agregar más características aquí...
        
        row_features["label"] = row["label"]
        features.append(row_features)

    df_features = pd.DataFrame(features)
    df_features = df_features.dropna()  # Eliminar filas con NaN en los ángulos
    return df_features


def process_and_save(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print("📊 Extrayendo características...")
    df_features = extract_features_from_landmarks(df)
    df_features.to_csv(output_csv, index=False)
    print(f"✅ Características guardadas en: {output_csv}")


if __name__ == "__main__":
    input_path = "c:/Users/CTecn/Desktop/Inteligencia-Artificial/data/annotated_data/preprocessed_data.csv"
    output_path = "c:/Users/CTecn/Desktop/Inteligencia-Artificial/data/annotated_data/feature_data.csv"
    process_and_save(input_path, output_path)


