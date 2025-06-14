# src/feature_extraction.py

import os
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
    Extrae múltiples características (ángulos, inclinaciones, distancias) desde landmarks.
    Requiere columnas nombradas como x0, y0, x1, y1, ..., xN, yN, y una columna "label".
    """

    features = []

    for idx, row in df.iterrows():
        row_features = {}
        try:
            # ----- Ángulos -----
            # Cadera derecha (hombro - cadera - rodilla)
            row_features["angle_hip"] = calculate_angle(
                (row["x11"], row["y11"]), (row["x23"], row["y23"]), (row["x25"], row["y25"])
            )

            # Rodilla derecha (cadera - rodilla - tobillo)
            row_features["angle_knee"] = calculate_angle(
                (row["x23"], row["y23"]), (row["x25"], row["y25"]), (row["x27"], row["y27"])
            )

            # Codo derecho (hombro - codo - muñeca)
            row_features["angle_elbow"] = calculate_angle(
                (row["x11"], row["y11"]), (row["x13"], row["y13"]), (row["x15"], row["y15"])
            )

            # Hombro derecho (cuello - hombro - codo)
            row_features["angle_shoulder"] = calculate_angle(
                (row["x5"], row["y5"]), (row["x11"], row["y11"]), (row["x13"], row["y13"])
            )

            # ----- Inclinación del tronco -----
            shoulder_mid = np.mean([row["x11"], row["x12"]]), np.mean([row["y11"], row["y12"]])
            hip_mid = np.mean([row["x23"], row["x24"]]), np.mean([row["y23"], row["y24"]])
            inclinacion = np.arctan2(hip_mid[1] - shoulder_mid[1], hip_mid[0] - shoulder_mid[0])
            row_features["inclinacion_tronco"] = np.degrees(inclinacion)

            # ----- Distancia entre hombros -----
            dist_hombros = np.linalg.norm(
                np.array([row["x11"], row["y11"]]) - np.array([row["x12"], row["y12"]])
            )
            row_features["dist_hombros"] = dist_hombros

        except KeyError as e:
            print(f"⚠️ Faltan coordenadas necesarias en la fila {idx}: {e}")
            continue
        except Exception as e:
            print(f"⚠️ Error general en fila {idx}: {e}")
            continue

        row_features["label"] = row["label"]
        features.append(row_features)

    df_features = pd.DataFrame(features)
    df_features = df_features.dropna()
    return df_features



def process_and_save(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    print("📊 Extrayendo características...")
    df_features = extract_features_from_landmarks(df)
    df_features.to_csv(output_csv, index=False)
    print(f"✅ Características guardadas en: {output_csv}")


if __name__ == "__main__":
    # Usar rutas relativas al proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_path = os.path.join(project_root, "data", "annotated_data", "preprocessed_data_fixed.csv")
    output_path = os.path.join(project_root, "data", "annotated_data", "feature_data_detailed.csv")

    process_and_save(input_path, output_path)