# src/data_preprocessing.py

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_landmark_data(landmark_dir):
    """
    Carga todos los CSV de landmarks desde la carpeta dada.
    Añade una columna 'label' basada en el nombre del archivo.
    """
    data_frames = []
    for file in os.listdir(landmark_dir):
        if file.endswith(".csv"):
            path = os.path.join(landmark_dir, file)
            df = pd.read_csv(path)
            
            # Inferir la clase desde el nombre del archivo: caminar_espalda_user1_espalda_back_1.csv
            label = file.split("_")[0]  # caminar, girar, etc.
            df["label"] = label
            
            data_frames.append(df)
    
    if not data_frames:
        raise ValueError("No se encontraron archivos CSV en la carpeta.")
    
    return pd.concat(data_frames, ignore_index=True)

def clean_and_normalize(df):
    """
    Limpia y normaliza las columnas numéricas (landmarks).
    """
    # Quitar filas completamente vacías o con muchos NaN
    df_cleaned = df.dropna(thresh=int(0.8 * len(df.columns)))
    
    # Separar features y etiqueta
    features = df_cleaned.drop(columns=["label"])
    labels = df_cleaned["label"].values
    
    # Normalizar
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    df_normalized = pd.DataFrame(features_scaled, columns=features.columns)
    df_normalized["label"] = labels
    
    return df_normalized

def preprocess_all(landmark_dir, output_path):
    """
    Proceso completo de carga, limpieza y normalización. Guarda un archivo final.
    """
    print("📥 Cargando archivos CSV...")
    df = load_landmark_data(landmark_dir)
    
    print("🧹 Limpiando y normalizando...")
    df_preprocessed = clean_and_normalize(df)
    
    print(f"💾 Guardando en {output_path}")
    df_preprocessed.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_dir = "c:/Users/CTecn/Desktop/Inteligencia-Artificial/data/processed_landmarks"
    output_file = "c:/Users/CTecn/Desktop/Inteligencia-Artificial/data/annotated_data/preprocessed_data.csv"
    os.makedirs("c:/Users/CTecn/Desktop/Inteligencia-Artificial/data/annotated_data", exist_ok=True)
    
    preprocess_all(input_dir, output_file)
