# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_prepare_data(path):
    """
    Carga las características desde un CSV y las divide en X, y
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El archivo {path} está vacío. Asegúrate de que contiene datos para entrenar el modelo.")
    X = df.drop(columns=["label"])
    y = df["label"]

    # Escalar características (muy útil para algunos modelos como SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_model(X_train, y_train):
    """
    Entrena un modelo de Random Forest
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    """
    Imprime el reporte de clasificación y matriz de confusión
    """
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, scaler, path_model, path_scaler):
    """
    Guarda el modelo y el scaler para inferencia futura
    """
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
    print(f"✅ Modelo guardado en: {path_model}")
    print(f"✅ Scaler guardado en: {path_scaler}")


if __name__ == "__main__":
    # Paths
    data_path = "c:/Users/gdjhb/Downloads/Inteligencia-Artificial/data/annotated_data/feature_data.csv"
    model_path = "c:/Users/gdjhb/Downloads/Inteligencia-Artificial/models/model.pkl"
    scaler_path = "c:/Users/gdjhb/Downloads/Inteligencia-Artificial/models/scaler.pkl"

    os.makedirs("models", exist_ok=True)

    # Cargar y preparar datos
    X, y, scaler = load_and_prepare_data(data_path)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar modelo
    model = train_model(X_train, y_train)

    # Evaluar modelo
    evaluate_model(model, X_test, y_test)

    # Guardar modelo y scaler
    save_model(model, scaler, model_path, scaler_path)
