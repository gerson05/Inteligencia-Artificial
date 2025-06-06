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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test, save_dir="results"):
    """
    Imprime métricas de evaluación y guarda gráficas en la carpeta de resultados.
    """
    os.makedirs(save_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    # --- Reporte de clasificación ---
    print("\n📊 Classification Report:")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # --- Matriz de confusión ---
    print("🧩 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Heatmap de la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()

    # --- F1-score por clase ---
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.iloc[:-3][["f1-score"]].plot(
        kind="bar", legend=False, title="F1-score por clase", figsize=(8, 4), color="#2E86C1"
    )
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "f1_scores.png"))
    plt.show()

    # --- Importancia de características (Random Forest) ---
    if hasattr(model, "feature_importances_"):
        feature_names = [f"F{i}" for i in range(X_test.shape[1])]
        importances = model.feature_importances_
        sorted_idx = importances.argsort()[::-1]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="viridis")
        plt.title("Importancia de características")
        plt.xlabel("Importancia")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "feature_importance.png"))
        plt.show()



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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "annotated_data", "feature_data_detailed.csv")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Cargar y preparar datos
    X, y, scaler = load_and_prepare_data(data_path)

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar modelo
    model = train_model(X_train, y_train)

    # Evaluar modelo
    evaluate_model(model, X_test, y_test, save_dir=os.path.join(project_root, "results"))


    # Guardar modelo y scaler
    save_model(model, scaler, model_path, scaler_path)
