# src/gui.py

import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import mediapipe as mp
import numpy as np
import joblib

def extract_pose_features(landmarks):
    """
    Extrae SOLO el ángulo de la cadera (como en el entrenamiento) a partir de la lista de landmarks (x, y, z, visibility)
    """
    import numpy as np
    try:
        hombro = landmarks[12][:3]
        cadera = landmarks[24][:3]
        rodilla = landmarks[26][:3]
        from feature_extraction import calculate_angle
        angle_hip = calculate_angle(hombro, cadera, rodilla)
        return [angle_hip]  # SOLO una feature
    except Exception as e:
        print(f"Error extrayendo features: {e}")
        return None

# Cargar modelo y scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Variables de control
running = False

def get_landmark_list(results):
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    return [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]

def run_camera(label):
    global running
    cap = cv2.VideoCapture(0)

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        landmarks = get_landmark_list(results)

        if landmarks:
            features = extract_pose_features(landmarks)
            if features:
                X = np.array([features])
                X_scaled = scaler.transform(X)
                prediction = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled).max()

                text = f"Actividad: {prediction} ({proba:.2f})"
                cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                label.config(text=text)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mostrar en ventana separada
        cv2.imshow("Actividad en tiempo real", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection(label):
    global running
    if not running:
        running = True
        thread = threading.Thread(target=run_camera, args=(label,))
        thread.start()
    else:
        messagebox.showinfo("Ya está en ejecución", "La detección ya está corriendo.")

def stop_detection():
    global running
    running = False

def main():
    root = tk.Tk()
    root.title("Video Activity Detection")
    root.geometry("400x200")

    title = tk.Label(root, text="Reconocimiento de Actividad Humana", font=("Helvetica", 14))
    title.pack(pady=10)

    result_label = tk.Label(root, text="Esperando detección...", font=("Helvetica", 12))
    result_label.pack(pady=10)

    start_button = tk.Button(root, text="Iniciar webcam", command=lambda: start_detection(result_label))
    start_button.pack(pady=5)

    stop_button = tk.Button(root, text="Detener", command=stop_detection)
    stop_button.pack(pady=5)

    exit_button = tk.Button(root, text="Salir", command=root.destroy)
    exit_button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
