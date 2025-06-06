# src/real_time_inference.py

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from gui import extract_pose_features

# Cargar modelo y scaler entrenado
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_landmark_list(results):
    """
    Extrae landmarks de pose y los convierte a una lista (x, y, z, visibility)
    """
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    return [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ No se puede abrir la cámara")
        return

    print("🎥 Iniciando predicción en tiempo real (presiona Q para salir)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error leyendo frame")
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

                # Mostrar predicción sobre el frame
                cv2.putText(
                    frame,
                    f'Actividad: {prediction} ({proba:.2f})',
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # Dibuja landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Real-Time Activity Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
