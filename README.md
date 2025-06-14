# Integrantes
## Rafaela Ruiz - A00395368
## Samuel Alvarez - A00394750
## Gerson Hurtado - A00394995


## Funcionamiento de la aplicación

Esta aplicación permite reconocer actividades humanas a partir de videos o en tiempo real usando la webcam, mediante el uso de visión por computadora y aprendizaje automático. El flujo general es:

1. **Extracción de landmarks:** Se procesan videos para extraer puntos clave del cuerpo (landmarks) usando MediaPipe.
2. **Generación de características:** A partir de los landmarks, se calculan ángulos y distancias relevantes para cada frame.
3. **Entrenamiento del modelo:** Se entrena un modelo de machine learning (Random Forest) con las características extraídas para clasificar actividades.
4. **Inferencia en tiempo real:** El sistema puede predecir la actividad de una persona usando la webcam, mostrando el resultado en pantalla.

## Paso a paso para correr la aplicación

1. **Clona o descarga este repositorio.**
2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Opcional) Procesa los videos para extraer landmarks:**
   Ejecuta el notebook `notebooks/02_landmark_extraction.ipynb` para generar los archivos CSV de landmarks a partir de los videos en `data/raw_videos`.
4. **Genera las características:**
   Ejecuta el notebook `notebooks/03_feature_engineering.ipynb` para crear el archivo de features a partir de los landmarks.
5. **Explora los datos (opcional pero recomendado):**
   Ejecuta el notebook `notebooks/01_data_exploration.ipynb` para visualizar y analizar las características extraídas.
6. **Entrena el modelo:**
   Ejecuta el script:
   ```bash
   python src/model_training.py
   ```
   Esto generará los archivos `models/model.pkl` y `models/scaler.pkl`.
7. **Corre la inferencia en tiempo real:**
   Ejecuta el script:
   ```bash
   python src/gui.py
   ```
   Se abrirá una ventana donde podrás iniciar la webcam y ver la predicción de actividad en tiempo real.

**Notas:**
- Asegúrate de tener una webcam conectada para la inferencia en tiempo real.
- Puedes modificar los scripts y notebooks para agregar nuevas actividades, features o mejorar el modelo.
