
## 🧠 Reconocimiento Facial con Machine Learning en Python

Este proyecto implementa un sistema de reconocimiento facial usando embeddings generados por la librería `face_recognition` (basada en `dlib`) y clasificadores tradicionales (`SVM`, `K-NN`). Está pensado como un ejercicio práctico para la materia de Machine Learning.

### 📌 ¿Qué hace este proyecto?

1. Extrae **embeddings faciales** de un conjunto de imágenes organizadas por persona.
2. Entrena dos modelos de clasificación (K-Nearest Neighbors y Support Vector Machines).
3. Evalúa ambos modelos y reporta precisión.
4. Permite hacer predicciones sobre nuevas imágenes con rostros no vistos.
5. Guarda los datos procesados para evitar reentrenamiento en cada ejecución.

### ⚙️ ¿Cómo funciona?

* Se usa `dlib` + `face_recognition` para detectar rostros y extraer vectores de características (embeddings de 128 dimensiones).
* Cada imagen con rostro detectado produce un vector asociado a una etiqueta (nombre).
* Se entrena un clasificador `K-NN` y otro `SVM` con estos vectores.
* Las predicciones se hacen comparando un nuevo embedding contra los vectores del dataset.
* Se usa un umbral de confianza para aceptar o rechazar una predicción.

### 🤖 ¿Dónde están las redes neuronales?
El aprendizaje profundo ya está **preentrenado** dentro de `face_recognition`, que usa la red neuronal convolucional del modelo `dlib_face_recognition_resnet_model_v1`:
* Esta red convierte una imagen facial en un vector de 128 dimensiones.
* Aunque tú no entrenas esta red, **sí usas sus embeddings como entrada** a tus propios clasificadores (`SVM`)
* El aprendizaje que tú haces es **supervisado clásico**, no entrenamiento profundo.

### 🧩 Requisitos

* Python 3.11 recomendado (por compatibilidad con `opencv-python`)
* Arch Linux (u otra distro con compilación de dlib)
* Librerías:

```bash
pip install face_recognition dlib opencv-python scikit-learn numpy tqdm
```

---

### 🗂️ Estructura del proyecto

```
face-recognition-ml/
├── dataset/                   # Directorio con subcarpetas por persona
│   ├── Emma Watson/
│   ├── Tom Hanks/
│   └── ...
├── face_recog_classifier.py  # Script de entrenamiento
├── predict_face.py           # Script de predicción
├── face_encodings.pkl        # Embeddings procesados (se genera)
├── svm_model.pkl             # Modelo entrenado (opcional si usas joblib)
├── README.md
```

---

### 🧪 ¿Cómo usarlo?

1. Coloca tus imágenes en `dataset/NOMBRE/imagen.jpg`.
2. Ejecuta el script de entrenamiento:

```bash
python face_recog_classifier.py
```

3. Una vez entrenado, puedes hacer predicciones:

```bash
python predict_face.py ruta/a/imagen.jpg
```
