import sys
import os
import cv2
import numpy as np
import face_recognition
import pickle
from sklearn import neighbors, svm

ENCODINGS_FILE = "face_encodings.pkl"

# Usa el mismo modelo que entrenaste, aquí elegimos SVM
USE_MODEL = "svm_model.pkl"

# Comprobación de argumentos
if len(sys.argv) != 2:
    print("Uso: python predict_face.py ruta_imagen.jpg")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print("❌ La imagen no existe:", image_path)
    sys.exit(1)

# Cargar embeddings y etiquetas
with open(ENCODINGS_FILE, "rb") as f:
    encodings, labels = pickle.load(f)

# Entrenar el modelo (rápido al tener ya los datos)
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(encodings, labels)

# Cargar imagen nueva
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detectar rostro
face_locations = face_recognition.face_locations(rgb)
if not face_locations:
    print("⚠️ No se detectó ningún rostro.")
    sys.exit(0)

face_encodings = face_recognition.face_encodings(rgb, face_locations)

# Umbral de confianza para considerar si es desconocido
THRESHOLD = 0.55

for i, encoding in enumerate(face_encodings):
    probs = clf.predict_proba([encoding])[0]
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]
    predicted = clf.classes_[best_idx]

    print(f"\n🧠 Rostro {i+1}:")
    if best_prob >= THRESHOLD:
        print(f"✅ Reconocido como: {predicted} (confianza: {best_prob:.2f})")
    else:
        print(f"❌ No reconocido (mejor coincidencia: {predicted} con {best_prob:.2f})")

