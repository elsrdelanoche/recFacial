import os
import cv2
import numpy as np
import face_recognition
import pickle
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import defaultdict
from tqdm import tqdm

DATASET_DIR = "dataset"
ENCODINGS_FILE = "face_encodings.pkl"

def extract_embeddings():
    encodings = []
    labels = []
    print("Procesando imágenes para extraer embeddings...")

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        image_files = os.listdir(person_dir)
        for i in range(0, len(image_files), 10):  # procesar por lotes de 10
            batch = image_files[i:i + 10]
            for image_file in batch:
                img_path = os.path.join(person_dir, image_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"⚠️ No se pudo leer {img_path}")
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb, model="hog")
                if len(face_locations) != 1:
                    continue

                encs = face_recognition.face_encodings(rgb, face_locations)
                if encs:
                    encodings.append(encs[0])
                    labels.append(person)
    print(f"✅ Total de embeddings extraídos: {len(encodings)}")
    return encodings, labels

# Cargar embeddings desde archivo si existen
if os.path.exists(ENCODINGS_FILE):
    print("📦 Cargando embeddings guardados...")
    with open(ENCODINGS_FILE, "rb") as f:
        ENCODINGS, LABELS = pickle.load(f)
else:
    ENCODINGS, LABELS = extract_embeddings()
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((ENCODINGS, LABELS), f)
    print(f"💾 Embeddings guardados en '{ENCODINGS_FILE}'")

# Verificar número de clases
unique_classes = set(LABELS)
if len(unique_classes) < 2:
    print("❌ Error: se necesitan al menos 2 personas/clases para entrenar modelos de clasificación.")
    print(f"Actualmente sólo hay: {list(unique_classes)}")
    exit(1)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    ENCODINGS, LABELS, test_size=0.2, random_state=42, stratify=LABELS
)

print("\n🔍 Entrenando modelo K-NN...")
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', weights='distance')
knn_clf.fit(X_train, y_train)

print("🔍 Entrenando modelo SVM...")
svm_clf = svm.SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)

# Evaluación
def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n📊 --- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate(knn_clf, "K-NN")
evaluate(svm_clf, "SVM")

