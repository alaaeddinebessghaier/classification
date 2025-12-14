import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import preprocess_image
from src.features import extract_hog
import cv2
import sys

# Ajouter le dossier racine du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Dossiers de données
folders = [
    "data/Child-Adult-Elderly/train",
    "data/Child-Adult-Elderly/valid",
    "data/Child-Adult-Elderly/test"
]

X, y = [], []

def get_label(row):
    if row['child'] == 1:
        return 'child'
    elif row['adult'] == 1:
        return 'adult'
    elif row['elderly'] == 1:
        return 'elderly'
    else:
        return None

# Charger images et extraire features
for folder in folders:
    csv_path = os.path.join(folder, "_classes.csv")
    if not os.path.exists(csv_path):
        print(f"Fichier CSV introuvable : {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        img_path = os.path.join(folder, row['filename'])
        if not os.path.exists(img_path):
            print(f"Image introuvable : {img_path}")
            continue
        label = get_label(row)
        if label is None:
            continue
        try:
            img = preprocess_image(cv2.imread(img_path))
            feat = extract_hog(img)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f"Erreur avec {img_path}: {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise ValueError("Aucune image chargée. Vérifie tes dossiers et CSV.")

# Classes et indices
classes = sorted(set(y))
class_to_idx = {cls:i for i, cls in enumerate(classes)}
y_idx = np.array([class_to_idx[c] for c in y])

# Séparation train/validation/test
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_idx, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

# Modèles
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "NaiveBayes": GaussianNB()
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"\n=== {name} ===")
    print("Train score:", model.score(X_train, y_train))
    print("Validation score:", model.score(X_val, y_val))
    print("Test score:", model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(model, f"models/{name}.pkl")
    print(f"{name} sauvegardé dans models/{name}.pkl")
