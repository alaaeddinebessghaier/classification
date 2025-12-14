import os
import sys
import cv2
import joblib
import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for

# Ajouter la racine au PYTHONPATH pour trouver src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_image
from src.features import extract_hog

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger tous les modèles existants
# Chemin absolu vers le dossier models
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Charger les modèles existants
models = {
    "KNN": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
    "DecisionTree": joblib.load(os.path.join(MODEL_DIR, "DecisionTree.pkl")),
    "NaiveBayes": joblib.load(os.path.join(MODEL_DIR, "NaiveBayes.pkl"))
}


classes = ['child', 'adult', 'elderly']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    img_url = None
    selected_model = "KNN"

    if request.method == "POST":
        # Choix du modèle depuis le formulaire
        selected_model = request.form.get("model", "KNN")

        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", prediction=prediction_text, model=selected_model)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        results = []

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi_prep = preprocess_image(roi)
            feat = extract_hog(roi_prep).reshape(1, -1)
            pred_idx = models[selected_model].predict(feat)[0]
            label = classes[pred_idx]
            results.append(label)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        annotated_path = os.path.join(UPLOAD_FOLDER, "annotated_" + file.filename)
        cv2.imwrite(annotated_path, img)
        img_url = url_for('uploaded_file', filename="annotated_" + file.filename)
        prediction_text = ", ".join(results) if results else "Aucune personne détectée"

    return render_template("index.html", prediction=prediction_text, img_url=img_url, model=selected_model)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
