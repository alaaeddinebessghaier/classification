import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import preprocess_image
from src.features import extract_hog

# ---------------------------
# CONFIGURATION
# ---------------------------
DATA_FOLDER = "data/Child-Adult-Elderly"
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

CLASSES = ['child', 'adult', 'elderly']

# ---------------------------
# 1. LOAD DATA
# ---------------------------
def load_data(folder):
    csv_path = os.path.join(DATA_FOLDER, folder, "_classes.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")
    
    df = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for _, row in df.iterrows():
        # Determine label from one-hot encoding
        label = [c for c in CLASSES if row[c]==1]
        if not label:
            continue
        label = label[0]
        img_path = os.path.join(DATA_FOLDER, folder, row['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_pre = preprocess_image(img)
        feat = extract_hog(img_pre)
        images.append(feat)
        labels.append(label)
    
    return np.array(images), np.array(labels)

X_train, y_train = load_data("train")
X_valid, y_valid = load_data("valid")
X_test, y_test = load_data("test")

X_train_val = np.concatenate([X_train, X_valid])
y_train_val = np.concatenate([y_train, y_valid])

# Convert labels to indices
class_to_idx = {c:i for i,c in enumerate(CLASSES)}
y_train_val_idx = np.array([class_to_idx[c] for c in y_train_val])
y_test_idx = np.array([class_to_idx[c] for c in y_test])

# ---------------------------
# 2. TRAIN MODELS
# ---------------------------
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
dt = DecisionTreeClassifier(max_depth=10, criterion='gini')
nb = GaussianNB()

models = {"KNN": knn, "DecisionTree": dt, "NaiveBayes": nb}

for name, model in models.items():
    model.fit(X_train_val, y_train_val_idx)
    print(f"{name} trained.")

# ---------------------------
# 3. EVALUATION
# ---------------------------
metrics_list = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    metrics_list.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_idx, y_pred),
        "Precision": precision_score(y_test_idx, y_pred, average="macro"),
        "Recall": recall_score(y_test_idx, y_pred, average="macro"),
        "F1-score": f1_score(y_test_idx, y_pred, average="macro")
    })
    
    # Confusion matrix
    cm = confusion_matrix(y_test_idx, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(os.path.join(RESULTS_FOLDER, f"cm_{name}.png"))
    plt.close()

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)
metrics_df.to_csv(os.path.join(RESULTS_FOLDER, "metrics_comparison.csv"), index=False)

# ---------------------------
# 4. SHOW SAMPLE IMAGES AND HOG
# ---------------------------
sample_folder = os.path.join(DATA_FOLDER, "train")
sample_images = [f for f in os.listdir(sample_folder) if f.endswith((".jpg", ".png"))][:5]

for fname in sample_images:
    img_path = os.path.join(sample_folder, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pre = preprocess_image(img)
    
    # HOG visualization
    feat, hog_img = hog(img_pre, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1,2,2)
    plt.imshow(hog_img, cmap="gray")
    plt.title("HOG Features")
    plt.axis("off")
    
    plt.savefig(os.path.join(RESULTS_FOLDER, f"sample_{fname}.png"))
    plt.close()

print(f"All results saved in '{RESULTS_FOLDER}' folder.")
