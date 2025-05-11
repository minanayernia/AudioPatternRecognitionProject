import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

def load_data(index_csv, feature_type="logmel"):
    df = pd.read_csv(index_csv)
    df = df[df['label'] != "Unknown"].reset_index(drop=True)
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {feature_type} features"):
        try:
            feature = np.load(row[f"{feature_type}_path"])
            X.append(feature.flatten())
            y.append(row['label'])
        except Exception as e:
            print(f"⚠️ Failed to load {row[f'{feature_type}_path']}: {e}")
    return np.array(X), np.array(y)

def weighted_accuracy(y_true, y_pred, label_encoder):
    labels = list(label_encoder.transform(["Normal", "Abnormal"]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape != (2, 2): return 0.0
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return (specificity + 2 * sensitivity) / 3

def train_knn(train_csv, val_csv, out_dir, feature_type="logmel", n_components=200):
    os.makedirs(out_dir, exist_ok=True)
    X_train, y_train_str = load_data(train_csv, feature_type)
    X_val, y_val_str = load_data(val_csv, feature_type)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)

    pca = PCA(n_components=n_components)
    X_train_p = pca.fit_transform(X_train)
    X_val_p = pca.transform(X_val)
    joblib.dump(pca, os.path.join(out_dir, "pca_knn.pkl"))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_p, y_train)
    joblib.dump(knn, os.path.join(out_dir, "knn_model.pkl"))

    y_pred = knn.predict(X_val_p)
    print("\n📊 KNN Evaluation:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred, average='macro'))
    print("Weighted Accuracy:", weighted_accuracy(y_val, y_pred, label_encoder))
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    train_knn(
        train_csv="features/splits/index_train.csv",
        val_csv="features/splits/index_val.csv",
        out_dir="results/knn_logmel/",
        feature_type="logmel"
    )
    train_knn(
        train_csv="features/splits/index_train.csv",
        val_csv="features/splits/index_val.csv",
        out_dir="results/knn_mfcc/",
        feature_type="mfcc"
    )
