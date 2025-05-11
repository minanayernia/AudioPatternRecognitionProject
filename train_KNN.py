# train_knn.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import matplotlib.pyplot as plt

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

def save_metrics(out_dir, name, acc, f1, wacc, report, cm, fpr, tpr, roc_auc):
    with open(os.path.join(out_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Weighted Accuracy: {wacc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{name}_roc_curve.png"))
    plt.close()

def evaluate_on_test(test_csv, out_dir, feature_type, label_encoder):
    model_path = os.path.join(out_dir, f"knn_model.pkl")
    pca_path = os.path.join(out_dir, f"pca_knn.pkl")

    X_test, y_test_str = load_data(test_csv, feature_type)
    y_test = label_encoder.transform(y_test_str)

    model = joblib.load(model_path)
    pca = joblib.load(pca_path)
    X_test_p = pca.transform(X_test)

    y_pred = model.predict(X_test_p)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_p)[:, 1] if len(label_encoder.classes_) == 2 else None
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    wacc = weighted_accuracy(y_test, y_pred, label_encoder)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print("\n🧪 Test Set Evaluation:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Weighted Accuracy:", wacc)
    print(report)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        save_metrics(out_dir, f"knn_{feature_type}_test", acc, f1, wacc, report, cm, fpr, tpr, roc_auc)

def train_knn(train_csv, val_csv, test_csv, out_dir, feature_type="logmel", n_components=200):
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
    if hasattr(knn, "predict_proba"):
        y_prob = knn.predict_proba(X_val_p)[:, 1] if len(label_encoder.classes_) == 2 else None
    else:
        y_prob = None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    wacc = weighted_accuracy(y_val, y_pred, label_encoder)
    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_val, y_pred)

    print("\n📊 KNN Evaluation:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Weighted Accuracy:", wacc)
    print(report)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        save_metrics(out_dir, f"knn_{feature_type}", acc, f1, wacc, report, cm, fpr, tpr, roc_auc)

    evaluate_on_test(test_csv, out_dir, feature_type, label_encoder)

if __name__ == "__main__":
    train_knn(
        train_csv="features/splits/index_train.csv",
        val_csv="features/splits/index_val.csv",
        test_csv="features/splits/index_test.csv",
        out_dir="results/knn_logmel/",
        feature_type="logmel"
    )
    train_knn(
        train_csv="features/splits/index_train.csv",
        val_csv="features/splits/index_val.csv",
        test_csv="features/splits/index_test.csv",
        out_dir="results/knn_mfcc/",
        feature_type="mfcc"
    )
