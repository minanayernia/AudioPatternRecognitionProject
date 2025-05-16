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
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {feature_type} features"):
        try:
            feature = np.load(row[f"{feature_type}_path"])
            X.append(feature.flatten())
            y.append(row['label'])
        except Exception as e:
            print(f"Failed to load {row[f'{feature_type}_path']}: {e}")
    return np.array(X), np.array(y)

def weighted_accuracy(y_true, y_pred, label_encoder):
    # Decode integer labels back to strings
    y_true_str = label_encoder.inverse_transform(y_true)
    y_pred_str = label_encoder.inverse_transform(y_pred)

    labels = ["Absent", "Present", "Unknown"]
    cm = {true: {pred: 0 for pred in labels} for true in labels}

    # Build confusion matrix manually
    for yt, yp in zip(y_true_str, y_pred_str):
        cm[yt][yp] += 1

    # Extract counts from matrix
    mAA = cm["Absent"]["Absent"]
    mPP = cm["Present"]["Present"]
    mUU = cm["Unknown"]["Unknown"]

    sum_iA = sum(cm[i]["Absent"] for i in labels)
    sum_iP = sum(cm[i]["Present"] for i in labels)
    sum_iU = sum(cm[i]["Unknown"] for i in labels)

    numerator = mAA + 5 * mPP + 3 * mUU
    denominator = sum_iA + 5 * sum_iP + 3 * sum_iU

    return numerator / (denominator + 1e-8)

def save_metrics(out_dir, name, acc, f1, wacc, report, cm, fpr, tpr, roc_auc):
    with open(os.path.join(out_dir, f"{name}_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Weighted Accuracy: {wacc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))


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

    print("Test Set Evaluation:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Weighted Accuracy:", wacc)
    print(report)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        save_metrics(out_dir, f"knn_{feature_type}_test", acc, f1, wacc, report, cm, fpr, tpr, roc_auc)
    else:
        save_metrics(out_dir, f"knn_{feature_type}_test", acc, f1, wacc, report, cm, None, None, None)


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
    y_prob = knn.predict_proba(X_val_p)[:, 1] if hasattr(knn, "predict_proba") and len(label_encoder.classes_) == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='macro')
    wacc = weighted_accuracy(y_val, y_pred, label_encoder)
    report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
    cm = confusion_matrix(y_val, y_pred)

    print("KNN Evaluation (Validation Set):")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Weighted Accuracy:", wacc)
    print(report)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        save_metrics(out_dir, f"knn_{feature_type}_val", acc, f1, wacc, report, cm, fpr, tpr, roc_auc)
    else:
        save_metrics(out_dir, f"knn_{feature_type}_val", acc, f1, wacc, report, cm, None, None, None)

    evaluate_on_test(test_csv, out_dir, feature_type, label_encoder)

if __name__ == "__main__":
    train_knn(
        train_csv="features/splits/index_train.csv",
        val_csv="features/splits/index_val.csv",
        test_csv="features/splits/index_test.csv",
        out_dir="results/knn_logmel_with_unknown_label_v1/",
        feature_type="logmel"
    )
    train_knn(
        train_csv="features/splits/index_train.csv",
        val_csv="features/splits/index_val.csv",
        test_csv="features/splits/index_test.csv",
        out_dir="results/knn_mfccwith_unknown_label_v1/",
        feature_type="mfcc"
    )
