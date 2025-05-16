import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === CONFIG ===
df = pd.read_csv("features/index.csv")
os.makedirs("results/cluster_viz", exist_ok=True)

def load_features(paths):
    data = []
    for path in tqdm(paths):
        try:
            arr = np.load(path)
            data.append(arr.flatten())
        except Exception as e:
            print(f"⚠️ Failed to load {path}: {e}")
    return np.stack(data)

def plot_true_labels(X_2d, labels, feature_name):
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_enc, cmap='tab10', alpha=0.6)
    plt.title(f"{feature_name} – True Labels")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.colorbar(scatter, ticks=range(len(le.classes_)), label="Label")
    plt.tight_layout()
    plt.savefig(f"results/cluster_viz/{feature_name}_true_labels.png", dpi=300)
    plt.close()

def plot_kmeans(X_2d, kmeans_labels, feature_name, k):
    plt.figure(figsize=(6, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    plt.title(f"{feature_name} – KMeans Clustering (k={k})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(f"results/cluster_viz/{feature_name}_kmeans_k{k}.png", dpi=300)
    plt.close()

def run_clustering(feature_type):
    print(f"Processing {feature_type}")
    paths = df[f"{feature_type}_path"]
    labels = df["label"].values

    X = load_features(paths)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # True labels
    plot_true_labels(X_2d, labels, feature_type)

    # KMeans clusters
    for k in [2, 3]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X)
        plot_kmeans(X_2d, clusters, feature_type, k)

# === Run for both features ===
run_clustering("logmel")
run_clustering("mfcc")
