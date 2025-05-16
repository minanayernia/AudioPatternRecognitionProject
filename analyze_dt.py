import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

MFCC_MODEL_PATH = 'results/dt_mfcc_unknown_label_v1/dt_model_mfcc.pkl'
LOGMEL_MODEL_PATH = 'results/dt_logmel_unknown_label_v1/dt_model_logmel.pkl'
OUTPUT_DIR = 'feature_importance_results'
TOP_N = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)



def load_model(path):
    return joblib.load(path)


def get_top_features(importances, top_n=TOP_N):
    indices = np.argsort(importances)[::-1][:top_n]
    return [(i, importances[i]) for i in indices]


def save_top_features_txt(features, filename, label_prefix='F'):
    with open(filename, 'w') as f:
        for idx, score in features:
            f.write(f"{label_prefix}{idx}: {score:.4f}\n")


def plot_importances(features, title, output_path, label_prefix='F'):
    labels = [f"{label_prefix}{idx}" for idx, _ in features]
    scores = [score for _, score in features]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, scores)
    plt.title(title)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Load models
clf_mfcc = load_model(MFCC_MODEL_PATH)
clf_logmel = load_model(LOGMEL_MODEL_PATH)

# Extract feature importances
importances_mfcc = clf_mfcc.feature_importances_
importances_logmel = clf_logmel.feature_importances_

# Get top N features
top_mfcc = get_top_features(importances_mfcc)
top_logmel = get_top_features(importances_logmel)

# Save results to text files
save_top_features_txt(top_mfcc, os.path.join(OUTPUT_DIR, 'top_mfcc_features.txt'), label_prefix='MFCC')
save_top_features_txt(top_logmel, os.path.join(OUTPUT_DIR, 'top_logmel_features.txt'), label_prefix='Mel')

# Save plots
plot_importances(top_mfcc, "Top MFCC Feature Importances", os.path.join(OUTPUT_DIR, 'mfcc_importance.png'),
                 label_prefix='MFCC')
plot_importances(top_logmel, "Top Log-Mel Band Importances", os.path.join(OUTPUT_DIR, 'logmel_importance.png'),
                 label_prefix='Mel')

print("Feature importance analysis complete. Results saved to:", OUTPUT_DIR)
