# Heart Sound Classification with Log-Mel and MFCC Features

This project implements a full pipeline for classifying heart sound recordings using traditional machine learning models (SVM, KNN, Decision Tree) trained on log-Mel and MFCC features.

## Project Structure

```
Audio-Pattern-Recognition-Project/
├── data/                                # Raw .wav, .tsv, .txt files
├── features/
│   ├── logmel_segments/                # Saved .npy logmel segments
│   ├── mfcc_segments/                  # Saved .npy mfcc segments
│   └── splits/
│       ├── index_train.csv             # Patient-level split
│       ├── index_val.csv
│       └── index_test.csv
│   ├── index.csv                       # Full dataset index
├── feature_importance_results/         # Decision tree feature importance plots & top features
│   ├── logmel_importance.png
│   ├── mfcc_importance.png
│   ├── top_logmel_features.txt
│   └── top_mfcc_features.txt
├── results/
│   ├── dt_logmel_unknown_label_v1/
│   ├── dt_mfcc_unknown_label_v1/
│   ├── knn_logmel_with_unknown_label_v1/
│   ├── knn_mfccwith_unknown_label_v1/
│   ├── svm_logmel_unknown_label_v1/
│   ├── svm_mfcc_unknown_label_v1/
│   └── cluster_viz/                    # Clustering (k-means) and true label visualizations
├── preprocess_and_extract.py           # Full preprocessing & segmentation + feature extraction
├── split_by_patient.py                 # Create stratified patient-level train/val/test split
├── train_SVM.py                        # Train and evaluate SVM
├── train_KNN.py                        # Train and evaluate KNN
├── train_dt.py                         # Train and evaluate Decision Tree
├── analyze_dt.py                       # Extract DT feature importances
├── k_mean.py                           # Unsupervised clustering of features
├── report.pdf                          # Final report
└── .gitignore                          # Ignore features, models, large binaries
```

## Preprocessing Pipeline

1. **Segmentation**: 5s frame with 1s overlap using librosa.
2. **Feature Extraction**:

   * Log-Mel Spectrograms (128 Mel bins)
   * MFCC (13 coefficients)
3. **Label Extraction**:

   * From `#Murmur:` line in subject description `.txt`
   * Labels used: `Absent`, `Present`, and optionally `Unknown`

## Models Trained

* **Support Vector Machine (RBF)**
* **K-Nearest Neighbors (k=5)**
* **Decision Tree**

All models:

* Are trained separately for `logmel` and `mfcc` features.
* Use PCA (n=200) before classification.
* Are evaluated on validation and test sets.

## Evaluation

Each model generates:

* `*_val_metrics.txt`, `*_test_metrics.txt`: accuracy, F1, weighted accuracy, classification report, confusion matrix
* `*_model.pkl`, `pca_*.pkl`: trained models and PCA transformers
* `*_roc_curve.png`: ROC curve plot with AUC (only for binary classification — earlier experiments with just "Normal" and "Abnormal")

## Clustering & Feature Analysis

* `k_mean.py` performs k-means clustering (k=2, k=3) to assess separability of logmel and mfcc features.
* `analyze_dt.py` computes feature importance from decision trees.

## How to Run

```bash
# Preprocess and extract features
python preprocess_and_extract.py

# Split dataset by patient
python split_by_patient.py

# Train models
python train_SVM.py
python train_KNN.py
python train_dt.py

# Analyze results
python analyze_dt.py
python k_mean.py
```

## Dependencies

* Python 3.10+
* `librosa`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `joblib`

## Notes

* Final results are saved in folders ending with `_unknown_label_v1` (includes "Unknown" cases)
* ROC curve visualizations are only generated for binary label setups
* All `.pkl` models and heavy features are excluded from version control via `.gitignore`
* No deep learning models were used — the focus is on classical ML and interpretability
