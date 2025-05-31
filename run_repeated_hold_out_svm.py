import os
from train_SVM import train_svm
from split_by_patient import split_index_by_patient

# Repeated hold-out settings
index_csv = "features/index.csv"
base_split_dir = "features/splits_repeated_SVM"
base_result_dir = "results/svm_repeated"
n_repeats = 5
feature_types = ["logmel", "mfcc"]

for repeat in range(n_repeats):
    seed = 42 + repeat
    split_dir = os.path.join(base_split_dir, f"repeat_{repeat}")
    result_dir = os.path.join(base_result_dir, f"repeat_{repeat}")

    split_index_by_patient(index_csv, split_dir, seed=seed)

    train_csv = os.path.join(split_dir, "index_train.csv")
    val_csv = os.path.join(split_dir, "index_val.csv")
    test_csv = os.path.join(split_dir, "index_test.csv")

    for feature_type in feature_types:
        out_dir = os.path.join(result_dir, feature_type)
        train_svm(train_csv, val_csv, test_csv, out_dir, feature_type=feature_type, n_components=200)
