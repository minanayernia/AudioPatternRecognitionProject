import csv
import os
def count_unique_patient_ids(filepath):
    unique_ids = set()
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if row:
                patient_id = row[0]
                unique_ids.add(patient_id)
    return len(unique_ids)

def count_total_audio_files(directory):
    count = 0
    for file in directory:
        if file.endswith('.wav'):
            count += 1
    return count


print("Total unique patient IDs:", count_unique_patient_ids("features/index.csv"))
print("Unique patient IDs in train set:", count_unique_patient_ids("features/splits/index_train.csv"))
print("Unique patient IDs in val set:", count_unique_patient_ids("features/splits/index_val.csv"))
print("Unique patient IDs in test set:", count_unique_patient_ids("features/splits/index_test.csv"))

print("Total audio files in dataset:", count_total_audio_files(os.listdir("data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data")))