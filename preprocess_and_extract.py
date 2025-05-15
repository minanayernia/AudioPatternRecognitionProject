import os
import librosa
from librosa import feature
import numpy as np
import pandas as pd

def preprocess_audio(y, sr, target_sr=22050):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y, target_sr

def segment_audio(y, sr, frame_duration=5.0, overlap_sec=1.0):
    frame_size = int(sr * frame_duration)
    step_size = int(sr * (frame_duration - overlap_sec))
    segments = []
    for start in range(0, len(y) - frame_size + 1, step_size):
        segments.append(y[start:start + frame_size])
    return segments


def extract_logmel(y, sr, n_mels=128, hop_length=256, n_fft=1024):
    assert callable(librosa.feature.melspectrogram), " melspectrogram is not callable! It was likely overwritten."

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    logmel = librosa.power_to_db(mel)
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel) + 1e-8)
    return logmel


def extract_mfcc(y, sr, n_mfcc=13, hop_length=256, n_fft=1024):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc

# ---------------------------
# Extract label from subject .txt file using Murmur status only
# ---------------------------
def get_label_from_txt(subject_id, subject_dir):
    txt_path = os.path.join(subject_dir, f"{subject_id}.txt")
    if not os.path.exists(txt_path):
        return "Unknown"

    try:
        with open(txt_path, "r") as f:
            for line in f:
                if line.startswith("#Murmur:"):
                    murmur = line.split(":")[1].strip().lower()
                    print(f"murmur : {murmur}")
                    return murmur.capitalize()  # Capitalize to match "Present" or "Absent"
                    # if murmur == "Present":
                    #     return "Present"
                    # elif murmur == "Absent":
                    #     return "Absent"
        return "Unknown"
    except Exception as e:
        print(f"Error reading label from {txt_path}: {e}")
        return "Unknown"


# ---------------------------
# Main dataset processing
# ---------------------------
def process_dataset(data_dir, index_csv):
    print("Processing dataset...")
    os.makedirs("features/logmel_segments/", exist_ok=True)
    os.makedirs("features/mfcc_segments/", exist_ok=True)
    index = []

    for fname in os.listdir(data_dir):
        if not fname.endswith(".wav"):
            continue

        path = os.path.join(data_dir, fname)
        patient_id = fname.split("_")[0]
        label = get_label_from_txt(patient_id, data_dir)

        try:
            y, sr = librosa.load(path, sr=None)
            y, sr = preprocess_audio(y, sr)
            segments = segment_audio(y, sr, frame_duration=5.0, overlap_sec=1.0)

            for i, seg in enumerate(segments):

                # LOGMEL FEATURE
                logmel = extract_logmel(seg, sr)
                logmel_name = f"{os.path.splitext(fname)[0]}_seg{i}.npy"
                logmel_path = os.path.join("features/logmel_segments/", logmel_name)
                np.save(logmel_path, logmel)

                # MFCC FEATURE
                mfcc = extract_mfcc(seg, sr)
                mfcc_name = f"{os.path.splitext(fname)[0]}_seg{i}_mfcc.npy"
                mfcc_path = os.path.join("features/mfcc_segments/", mfcc_name)
                np.save(mfcc_path, mfcc)

                index.append({
                    "patient_id": patient_id,
                    "filename": fname,
                    "segment_id": i,
                    "label": label,
                    "logmel_path": logmel_path,
                    "mfcc_path": mfcc_path
                })

            print(f"Processed {fname} into {len(segments)} segments")

        except Exception as e:
            print(f"Failed on {fname}: {e}")

    df = pd.DataFrame(index)
    df.to_csv(index_csv, index=False)
    print(f"Saved index file to {index_csv}")


if __name__ == "__main__":
    process_dataset(
        data_dir="data/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data",
        index_csv="features/index.csv"
    )
