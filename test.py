import librosa
import numpy as np

y, sr = librosa.load(librosa.example('trumpet'))
print("DEBUG:", librosa.feature.melspectrogram)
print("TYPE:", type(librosa.feature.melspectrogram))
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
print("✅ Success! Shape:", mel.shape)
