import noisereduce as nr
import librosa
import numpy as np

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    # Noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    # Silence trimming
    y_trimmed, _ = librosa.effects.trim(y_denoised)
    return y_trimmed, sr
