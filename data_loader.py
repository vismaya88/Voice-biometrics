import os
import librosa
import shap
import pickle

# Load your model
svm_clf = pickle.load(open('models/svm_model.pkl', 'rb'))
rf_clf = pickle.load(open('models/rf_model.pkl', 'rb'))
mlp_clf = pickle.load(open('models/mlp_model.pkl', 'rb'))

# Feature extraction should already be done using the extract_mfcc function


def load_audio_data(data_path, sample_rate=16000):
    audio_list = []
    label_list = []
    for speaker in os.listdir(data_path):
        speaker_folder = os.path.join(data_path, speaker)
        if os.path.isdir(speaker_folder):
            for file in os.listdir(speaker_folder):
                if file.endswith(".flac"):
                    file_path = os.path.join(speaker_folder, file)
                    signal, _ = librosa.load(file_path, sr=sample_rate)
                    audio_list.append(signal)
                    label_list.append(speaker)
    return audio_list, label_list