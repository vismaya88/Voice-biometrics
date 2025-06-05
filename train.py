import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from audio_cleaning import preprocess_audio
from feature_extractor import extract_mfcc
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data"
MODEL_PATH = "trained_models"
os.makedirs(MODEL_PATH, exist_ok=True)

X, y = [], []

# Combine both phrase types but label only by speaker ID
for folder in ['differentPhrase', 'samePhrase']:
    folder_path = os.path.join(DATA_PATH, folder)
    for speaker in os.listdir(folder_path):
        speaker_path = os.path.join(folder_path, speaker)
        if not os.path.isdir(speaker_path): continue
        for file in os.listdir(speaker_path):
            if file.endswith(".flac"):
                file_path = os.path.join(speaker_path, file)
                try:
                    y_audio, sr = preprocess_audio(file_path)
                    features = extract_mfcc(y_audio, sr)
                    if features is not None:
                        X.append(features)
                        y.append(speaker)  # ✅ Use only speaker ID
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save test set
np.save(os.path.join(MODEL_PATH, "X_test.npy"), X_test)
np.save(os.path.join(MODEL_PATH, "y_test.npy"), y_test)

# Define models
models = {
    "svm": SVC(kernel='rbf', probability=True),
    "rf": RandomForestClassifier(),
    "mlp": MLPClassifier(max_iter=500),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

best_model_name, best_model, best_score = None, None, 0

for name, clf in models.items():
    print(f"\nTraining {name.upper()} model...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name.upper()} Accuracy: {acc:.4f}")
    pickle.dump(clf, open(os.path.join(MODEL_PATH, f"{name}_model.pkl"), 'wb'))
    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_model = clf

# Save label encoder and best model info
pickle.dump(le, open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'wb'))
with open(os.path.join(MODEL_PATH, "best_model.txt"), "w") as f:
    f.write(best_model_name)

print(f"\n✅ Best model: {best_model_name.upper()} with accuracy {best_score:.4f}")
