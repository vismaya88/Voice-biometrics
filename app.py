from flask import Flask, render_template, request
import os
import pickle
from werkzeug.utils import secure_filename
from audio_cleaning import preprocess_audio
from feature_extractor import extract_mfcc

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
MODEL_FOLDER = "trained_models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load best model from trained_models/
with open(os.path.join(MODEL_FOLDER, "best_model.txt"), "r") as f:
    best_model_name = f.read().strip()

model = pickle.load(open(os.path.join(MODEL_FOLDER, f"{best_model_name}_model.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(MODEL_FOLDER, "label_encoder.pkl"), "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return "No file uploaded", 400

    file = request.files['audio']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    y_audio, sr = preprocess_audio(filepath)
    features = extract_mfcc(y_audio, sr)
    if features is None:
        return "Feature extraction failed", 500

    pred_proba = model.predict_proba([features])[0]
    pred = model.predict([features])[0]
    speaker_label = encoder.inverse_transform([pred])[0]  # e.g., '45'

    confidence = max(pred_proba) * 100

    return render_template('result.html', speaker=speaker_label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
