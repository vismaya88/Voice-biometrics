from scipy.spatial.distance import cosine
from audio_cleaning import preprocess_audio
from feature_extractor import extract_mfcc

def is_same_speaker(file1, file2, threshold=0.5):
    y1, sr1 = preprocess_audio(file1)
    y2, sr2 = preprocess_audio(file2)
    mfcc1 = extract_mfcc(y1, sr1)
    mfcc2 = extract_mfcc(y2, sr2)
    similarity = 1 - cosine(mfcc1, mfcc2)
    return similarity > threshold
