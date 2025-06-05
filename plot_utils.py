import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa.display
from sklearn.metrics import confusion_matrix

def plot_mfcc_heatmap(y, sr, title='MFCC Heatmap'):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_prediction_probabilities(probs, class_names, title='Prediction Probabilities'):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=probs, y=class_names, palette="viridis")
    plt.xlabel('Probability')
    plt.title(title)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()
