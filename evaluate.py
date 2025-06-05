import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "trained_models"

X_test = np.load(f"{MODEL_PATH}/X_test.npy")
y_test = np.load(f"{MODEL_PATH}/y_test.npy")
le = pickle.load(open(f'{MODEL_PATH}/label_encoder.pkl', 'rb'))

model_names = ['svm', 'rf', 'mlp', 'knn', 'xgb']
accuracies = {}

for model_name in model_names:
    try:
        print(f"\n=== {model_name.upper()} Evaluation ===")
        model_path = f"{MODEL_PATH}/{model_name}_model.pkl"
        clf = pickle.load(open(model_path, "rb"))

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[model_name.upper()] = acc
        print(f"Accuracy: {acc:.4f}")

        present_classes = np.unique(np.concatenate((y_test, y_pred)))
        target_names = le.inverse_transform(present_classes)

        print("Classification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=present_classes,
            target_names=target_names.astype(str),
            zero_division=0
        ))

        cm = confusion_matrix(y_test, y_pred, labels=present_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{model_name.upper()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Summary
print("\n=== Summary ===")
for model, acc in accuracies.items():
    print(f"{model} Accuracy: {acc:.4f}")
best_model = max(accuracies, key=accuracies.get)
print(f"Best model: {best_model} with accuracy {accuracies[best_model]:.4f}")
