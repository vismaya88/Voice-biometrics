import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "trained_models"

X_test = np.load(f"{MODEL_PATH}/X_test.npy")
y_test = np.load(f"{MODEL_PATH}/y_test.npy")
le = pickle.load(open(f'{MODEL_PATH}/label_encoder.pkl', 'rb'))

model_names = ['svm', 'rf', 'mlp', 'knn', 'xgb']
summary_metrics = {}
best_model_name = ""
best_f1_weighted = 0

print("\n=== Model Performance Summary ===")

for model_name in model_names:
    try:
        model_path = f"{MODEL_PATH}/{model_name}_model.pkl"
        clf = pickle.load(open(model_path, "rb"))

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        f1_weighted = round(report['weighted avg']['f1-score'], 4)

        summary_metrics[model_name.upper()] = {
            'Accuracy': round(acc, 4),
            'Precision (macro avg)': round(report['macro avg']['precision'], 4),
            'Recall (macro avg)': round(report['macro avg']['recall'], 4),
            'F1-score (macro avg)': round(report['macro avg']['f1-score'], 4),
            'Precision (weighted avg)': round(report['weighted avg']['precision'], 4),
            'Recall (weighted avg)': round(report['weighted avg']['recall'], 4),
            'F1-score (weighted avg)': f1_weighted
        }

        # Select best model based on weighted F1-score
        if f1_weighted > best_f1_weighted:
            best_f1_weighted = f1_weighted
            best_model_name = model_name.upper()

    except Exception as e:
        print(f"Error with {model_name}: {e}")

# Print all model summaries
for model, metrics in summary_metrics.items():
    print(f"\n=== {model} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

# Print best model
print("\n=== Best Model Summary ===")
print(f"🏆 Best model based on Weighted F1-Score: {best_model_name} with F1-score {best_f1_weighted}")
