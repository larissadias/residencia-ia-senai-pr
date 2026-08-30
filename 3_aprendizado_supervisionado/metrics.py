import time
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

LATENCY_SAMPLES = 2000

def compute_metrics(model, X_test, y_test):
    """Computes every evaluation metric for a trained model."""
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        probabilities = predictions.astype(float)
        auc = 0.5

    return {
        "auc": auc,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision_class1": precision_score(y_test, predictions, zero_division=0),
        "recall_class1": recall_score(y_test, predictions, zero_division=0),
        "recall_class0": recall_score(y_test, predictions, pos_label=0, zero_division=0),
        "f1_class1": f1_score(y_test, predictions, zero_division=0),
        "confusion": confusion_matrix(y_test, predictions),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def measure_latency(model, X_test, n_samples=LATENCY_SAMPLES):
    """Measures how long the model takes to classify one sample, in milliseconds."""
    sample = X_test[:n_samples]

    start_time = time.perf_counter()
    model.predict(sample)
    prediction_time = time.perf_counter() -  start_time

    return 1000.0 * prediction_time / len(sample)


def best_threshold(y_test, probabilities):
    """Finds the decision threshold that maximizes the Youden's J statistic."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probabilities)

    youden_j = true_positive_rate - false_positive_rate
    best = int(youden_j.argmax())

    return thresholds[best]

def print_confusion(y_test, predictions, label):
    """Prints the confusion matrix with the reading of each cell."""
    matrix = confusion_matrix(y_test, predictions)
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]

    print(f"\nMatriz de confusão - {label}")
    print(f"{'':>18}{'previsto 0':>13}{'previsto 1':>13}")
    print(f"{'real 0':>18}{true_negative:>13}{false_positive:>13}")
    print(f"{'real 1':>18}{false_negative:>13}{true_positive:>13}")

    total_zeros = true_negative + false_positive
    print(f"  Da classe 0 ({total_zeros} casos), identificou {true_negative} "
          f"({100 * true_negative / total_zeros:.1f}%).")  


