import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def create_probability_results(y_test, probabilities):
    # Create results DataFrame

    results = pd.DataFrame({
        "Actual_class": y_test.values,
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1]
    })

    # Convert actual class to label
    results["Actual_label"] = results["Actual_class"].map({
        0: "Malignant",
        1: "Benign"
    })

    return results


def apply_threshold(results, threshold):
    # Apply threshold to malignant probability

    results["Predicted_malignant"] = (
        results["P_malignant"] >= threshold
    ).astype(int)

    results["Predicted_label"] = results[
        "Predicted_malignant"
    ].map({
        1: "Malignant",
        0: "Benign"
    })

    return results


def threshold_prediction_count(results):
    # Check number of predicted malignant cases
    for threshold in [0.30, 0.50, 0.70]:

        predictions = (
            results["P_malignant"] >= threshold
        ).astype(int)

        print(
            f"Threshold = {threshold}: "
            f"Predicted malignant cases = {predictions.sum()}"
        )


def calculate_confusion_matrices(y_test, probabilities):
    # Construct confusion matrices

    actual_malignant = (
        y_test.values == 0
    ).astype(int)

    for threshold in [0.30, 0.50, 0.70]:

        predicted_malignant = (
            probabilities[:, 0] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            actual_malignant,
            predicted_malignant
        )

        print(f"\nThreshold = {threshold}")
        print(cm)


def calculate_metrics(y_test, predictions):
    # Calculate sklearn metrics

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print("Sklearn accuracy:", accuracy)
    print("Sklearn precision:", precision)
    print("Sklearn recall:", recall)
    print("Sklearn f1-score:", f1)

    return accuracy, precision, recall, f1


def compare_thresholds(y_test, probabilities):
    # Compare all metrics across thresholds

    metric_results = []

    for threshold in [0.10, 0.30, 0.50, 0.70, 0.90]:

        y_pred_threshold = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred_threshold,
            labels=[0, 1]
        ).ravel()

        metric_results.append({
            "Threshold": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": accuracy_score(
                y_test,
                y_pred_threshold
            ),
            "Precision": precision_score(
                y_test,
                y_pred_threshold
            ),
            "Recall": recall_score(
                y_test,
                y_pred_threshold
            ),
            "F1 Score": f1_score(
                y_test,
                y_pred_threshold
            )
        })

    metric_results = pd.DataFrame(
        metric_results
    )

    return metric_results