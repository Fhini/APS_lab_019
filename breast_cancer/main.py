import pandas as pd

from data_loader import load_data
from preprocessing import split_data

from model import (
    create_model,
    train_model,
    predict_probabilities,
    predict_classes
)

from evaluation import (
    create_probability_results,
    apply_threshold,
    threshold_prediction_count,
    calculate_confusion_matrices,
    calculate_metrics,
    compare_thresholds
)

from visualization import plot_class_distribution


def main():

    print("=" * 60)
    print("BREAST CANCER LOGISTIC REGRESSION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    print("\n1. Loading dataset...")

    data, X, y = load_data()

    print("\nClass counts:")
    print(y.value_counts())

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class names:", data.target_names)

    # --------------------------------------------------
    # 2. CLASS DISTRIBUTION
    # --------------------------------------------------

    print("\n2. Class distribution...")

    class_counts = y.value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Class": data.target_names,
        "Count": class_counts.values,
        "Probability": class_counts.values / len(y)
    })

    print(class_distribution)

    # Plot
    plot_class_distribution(class_distribution)

    # --------------------------------------------------
    # 3. TRAIN TEST SPLIT
    # --------------------------------------------------

    print("\n3. Creating training and testing samples...")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("Training size:", len(y_train))
    print("Testing size:", len(y_test))

    print("\nTraining proportions:")
    print(
        y_train.value_counts(
            normalize=True
        ).sort_index()
    )

    print("\nTesting proportions:")
    print(
        y_test.value_counts(
            normalize=True
        ).sort_index()
    )

    # --------------------------------------------------
    # 4. CREATE AND TRAIN MODEL
    # --------------------------------------------------

    print("\n4. Training Logistic Regression model...")

    model = create_model()

    model = train_model(
        model,
        X_train,
        y_train
    )

    print("Model trained successfully.")

    # --------------------------------------------------
    # 5. PREDICT PROBABILITIES
    # --------------------------------------------------

    print("\n5. Predicting probabilities...")

    probabilities = predict_probabilities(
        model,
        X_test
    )

    print("\nFirst 5 probabilities:")
    print(probabilities[:5])

    # --------------------------------------------------
    # 6. CREATE RESULTS TABLE
    # --------------------------------------------------

    print("\n6. Creating probability results...")

    results = create_probability_results(
        y_test,
        probabilities
    )

    print(
        results[
            [
                "Actual_label",
                "P_malignant",
                "P_benign"
            ]
        ].head(10)
    )

    # --------------------------------------------------
    # 7. APPLY 0.50 THRESHOLD
    # --------------------------------------------------

    print("\n7. Applying threshold = 0.50...")

    results = apply_threshold(
        results,
        0.50
    )

    print(
        results[
            [
                "Actual_label",
                "P_malignant",
                "Predicted_label"
            ]
        ].head(10)
    )

    # --------------------------------------------------
    # 8. THRESHOLD ANALYSIS
    # --------------------------------------------------

    print("\n8. Threshold analysis...")

    threshold_prediction_count(
        results
    )

    # --------------------------------------------------
    # 9. CONFUSION MATRICES
    # --------------------------------------------------

    print("\n9. Confusion matrices...")

    calculate_confusion_matrices(
        y_test,
        probabilities
    )

    # --------------------------------------------------
    # 10. MODEL PREDICTIONS
    # --------------------------------------------------

    print("\n10. Model predictions...")

    predictions = predict_classes(
        model,
        X_test
    )

    # --------------------------------------------------
    # 11. MODEL METRICS
    # --------------------------------------------------

    print("\n11. Model evaluation...")

    calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # 12. COMPARE THRESHOLDS
    # --------------------------------------------------

    print("\n12. Comparing all thresholds...")

    metric_results = compare_thresholds(
        y_test,
        probabilities
    )

    print("\nThreshold comparison:")
    print(metric_results)

    # --------------------------------------------------
    # END
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()