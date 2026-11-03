import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    # Load the dataset
    data = load_breast_cancer()

    # Create feature matrix
    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Convert target:
    # 1 = malignant
    # 0 = benign
    y = pd.Series(
        (data.target == 0).astype(int),
        name="malignant"
    )

    return data, X, y