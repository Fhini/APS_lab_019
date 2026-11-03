from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def create_model():
    # Create Logistic Regression model
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    return model


def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train)

    return model


def predict_probabilities(model, X_test):
    # Predict probabilities
    probabilities = model.predict_proba(X_test)

    return probabilities


def predict_classes(model, X_test):
    # Predict class labels
    predictions = model.predict(X_test)

    return predictions