from sklearn.dummy import DummyRegressor, DummyClassifier
def train_baseline_regression(X_train, y_train):
    # e.g. mean‐predictor
    model = DummyRegressor(strategy="mean")
    return model.fit(X_train, y_train)

def train_baseline_classification(X_train, y_train):
    # e.g. most‐frequent direction
    model = DummyClassifier(strategy="most_frequent")
    return model.fit(X_train, y_train)
