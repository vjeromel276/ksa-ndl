if __name__ == "__main__":
    # 1) load data
    X, y = load_features(...), load_targets(...)
    # 2) split
    X_tr, X_te, y_tr, y_te = train_test_split_time_series(...)
    # 3) train
    model = train_baseline_classification(X_tr, y_tr)
    # 4) evaluate
    preds = model.predict(X_te)
    print("Accuracy:", return_accuracy(y_te, preds))
