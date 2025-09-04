import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

RANDOM_STATE = 42
TARGETS = ["math score", "reading score", "writing score"]
DATA_PATH = "data/StudentsPerformance.csv"

def load_xy():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=TARGETS)
    y = df[TARGETS]
    return df, X, y

def make_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="drop"
    )
    return pre

def split_sets(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate(model, X, y, label=""):
    preds = model.predict(X)
    mses, maes, r2s = [], [], []
    for i, t in enumerate(TARGETS):
        y_true = y.iloc[:, i].values
        y_pred = preds[:, i]
        mses.append(mean_squared_error(y_true, y_pred))
        maes.append(mean_absolute_error(y_true, y_pred))
        r2s.append(r2_score(y_true, y_pred))
    print(f"\n=== {label} ===")
    print("MSE:", ", ".join([f"{k}: {v:.3f}" for k, v in dict(zip(TARGETS, mses)).items()]))
    print("MAE:", ", ".join([f"{k}: {v:.3f}" for k, v in dict(zip(TARGETS, maes)).items()]))
    print("R2 :", ", ".join([f"{k}: {v:.3f}" for k, v in dict(zip(TARGETS, r2s)).items()]))
    print(f"Avg → MSE {np.mean(mses):.3f} | MAE {np.mean(maes):.3f} | R2 {np.mean(r2s):.3f}")

def main():
    df, X, y = load_xy()
    pre = make_preprocessor(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_sets(X, y)

    pipe = Pipeline([
        ("pre", pre),
        ("mlp", MLPRegressor(
            random_state=RANDOM_STATE,
            max_iter=3000,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1
        ))
    ])

    grid = {
        "mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
        "mlp__alpha": [1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-3, 3e-3],
    }

    gs = GridSearchCV(pipe, grid, scoring="r2", cv=5, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    best = gs.best_estimator_
    evaluate(best, X_train, y_train, "Train")
    evaluate(best, X_val,   y_val,   "Validation")

    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0)
    best.fit(X_trval, y_trval)
    evaluate(best, X_test, y_test, "Test")

    joblib.dump(best, "best_model.joblib")
    print("\nSaved best model to best_model.joblib")

if __name__ == "__main__":
    main()
