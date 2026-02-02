
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

def main():
    os.makedirs("models", exist_ok=True)

    # load data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    # load preprocessor
    preprocessor = joblib.load("models/preprocessor.joblib")

    # transform
    X_train_prepared = preprocessor.transform(X_train)

    # train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_prepared, y_train)

    joblib.dump(model, "models/model.joblib")

if __name__ == "__main__":
    main()
