
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    os.makedirs("report", exist_ok=True)

    # load data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # load preprocessor + model
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/model.joblib")

    # transform + predict
    X_test_prepared = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_prepared)

    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")

    results = {
        "model": "RandomForestRegressor",
        "mae": float(mae),
        "rmse": float(rmse)
    }

    with open("report/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
