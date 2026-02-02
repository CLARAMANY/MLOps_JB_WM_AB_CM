
import os
import pandas as pd

def main():
    os.makedirs("data/raw", exist_ok=True)

    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)

    print("Dataset shape:", df.shape)

    df.to_csv("data/raw/housing_raw.csv", index=False)

if __name__ == "__main__":
    main()
