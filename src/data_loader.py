import pandas as pd

# src/data_loader.py
import pandas as pd
import os

def load_adult_data(path="data/adult.csv"):
    """
    Load local adult.csv if available; otherwise fallback to UCI URL.
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = [
            "age","workclass","fnlwgt","education","educational-num",
            "marital-status","occupation","relationship","race","gender",
            "capital-gain","capital-loss","hours-per-week","native-country","income"
        ]
        df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
        return df.dropna()

if __name__ == "__main__":
    df2 = load_adult_data()   # 修正函数名
    print("\nAdult Dataset:")
    print(df2.head())

