import pandas as pd

def load_diabetes_data(path="data/diabetes.csv"):
    ##  Load the local diabetes.csv dataset
    return pd.read_csv(path)

def load_adult_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    df = df.dropna()
    return df

if __name__ == "__main__":
    df1 = load_diabetes_data()
    print("🩺 Diabetes Dataset:")
    print(df1.head())

    df2 = load_adult_dataset()
    print("\n Adult Dataset:")
    print(df2.head())
