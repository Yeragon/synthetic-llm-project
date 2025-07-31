# test_loader.py
# test git is Ok
from src.data_loader import load_diabetes_data

df = load_diabetes_data()

print("前5行数据预览：")
print(df.head())

print("\n描述性统计：")
print(df.describe())
