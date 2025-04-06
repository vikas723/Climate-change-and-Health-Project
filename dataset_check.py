import pandas as pd

df = pd.read_csv("climate_fever_dataset2.csv")
print(df.head())  # Show first few rows
print(df.columns)  # Show available column names
print(df["tweet"].isna().sum())  # Count missing values
print(df["tweet"].str.strip().eq("").sum())  # Count empty values
