import pandas as pd

df = pd.read_csv("climate_fever_dataset2.csv")

# Check for missing values
print("Missing tweets:", df["tweet"].isna().sum())

# Check for empty tweets
print("Empty tweets:", df["tweet"].str.strip().eq("").sum())

# Print first few tweets
print(df["tweet"].head())
