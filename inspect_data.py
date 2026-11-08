
import pandas as pd

# Load the dataset using the correct delimiter
df = pd.read_csv("data.csv", sep=";")

# Clean column names
df.columns = df.columns.str.strip()

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["Target"].value_counts())

print("\nTarget Distribution (%):")
print(df["Target"].value_counts(normalize=True) * 100)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDataset Preview:")
print(df.head())
