import pandas as pd
import numpy as np

# Define dataset path
dataset_path = r"C:\Users\HP\.cache\kagglehub\datasets\parasharmanas\movie-recommendation-system\versions\1"

# Load datasets
movies = pd.read_csv(f"{dataset_path}/movies.csv")
ratings = pd.read_csv(f"{dataset_path}/ratings.csv")

# Handle missing values
movies.fillna("", inplace=True)
ratings.dropna(inplace=True)

# Extract year from movie title
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

# Normalize ratings
ratings["rating"] = ratings["rating"] / ratings["rating"].max()

# Save processed data
movies.to_csv(f"{dataset_path}/movies_processed.csv", index=False)
ratings.to_csv(f"{dataset_path}/ratings_processed.csv", index=False)

print("✅ Data preprocessing completed!")