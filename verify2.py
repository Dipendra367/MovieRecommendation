import pandas as pd

# Define dataset path
dataset_path = r"C:\Users\HP\.cache\kagglehub\datasets\parasharmanas\movie-recommendation-system\versions\1"

# Load CSV files
movies = pd.read_csv(f"{dataset_path}/movies.csv")
ratings = pd.read_csv(f"{dataset_path}/ratings.csv")

# Display first few rows
print("Movies Dataset:")
print(movies.head())

print("\nRatings Dataset:")
print(ratings.head())