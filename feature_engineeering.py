import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load processed data
dataset_path = r"C:\Users\HP\.cache\kagglehub\datasets\parasharmanas\movie-recommendation-system\versions\1"
movies = pd.read_csv(f"{dataset_path}/movies_processed.csv")
ratings = pd.read_csv(f"{dataset_path}/ratings_processed.csv")

# One-hot encode genres
encoder = OneHotEncoder(handle_unknown="ignore")
genres_encoded = encoder.fit_transform(movies["genres"].explode().values.reshape(-1, 1)).toarray()
genres_df = pd.DataFrame(genres_encoded, columns=encoder.categories_[0])

# Merge encoded genres back into movies dataset
movies = movies.join(genres_df)

# Compute user preferences
user_preferences = ratings.groupby("userId")["rating"].mean().reset_index()

# Compute movie popularity score
movies["popularity_score"] = ratings.groupby("movieId")["rating"].mean()

# Save feature-engineered data
movies.to_csv(f"{dataset_path}/movies_features.csv", index=False)

print("✅ Feature engineering completed!")