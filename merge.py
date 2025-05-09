import pandas as pd

# ✅ Load datasets
movies_path = r"C:\Users\HP\.cache\kagglehub\datasets\parasharmanas\movie-recommendation-system\versions\1\movies.csv"
ratings_path = r"C:\Users\HP\.cache\kagglehub\datasets\parasharmanas\movie-recommendation-system\versions\1\ratings.csv"

movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)

# ✅ Calculate average rating per movie
average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

# ✅ Normalize ratings (scale between 0 and 1)
average_ratings["rating"] = (average_ratings["rating"] - 0.5) / (5.0 - 0.5)  # ✅ Scale between 0 and 1

# ✅ Merge datasets on "movieId"
movies_final = pd.merge(movies, average_ratings, on="movieId", how="left")

# ✅ Fill missing ratings with **median** instead of mean
movies_final["rating"] = movies_final["rating"].fillna(movies_final["rating"].median())

# ✅ Save the merged dataset
movies_final.to_csv("movies_final.csv", index=False)

print("✅ movies_final.csv created successfully with normalized ratings!")