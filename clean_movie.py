import pandas as pd

# ✅ Load your existing `movies.csv`
file_path = r"C:\Users\HP\.cache\kagglehub\datasets\parasharmanas\movie-recommendation-system\versions\1\movies.csv"
movies = pd.read_csv(file_path)

# ✅ Check available columns
print("Available columns:", movies.columns)

# ✅ Select relevant columns (Modify column names if needed)
# Since "rating" is missing, we only use "movieId", "title", and "genres"
movies = movies[["movieId", "title", "genres"]]  # Adjust column names if needed

# ✅ Rename "genres" to "genre" for consistency
movies.rename(columns={"genres": "genre"}, inplace=True)

# ✅ Save the cleaned file
movies.to_csv("movies_cleaned.csv", index=False)

print("✅ movies_cleaned.csv created successfully!")