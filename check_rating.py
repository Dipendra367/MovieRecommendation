import pandas as pd

# ✅ Load final movie dataset
file_path = r"C:\Users\HP\PycharmProjects\MovieRecommendation\movies_final.csv"
movies = pd.read_csv(file_path)

# ✅ Check rating distribution
print("Rating Statistics:")
print(movies["rating"].describe())

# ✅ Show sample ratings
print("Sample Ratings:")
print(movies[["title", "rating"]].head(10))