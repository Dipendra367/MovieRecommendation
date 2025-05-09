import streamlit as st
import pandas as pd

# ✅ Load final movie dataset with ratings
file_path = r"C:\Users\HP\PycharmProjects\MovieRecommendation\movies_final.csv"
movies = pd.read_csv(file_path)

# 🎨 **Apply Custom Styling**
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# 🎬 **Header Section**
st.markdown(
    """
    <h1 style='text-align: center; color: #FF5733;'>🎬 Movie Recommendation System</h1>
    <h3 style='text-align: center; color: #C70039;'>Find the best movies based on your favorite genre!</h3>
    """,
    unsafe_allow_html=True
)

# 🎨 **Sidebar for Genre Selection**
st.sidebar.header("🎭 Choose Your Preferences")
all_genres = set()
for genre_list in movies["genres"]:
    genres = genre_list.split("|")  # Split genres
    all_genres.update(genres)

selected_genre = st.sidebar.selectbox("🎥 Select a Genre:", sorted(all_genres))

# 🎭 **Slider for Number of Recommendations**
num_recommendations = st.sidebar.slider("🎯 Number of Movies to Show:", 1, 10, 5)

# 🎬 **Recommendation Button**
if st.sidebar.button("🔍 Get Recommendations"):
    filtered_movies = movies[movies["genres"].str.contains(selected_genre)].sort_values(by="rating",
                                                                                        ascending=False).head(
        num_recommendations)

    # 🎬 **Display Recommended Movies**
    st.markdown(f"<h2 style='text-align: center; color: #FF5733;'>🎯 Recommended Movies</h2>", unsafe_allow_html=True)

    for index, row in filtered_movies.iterrows():
        st.markdown(
            f"""
            <div style="background-color:#F5F5F5; padding:10px; border-radius:10px; margin-bottom:10px;">
                <h3 style="color:#C70039;">🎬 {row['title']}</h3>
                <p style="color:#900C3F;">⭐ Rating: {row['rating']:.1f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# 🎨 **Footer**
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #555;'>Made with ❤️ by Dipendra</p>
    """,
    unsafe_allow_html=True
)