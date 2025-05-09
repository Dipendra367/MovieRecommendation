import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
from sklearn.model_selection import train_test_split

# ✅ Load dataset
file_path = r"C:\Users\HP\PycharmProjects\MovieRecommendation\movies_final.csv"
movies = pd.read_csv(file_path)

# ✅ Convert user and movie IDs to integer indices
user_ids = movies["movieId"].unique()
movie_ids = movies["movieId"].unique()

user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

movies["user_index"] = movies["movieId"].map(user_to_index)
movies["movie_index"] = movies["movieId"].map(movie_to_index)

# ✅ Split data
train, test = train_test_split(movies, test_size=0.2, random_state=42)


# ✅ Convert data to TensorFlow dataset format
def create_tf_dataset(data):
    user_tensor = tf.convert_to_tensor(data["user_index"].values, dtype=tf.int32)
    movie_tensor = tf.convert_to_tensor(data["movie_index"].values, dtype=tf.int32)
    rating_tensor = tf.convert_to_tensor(data["rating"].values, dtype=tf.float32)  # ✅ Use normalized ratings

    dataset = tf.data.Dataset.from_tensor_slices(({"userId": user_tensor, "movieId": movie_tensor}, rating_tensor))
    return dataset.batch(512).cache().prefetch(tf.data.experimental.AUTOTUNE)


train_ds = create_tf_dataset(train)
test_ds = create_tf_dataset(test)


# ✅ Fix Model Training to Predict Correct Ratings
@tf.keras.utils.register_keras_serializable()
class MovieRecommendationModel(tfrs.Model):
    def __init__(self, num_users, num_movies, **kwargs):
        super().__init__(**kwargs)  # ✅ Pass kwargs to handle extra arguments like `trainable`

        self.num_users = num_users
        self.num_movies = num_movies

        self.user_embedding = tf.keras.layers.Embedding(num_users, 32)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, 32)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1)  # ✅ Remove sigmoid activation
        ])

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

    def call(self, inputs):
        user_vector = self.user_embedding(inputs["userId"])
        movie_vector = self.movie_embedding(inputs["movieId"])
        concat = tf.concat([user_vector, movie_vector], axis=1)
        return self.dense(concat)

    def compute_loss(self, features, training=False):
        labels = features[1]
        predictions = self(features[0])
        return self.task(labels=labels, predictions=predictions)

    # ✅ Fix: Ensure `get_config()` includes all arguments used in `__init__()`
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_movies": self.num_movies
        })
        return config

    # ✅ Fix: Ensure `from_config()` correctly reconstructs the model
    @classmethod
    def from_config(cls, config):
        return cls(num_users=config["num_users"], num_movies=config["num_movies"])


# ✅ Train model with more epochs (Increase from 5 → 20)
num_users = len(user_ids)
num_movies = len(movie_ids)
model = MovieRecommendationModel(num_users=num_users, num_movies=num_movies)
model.compile(optimizer="adam")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
model.fit(train_ds, epochs=20, callbacks=[early_stopping])

# ✅ Save model
model.save("models/tfrs_model_optimized.keras")
print("✅ Optimized model training completed!")