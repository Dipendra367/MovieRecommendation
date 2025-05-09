import tensorflow as tf
from model_training import MovieRecommendationModel

# ✅ Load trained model
model_path = r"C:\Users\HP\PycharmProjects\MovieRecommendation\models\tfrs_model_optimized.keras"
model = tf.keras.models.load_model(model_path, custom_objects={"MovieRecommendationModel": MovieRecommendationModel})

# ✅ Test predictions for random users & movies
test_cases = [
    {"userId": 5, "movieId": 20},
    {"userId": 10, "movieId": 50},
    {"userId": 15, "movieId": 100}
]

for case in test_cases:
    sample_user = tf.convert_to_tensor([case["userId"]])
    sample_movie = tf.convert_to_tensor([case["movieId"]])
    prediction = model.predict({"userId": sample_user, "movieId": sample_movie})

    # ✅ Convert predictions back to original scale (0.5 - 5.0)
    scaled_prediction = prediction[0][0] * (5.0 - 0.5) + 0.5

    print(f"🎯 Predicted rating for User {case['userId']} & Movie {case['movieId']}: {scaled_prediction:.2f}")