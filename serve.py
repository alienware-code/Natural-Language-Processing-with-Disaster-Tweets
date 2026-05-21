from flask import Flask, request, jsonify
import pickle
import re
from text_utils import clean_tweet
app = Flask(__name__)

with open('tfidf_vectorizer.pkl', 'rb') as f:   #завантажуєм словник
    vectorizer = pickle.load(f)
with open('logistic_model.pkl', 'rb') as f:      #завантажуєм модель
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():              #отримуємо дані від користувача
    data = request.get_json()
    if 'texts' in data:
        texts = data['texts']
        cleaned = [clean_tweet(t) for t in texts]
        vectorized = vectorizer.transform(cleaned)
        predictions = model.predict(vectorized).tolist() # масив прогнозів
        return jsonify({"batch_predictions": predictions})
    tweet_text = data['text']
    cleaned = clean_tweet(tweet_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = int(model.predict(vectorized)[0])     #робимо прогноз
    return jsonify({
        "original_text": tweet_text,
        "prediction_class": prediction,
        "is_disaster": True if prediction == 1 else False
    })

if __name__ == '__main__':
    print("Сервер запущено! Очікую запити...")
    app.run(port=5000)