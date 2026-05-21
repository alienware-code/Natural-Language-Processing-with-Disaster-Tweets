import pandas as pd       #щоб працювати із базою даних
import re           #щоб обробляти твіти
from sklearn.feature_extraction.text import TfidfVectorizer    #щоб перекласти текст в числа і вираховує вагу слів
from sklearn.linear_model import LogisticRegression     #це алгоритм навчання
import pickle          #зберагіє результати навчання
from text_utils import clean_tweet

print("1. Завантаження даних...")
train_data = pd.read_csv("train.csv")   #завантажуємо дані із бази

print("2. Балансування даних...")
disaster_tweets = train_data[train_data['target'] == 1]    #розподіляємо твіти по двох класах
normal_tweets = train_data[train_data['target'] == 0]
difference = len(normal_tweets) - len(disaster_tweets)    #рахуємо різницю в кількості твітів про катастрофи та звичайних
synthetic_data = disaster_tweets.sample(n=difference, replace=True, random_state=42)    #генеруємо синтетичні дані
balanced_data = pd.concat([train_data, synthetic_data])  #додаємо синтетичні дані до загальної бази
print(f"Дані збалансовано! Всього записів: {len(balanced_data)}")


cleaned_texts = []
for tweet in balanced_data['text']:       #очищуємо твіти
    cleaned_texts.append(clean_tweet(tweet))

balanced_data['clean_text'] = cleaned_texts

print("4. Векторизація тексту...")
vectorizer = TfidfVectorizer(max_features=3000)     #перетворюємо твіти в числові вектори зі зваженими словами, але берем тільки 3000 найпопулярніших
X = vectorizer.fit_transform(balanced_data['clean_text'])       #матриця із числами, в які перетворились твіти
y = balanced_data['target']      #матриця значень 0 і 1 (не катастрофа і катастрофа)

print("5. Навчання Логістичної Регресії...")
model = LogisticRegression()
model.fit(X, y)

print("6. Збереження моделей...")
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Готово.")

