import pandas as pd
import pickle
import re
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from text_utils import clean_tweet
from sklearn.metrics import classification_report, confusion_matrix

print("Запуск перевірки метрик моделі...")
print("Завантаження файлів .pkl...")
with open('tfidf_vectorizer.pkl', 'rb') as f:   #завантажуємо словник
    vectorizer = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:  #завантажуємо модель
    model = pickle.load(f)

print("Підготовка тестових даних...")
data = pd.read_csv("train.csv")   #завантажуємо дані для тестування

cleaned_texts = []
for tweet in data['text']:
    cleaned_texts.append(clean_tweet(tweet))       #очищуємо дані для тестування
    
data['clean_text'] = cleaned_texts

X_train, X_test, y_train, y_test = train_test_split(     #відділяємо 20% даних суто для перевірки 
    data['clean_text'], data['target'], test_size=0.2, random_state=42
)

print("Обчислення прогнозів...")
X_test_vectorized = vectorizer.transform(X_test)     #перетворюємо текст на числа 
predictions = model.predict(X_test_vectorized)         #робимо прогноз

print("\nМЕТРИКИ:")
print("Загальна точність (Accuracy):", round(accuracy_score(y_test, predictions), 3))
print("Метрика F1 Score:", round(f1_score(y_test, predictions), 3))
print("Малювання Матриці Помилок (Confusion Matrix)...")
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    predictions, 
    display_labels=["Not Disaster", "Disaster"], 
    cmap=plt.cm.Blues
)

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))
plt.title("Confusion Matrix - Evaluation")
plt.show()
