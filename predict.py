import requests
import json
import pandas as pd

url = 'http://localhost:5000/predict'     #лінкс нашого сервісу
df = pd.read_csv("train.csv")
sample_data = df.sample(5, random_state=42)['text'].tolist()

data_to_send = {
    "texts": sample_data
}

print("Відправляємо запит до сервісу...")
try:
    response = requests.post(url, json=data_to_send)
    print("Отримана відповідь (Масив предіктів):")
    print(json.dumps(response.json(), indent=4))
    
    print("\nДетальний розбір:")
    predictions = response.json()['batch_predictions']
    for text, pred in zip(sample_data, predictions):
        status = "🚨 Катастрофа" if pred == 1 else "✅ Звичайний"
        print(f"[{status}] -> {text[:80]}...")
except Exception as e:
    print("Помилка! Перевірте, чи файл serve.py зараз запущений в іншому терміналі.")