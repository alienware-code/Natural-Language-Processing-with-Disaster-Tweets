import tkinter as tk
from tkinter import messagebox
import pickle
import re
from text_utils import clean_tweet

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)


def analyze_text():
    user_input = text_box.get("1.0", tk.END).strip()
    
    if len(user_input) == 0:
        messagebox.showwarning("Увага", "Будь ласка, введіть текст!")
    else:
        cleaned = clean_tweet(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = int(model.predict(vectorized)[0])
        
        if prediction == 1:
            messagebox.showerror("Результат", "⚠️ Це повідомлення про РЕАЛЬНУ катастрофу!")
        else:
            messagebox.showinfo("Результат", "✅ Це звичайне повідомлення (не катастрофа).")


window = tk.Tk()
window.title("Аналізатор твітів (Desktop Version)")
window.geometry("450x250")

label = tk.Label(window, text="Введіть текст твіту англійською:")
label.pack(pady=10)

text_box = tk.Text(window, height=5, width=50)
text_box.pack(pady=5)

analyze_button = tk.Button(window, text="Аналізувати", command=analyze_text, bg="lightblue", font=("Arial", 10, "bold"))
analyze_button.pack(pady=10)

window.mainloop()