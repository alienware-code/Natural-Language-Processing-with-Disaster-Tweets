import streamlit as st
import pickle
import re
from text_utils import clean_tweet

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Аналізатор твітів про катастрофи 🚨")
st.write("Введіть текст повідомлення англійською мовою, щоб перевірити його.")

user_input = st.text_area("Текст твіту:")

if st.button("Аналізувати"):
    if len(user_input.strip()) == 0:
        st.warning("Будь ласка, введіть текст для перевірки!")
    else:
        cleaned = clean_tweet(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.error("⚠️ Це повідомлення про РЕАЛЬНУ катастрофу!")
        else:
            st.success("✅ Це звичайне повідомлення (не катастрофа).")