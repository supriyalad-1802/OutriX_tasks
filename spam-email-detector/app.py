import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("📧 Spam Email Detector")

email = st.text_area("Enter Email Text")

if st.button("Check"):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)
    st.success("Spam 🚨" if result[0] == 1 else "Ham ✅")
