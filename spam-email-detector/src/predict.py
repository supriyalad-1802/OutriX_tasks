import pickle
from src.preprocess import clean_text

model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf.pkl", "rb"))

def predict_email(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Ham"
