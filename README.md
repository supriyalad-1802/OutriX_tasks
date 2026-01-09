🚀 Outrix Tasks
This repository contains projects completed as part of Outrix tasks, focused on applying Machine Learning and NLP concepts to real-world problems.

✅ Completed Task
📧 Spam Email Detection Using Machine Learning

This project implements a Spam Email Detection System using Natural Language Processing (NLP) and Machine Learning. It classifies emails as Spam or Ham (Not Spam) and is deployed using a Streamlit web application.

🎯 Project Goal

1. Preprocess email text using NLP techniques
2. Extract features using TF-IDF
3. Train a Support Vector Machine (SVM) classifier
4. Predict whether an email is Spam or Ham
5. Deploy the model with an interactive web interface

🛠️ Tools & Technologies

Language: Python
Libraries: Pandas, NumPy, NLTK, Scikit-learn
ML Algorithm: SVM (LinearSVC)
Feature Extraction: TF-IDF Vectorizer
UI: Streamlit
IDE: Google Colab, VS Code

- Dataset

CSV file containing labeled messages
v1 → Label (spam / ham)
v2 → Email text
Public SMS Spam Collection Dataset

How to Run the Project
Step 1: Clone Repository
git clone https://github.com/your-username/spam-email-detector.git
cd spam-email-detector

Step 2: Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

Step 3: Install Dependencies
python -m pip install -r requirements.txt

Step 4: Download NLTK Resources
python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
>>> exit()

Step 5: Run Streamlit App
python -m streamlit run src/app.py


Open in browser:

http://localhost:8501

📌 Key Learnings

1. Practical NLP preprocessing
2. TF-IDF feature extraction
3. ML model training & saving
4. Model deployment using Streamlit
5. Handling real-world environment issues

👩‍💻 Author
Supriya Lad
Final Year Computer Science & Engineering Student

📄 License
This project is created for educational and learning purposes.

📌 Note
More Outrix tasks will be added to this repository.