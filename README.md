Outrix Tasks
This repository contains projects completed as part of the Outrix Internship, focused on applying Machine Learning, Computer Vision, and NLP concepts to real-world problems.

- Completed Tasks
Task 1: Spam Email Detection Using Machine Learning

This project implements a Spam Email Detection System using Natural Language Processing (NLP) and Machine Learning.
It classifies emails as Spam or Ham (Not Spam) and is deployed using a Streamlit web application.

🎯 Project Goal
1. Preprocess email text using NLP techniques
2. Extract features using TF-IDF Vectorization
3. Train a Support Vector Machine (SVM) classifier
4. Predict whether an email is Spam or Ham
5. Deploy the trained model using a Streamlit UI

🛠️ Tools & Technologies

Language: Python
Libraries: Pandas, NumPy, NLTK, Scikit-learn
ML Algorithm: SVM (LinearSVC)
Feature Extraction: TF-IDF
UI: Streamlit
IDE: Google Colab, VS Code

📊 Dataset

Public SMS Spam Collection Dataset

CSV file with:
v1 → Label (spam / ham)
v2 → Email text

Task 2: Face Recognition System (Real-Time)

This project is a real-time face recognition system that detects and identifies faces from webcam input using computer vision and deep learning embeddings.

🎯 Project Goal

1. Detect faces from live webcam feed
2. Extract facial embeddings using a pre-trained deep learning model
3. Compare faces using similarity metrics
4. Display recognized person’s name in real-time
5. Build a lightweight and efficient recognition pipeline

🛠️ Tools & Technologies

Language: Python
Libraries: OpenCV, NumPy, scikit-learn
Face Detection: MTCNN / Haar Cascade
Face Recognition: FaceNet embeddings (pre-trained)
Similarity Metric: Cosine Similarity
UI: Streamlit
IDE: Google Colab, VS Code

📌 Key Highlights

1. Face embeddings generated once and stored as .pkl file
2. Lightweight runtime (no heavy model loading in VS Code)
3. Real-time recognition with webcam

▶️ How to Run Projects (Common Steps)
Step 1: Clone Repository
git clone https://github.com/your-username/outrix-tasks.git
cd outrix-tasks

Step 2: Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

Step 3: Install Dependencies
python -m pip install -r requirements.txt

Step 4: Run Streamlit App
python -m streamlit run app.py


Open in browser:

http://localhost:8501

📌 Key Learnings from Internship

1. Practical NLP preprocessing & feature engineering
2. Training and deploying ML models
3. Face detection & recognition using embeddings
4. Real-time webcam integration
5. Streamlit app development
6. Managing models between Colab & VS Code

Debugging real-world environment issues

👩‍💻 Author
Final Year Computer Science & Engineering Student

Supriya Lad
Final Year Computer Science & Engineering Student

📄 License

This repository is created for educational and learning purposes as part of the Outrix Internship.
