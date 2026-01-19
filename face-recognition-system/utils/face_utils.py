import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Initialize models once
detector = MTCNN()
embedder = FaceNet()

def extract_face(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)

    if not faces:
        return None, None

    # Get coordinates
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    
    face = img[y:y+h, x:x+w]
    if face.size == 0: return None, None
    
    face_resized = cv2.resize(face, (160, 160))
    
    # Generate embedding
    embedding = embedder.embeddings([face_resized])[0]
    return embedding.reshape(1, -1), (x, y, w, h)

def recognize_face(face_embedding, known_data, threshold=0.70):
    embeddings, labels = known_data
    best_match = "Unknown"
    best_score = -1.0

    for i, saved_emb in enumerate(embeddings):
        score = cosine_similarity(face_embedding, saved_emb.reshape(1, -1))[0][0]
        if score > best_score:
            best_score = score
            best_match = labels[i]

    return best_match if best_score > threshold else "Unknown"