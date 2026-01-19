import streamlit as st
import cv2
import pickle
import time
from utils.face_utils import extract_face, recognize_face

st.set_page_config(page_title="Face Recognition Pro", layout="centered")

# Custom CSS for a better UI
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .stHeader { text-align: center; color: #0E1117; }
    </style>
    """, unsafe_allow_html=True)

st.title("📸 Real-Time Face Recognition")
st.write("System Status: **Ready**")

@st.cache_resource
def load_data():
    with open("face_embeddings.pkl", "rb") as f:
        return pickle.load(f)

known_embeddings = load_data()

# Sidebar Info
st.sidebar.title("Settings")
threshold = st.sidebar.slider("Recognition Threshold", 0.0, 1.0, 0.70)
st.sidebar.info("Processing every 5th frame for smoothness.")

# Start/Stop Logic
if 'run' not in st.session_state:
    st.session_state.run = False

col1, col2 = st.columns(2)
if col1.button("▶ Start Camera"):
    st.session_state.run = True
if col2.button("🛑 Stop Camera"):
    st.session_state.run = False
    st.rerun()

FRAME_WINDOW = st.image([])

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    # Optimization: Set camera resolution lower for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    last_name = ""
    last_box = None

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        # FRAME SKIPPING: Only run recognition every 5 frames
        if frame_count % 5 == 0:
            face_embedding, box = extract_face(frame)
            if face_embedding is not None:
                last_name = recognize_face(face_embedding, known_embeddings, threshold)
                last_box = box
            else:
                last_name = ""
                last_box = None

        # Draw UI Elements on frame
        if last_box:
            x, y, w, h = last_box
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Friendly Message
            msg = f"Hi {last_name}" if last_name != "Unknown" else "Identifying..."
            cv2.putText(frame, msg, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        
        frame_count += 1
        
    cap.release()
else:
    st.info("Camera is currently OFF. Click 'Start' to begin.")