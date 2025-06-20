import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle
import os

import mediapipe as mp

# Load model
@st.cache_data
def load_model():
    with open('Model/model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Streamlit UI


st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Home", "About Project", "Contact Us"])

if section == "About Project":
    st.title("About Project")
    st.markdown("""👋 About Us
At Unlocking Silent Signals, we are passionate about harnessing the power of computer vision and machine learning to bridge the gap between unspoken human gestures and actionable insights. Our project aims to decode body language and expressions in real-time using state-of-the-art MediaPipe technology, making human-computer interaction more natural, inclusive, and intelligent.

Through this initiative, we strive to:

🔍 Understand Human Movement: By extracting and analyzing landmarks from body, face, and hand positions, we reveal hidden patterns and expressions.

💡 Enable Smart Interaction: Helping developers build applications that respond intuitively to gestures and emotions.

🌍 Make Technology More Human-Centered: Focusing on accessibility, inclusivity, and emotional understanding.

With a seamless blend of innovative algorithms, deep learning, and interactive design, we aim to create tools that unlock the silent signals people send every day — making digital spaces more connected and meaningful.""")
elif section == "Contact Us":
    st.title("Contact Us")
    st.markdown("""
## 📞 Contact Us

For any questions, collaboration, or support, feel free to get in touch:

- 👤 **Sarthak Gupta**  
  📱 **Phone**: +91-84486-05613  
  📧 **Email**: sarthak.gupta2023@vitstudent.ac.in 

- 👤 **Nachiketa Venkat**  
  📱 **Phone**: +91-98188-13710  
  📧 **Email**: nachiketa.venkat2023@vitstudent.ac.in  
""")
else:
    st.title("🤖 Body Language Detection")
    st.markdown("This app uses **MediaPipe** and a custom trained model to predict body language.")
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # Streamlit components
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    # Main logic
    if run:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while camera.isOpened():
                ret, frame = camera.read()
                if not ret:
                    continue

                # Recolor for Mediapipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                # Recolor back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

                try:
                    # Extract pose and face
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten())

                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten())

                    row = pose_row + face_row
                    X = pd.DataFrame([row], columns=model.feature_names_in_)


                    # Make prediction
                    body_language_class = model.predict(X)[0]
                    if hasattr(model, "predict_proba"):
                        body_language_prob = model.predict_proba(X)[0]
                        confidence = round(np.max(body_language_prob), 2)
                    else:
                        body_language_prob = None
                        confidence = "N/A"


                    # Ear coordinates for placing label
                    coords = tuple(np.multiply(
                        np.array([
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y
                        ]),
                        [640, 480]
                    ).astype(int))

                    # Draw rectangle and class name
                    cv2.rectangle(image,
                                (coords[0], coords[1] + 5),
                                (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                                (245, 117, 16), -1)
                    cv2.putText(image, body_language_class, coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Top status box
                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0],
                                (90, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    print("Error during prediction:", e)

                FRAME_WINDOW.image(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        camera.release()
        cv2.destroyAllWindows()


