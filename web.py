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
st.title('🤖 Body Language Detection')
st.markdown("This app uses **MediaPipe** and a custom trained model to predict body language.")

# Mediapipe setup
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
