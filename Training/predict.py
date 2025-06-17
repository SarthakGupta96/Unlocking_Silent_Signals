


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open('Model/model.pkl', 'rb') as f:
    model = pickle.load(f)




# Initialize mediapipe modules
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Open webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        ## FACE
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

        ## RIGHT HAND
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )

        ## LEFT HAND
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )

        ## POSE
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Prediction logic
        try:
    # Extract landmarks
            pose = results.pose_landmarks.landmark if results.pose_landmarks else []
            face = results.face_landmarks.landmark if results.face_landmarks else []

            if len(pose) == 33 and len(face) == 468:
            # Convert landmarks to flat row
                pose_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]).flatten()
                face_row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in face]).flatten()

            # Combine all landmarks
                row = np.concatenate([pose_row, face_row])

            # Create feature names to match training
                feature_names = [f'{name}_{i}_{axis}' for name, count in [('pose', 33), ('face', 468)]
                             for i in range(count) for axis in ['x', 'y', 'z', 'v']]

            # Create DataFrame with correct column names
                X = pd.DataFrame([row], columns=feature_names)

            # Predict with model
                prediction_class = model.predict(X)[0]
                

            # Display results
                print(f'Prediction: {prediction_class}')

        
            # Optional: Overlay prediction on the image
                cv2.putText(image, f'{prediction_class}',
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")

        cv2.imshow('Raw webcam feed', image)

        if cv2.waitKey(10) & 0xFF == ord('w'):
            break

cap.release()
cv2.destroyAllWindows()


