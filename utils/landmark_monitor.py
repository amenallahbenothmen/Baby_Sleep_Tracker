from PIL import Image
import mediapipe as mp
import cv2
import os
import pandas as pd
import numpy as np
import csv


def image_landmark_monitor(data_path, model):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i, image_path in enumerate(os.listdir(data_path)):
            image = np.array(Image.open(os.path.join(data_path, image_path)))
            image.setflags(write=False)
            
            results = holistic.process(image)

            image.setflags(write=True)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in face]).flatten())

                row = pose_row + face_row
                X = pd.DataFrame([row])
                body_language_prob = model.predict(X)[0]
                body_language_class = "Asleep" if body_language_prob <= 0.5 else "Awake"
                body_language_prob = body_language_prob if body_language_class == "Awake" else abs(1 - body_language_prob)
                print(body_language_class, body_language_prob)

                coords = tuple(np.multiply(
                    np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                              results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), 
                    [640, 480]).astype(int))
                
                cv2.rectangle(image, (coords[0], coords[1] + 5), 
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30), 
                              (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                print(image_path, "WAS NOT TREATED, probably no keypoints detected", i)
                pass
            cv2.imshow('live feed and detection Cam!', image)
            print(image_path, "  ", i)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def capture_landmark_monitor(data_path, model, source="camera"):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    if source == "video":
        cap = cv2.VideoCapture(data_path)
    elif source == "camera":
        cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.setflags(write=False)
            
            results = holistic.process(image)

            image.setflags(write=True)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in face]).flatten())

                row = pose_row + face_row
                X = pd.DataFrame([row])
                body_language_prob = model.predict(X)[0]
                body_language_class = "Asleep" if body_language_prob <= 0.5 else "Awake"
                body_language_prob = body_language_prob if body_language_class == "Awake" else abs(1 - body_language_prob)
                print(body_language_class, body_language_prob)

                # Visualize
                coords = tuple(np.multiply(
                    np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                              results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), 
                    [640, 480]).astype(int))
                
                cv2.rectangle(image, (coords[0], coords[1] + 5), 
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30), 
                              (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob, 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            except:
                pass
                            
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
