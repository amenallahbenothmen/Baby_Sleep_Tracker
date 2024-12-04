from PIL import Image
import mediapipe as mp
import cv2
import os
import numpy as np
import csv

def image_landmark_extractor(data_path, output_csv, state="Asleep"):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for i, image_path in enumerate(os.listdir(data_path)):
            image = np.array(Image.open(os.path.join(data_path, image_path)))
            image.setflags(write=False)
            
            results = holistic.process(image)

            image.setflags(write=True)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            #Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            #Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            #Pose 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Capture the landmarks and write them into the csv labeled
            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in face]).flatten())

                row = pose_row + face_row
                row.insert(0, state)
                with open(output_csv, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except:
                print(image_path, "WAS NOT TREATED, probably no keypoints detected", i)
                pass
            cv2.imshow('live feed and detection Cam!', image)
            print(image_path, " ", i)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

def capture_landmark_extractor(data_path='/', output_csv='default.csv', source="Camera", state="Asleep"):
    if source == "video":
        cap = cv2.VideoCapture(data_path)
    elif source == "camera":
        cap = cv2.VideoCapture(0)

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.setflags(write=False)
            
            results = holistic.process(image)
            image.setflags(write=True)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                #pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in pose]).flatten())

                #face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[val.x, val.y, val.z, val.visibility] for val in face]).flatten())

                #write to the CSV
                row = pose_row + face_row
                row.insert(0, state)
                with open(output_csv, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except:
                pass
            cv2.imshow('live feed and detection Cam!', image)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
