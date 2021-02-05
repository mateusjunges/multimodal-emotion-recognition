from __future__ import division
import cv2
import numpy as np
from scipy.ndimage import zoom
from keras.models import model_from_json


class FaceEmotionRecognition:
    def __init__(self):
        self._emotions = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }

    def get_emotion_name(self, key):
        return self._emotions.get(key, "Emoção não encontrada")

    def face_detector(self, frame, shape_x=48, shape_y=48):
        cascade_classifier_path = "models/cascade-classifier.xml"
        face_cascade = cv2.CascadeClassifier(cascade_classifier_path)
        bgr2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(bgr2gray, scaleFactor=1.1, minNeighbors=6,
                                                       minSize=(shape_x, shape_y), flags=cv2.CASCADE_SCALE_IMAGE)

        coordinate = []  # Armazena as regiões em que a face foi detectada
        for x, y, w, h in detected_faces:
            if w > 100:
                sub_image = frame[y:y + h, x:x + w]
                coordinate.append([x, y, w, h])

        return bgr2gray, detected_faces, coordinate

    def extract_features_from_face(self, faces, offset_coefficients=(0.075, 0.05), shape_x=48, shape_y=48):
        gray = faces[0]
        detected_face = faces[1]

        new_face = []

        for det in detected_face:
            x, y, w, h = det  # Região em que a face foi detectada
            h_offset = np.int(np.floor(offset_coefficients[0] * w))
            v_offset = np.int(np.floor(offset_coefficients[1] * h))

            face = gray[y + v_offset:y + h, x + h_offset:x - h_offset + w]
            face = zoom(face, (shape_x / face.shape[0], shape_y / face.shape[1]))
            face = face.astype(np.float32)
            face = face / float(face.max())

            new_face.append(face)

        return new_face