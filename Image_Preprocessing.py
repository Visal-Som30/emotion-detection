import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt
import dlib

class ImagePreprocessor:
    predictor_path  = 'shape_predictor_68_face_landmarks.dat'
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def facial_detection(self,img_path):
        # Load an image
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector(gray)

        # Iterate over detected faces
        for face in faces:
            # Predict facial landmarks
            landmarks = self.predictor(gray, face)

            # Extract face region using the bounding box of the detected face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = img[y:y+h, x:x+w]

            # Draw landmarks on the face image (optional)
            for point in landmarks.parts():
                cv2.circle(face_img, (point.x - x, point.y - y), 2, (0, 255, 0), -1)
            return face_img
        