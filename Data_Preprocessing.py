from PIL import Image
import numpy as np
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import cv2
from matplotlib import pyplot as plt
import dlib
from skimage.feature import hog

class ImagePreprocessing:
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    def __init__(self):
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.detector = dlib.get_frontal_face_detector()
    def preprocess_labels(labels):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        one_hot_labels = to_categorical(encoded_labels)
        return one_hot_labels
    def load_image(image_path):
        img = Image.open(image_path)
        return img

    def resize_image(img, target_size=(224, 224)):
        resized_img = img.resize(target_size)
        return resized_img

    def normalize_image(img_array):
        normalized_img = img_array / 255.0  # Rescale pixel values to [0, 1]
        return normalized_img

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


    def preprocess_image(self,image_path, emotion_label, target_size=(224, 224)):
        img = self.load_image(image_path)
        resized_img = self.resize_image(img, target_size)
        img_array = np.array(resized_img)
        normalized_img = self.normalize_image(img_array)
        augmented_img = self.datagen.flow(np.expand_dims(normalized_img, axis=0), batch_size=1)[0][0]
        label = self.preprocess_labels([emotion_label])[0]
        return augmented_img, label

    def facial_detection_img(self,img):
        # Load an image
        # try catch
        try:
             
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = self.detector(gray)
            # Iterate over detected faces
            for face in faces:
                # Predict facial landmarks
                landmarks = self.predictor(gray, face)

                # Extract face region using the bounding box of the detected face
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_img = gray[y:y+h, x:x+w]#img

                # Draw landmarks on the face image (optional)
                for point in landmarks.parts():
                    cv2.circle(face_img, (point.x - x, point.y - y), 2, (0, 255, 0), -1)
                if face_img is None:
                    return None
                # Resize the face image to 224x224
                _img = cv2.resize(face_img, (224, 224))

                # select point on face
                _img_ = []
                for x in _img:
                    arr = []
                    for y in x:
                        if y>5:
                            arr.append([255,255,255])
                        else:
                            arr.append([0,0,0])
                    _img_.append(arr)
                return _img_
        except:
            return None