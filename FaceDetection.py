from Data_Preprocessing import ImagePreprocessing
from keras.models import load_model
import numpy as np

class FaceDetection:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self._shared_state:
            self.model = load_model('model.h5')
            self.image_preprocessing = ImagePreprocessing()

    def predict(self,image):
        img = self.image_preprocessing.facial_detection_img(image)
        if img is None:
            return "Image can't detect"
        emotion = self.model.predict([img])
        result = np.array(emotion)
        maxidx = np.argmax(result)
        label = ['Angry', 'Happy', 'Nuetral'] 
        return label[maxidx]