import sys
import keras
print(keras.__version__)
import cv2

model = keras.models.load_model('models/best_EDGAN_model.keras')
