from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import base64

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Image processing libraries
import cv2

# Flask utilities
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Keras utilities
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
# from keras.utils import load_img, img_to_array
# from keras.preprocessing import image
from keras import layers, ops
import keras
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

app = Flask(__name__)

# Define paths to the models and load them
MODEL_PATHS = {
    'resnet50': 'models/best_EDGAN_model.keras',
    'resnet101': 'models/resnet_model.keras',
    'caries_GAN': 'models/caries_gan.h5',
    'caries_encoder': 'models/caries_encoder.h5',
    'caries_decoder': 'models/caries_decoder.h5',
    'gingivitis_GAN': 'models/gingivitis_gan.h5',
    'gingivitis_encoder': 'models/gingivitis_encoder.h5',
    'gingivitis_decoder': 'models/gingivitis_decoder.h5',
    'wsl_GAN': 'models/wsl_gan.h5',
    'wsl_encoder': 'models/wsl_encoder.h5',
    'wsl_decoder': 'models/wsl_decoder.h5',
}

custom_objects = {'Sampling': Sampling}

def load_model_on_demand(model_key):
    return load_model(MODEL_PATHS[model_key], custom_objects=custom_objects, compile=False)

# Load models
models = {
    'resnet50': load_model(MODEL_PATHS['resnet50']),
}

# for model in models.values():
#     model.make_predict_function()

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/literature_review')
def literature_review():
    return render_template('literature_review.html')

@app.route('/classification/resnet50')
def classification_resnet50():
    return render_template('resnet50.html')

@app.route('/classification/resnet101')
def classification_resnet101():
    return render_template('resnet101.html')

@app.route('/model/architecture')
def model_architecture():
    return render_template('architecture.html')

@app.route('/model/history')
def model_history():
    return render_template('history.html')

@app.route('/model/generate')
def model_generate():
    return render_template('generate.html')

@app.route('/predict_resnet50', methods=['POST'])
def predict_resnet50():
    return predict('resnet50')

@app.route('/predict_resnet101', methods=['POST'])
def predict_resnet101():
    return predict('resnet101')


def generate_gan_image(encoder, decoder, generator):
    # This function generates images using the GAN models
    noise = tf.random.normal((6, 1024))
    noise = decoder.predict(noise)
    noise = encoder.predict(noise)[2]
    noise = tf.convert_to_tensor(noise)
    generated_images = generator.predict(noise)

    images = []
    for image in generated_images:
        image = ((image + 1) * 127.5).astype(np.uint8)  # Normalize image to 0-255
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        images.append(image_rgb)

    return images


@app.route('/generate_images/<image_type>')
def generate_images(image_type):
    if image_type not in ['caries', 'gingivitis', 'wsl']:
        return jsonify({"error": "Invalid image type"}), 400

    encoder = load_model_on_demand(f'{image_type}_encoder')
    decoder = load_model_on_demand(f'{image_type}_decoder')
    generator = load_model_on_demand(f'{image_type}_GAN')

    images = generate_gan_image(encoder, decoder, generator)
    images_base64 = []
    keras.backend.clear_session()
    for image in images:
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('ascii')
        img_url = f"data:image/png;base64,{img_base64}"
        images_base64.append(img_url)

    return jsonify(images_base64)

def predict(model_name):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Preprocess the image and make predictions
        img = cv2.imread(filepath)  # Change target size according to your model
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (128,128))
        x = x / 255.0  # Scale the image pixels if required by your model
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = models[model_name].predict(x)
        result = np.argmax(preds, axis=1)  # Example to get the predicted class index
        classes = {0: 'Caries', 1: 'Gingivitis', 2:'White Spot Lesion'}
        return jsonify({'prediction': str(classes[result[0]])})

    return jsonify({'error': 'Allowed file types are png, jpg, jpeg'})

if __name__ == '__main__':
    app.run(debug=True)
