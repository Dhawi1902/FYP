import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras import layers, Sequential, datasets, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.src.saving

import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
import cv2
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

from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
set_session(sess)

from common_function import *
from VAE import *
from EDGAN import *

num_images_to_generate = 20

# Task 1: Import images with labels and print in a tabular format
data_folder = "/home/dhawi/Documents/dataset"
dataset = data_folder + "/AI_project"
model_folder = "/home/dhawi/Documents/model"
history_folder = "/home/dhawi/Documents/History"

caries, caries_gray = load_images_from_folder(dataset, "Caries")

gingivitis, gingivitis_gray = load_images_from_folder(dataset, "Gingivitis")
wsl, wsl_gray = load_images_from_folder(dataset, "White Spot Lesion")

caries_VAE_history, caries_VAE = train_vae(caries)
sess.close()
tf.compat.v1.reset_default_graph()

gingivitis_VAE_history, gingivitis_VAE = train_vae(gingivitis)
sess.close()
tf.compat.v1.reset_default_graph()

wsl_VAE_history, wsl_VAE = train_vae(wsl)
sess.close()
tf.compat.v1.reset_default_graph()

caries_edgan = build_model("Caries", caries_VAE.encoder, caries_VAE.decoder)
caries_edgan_history = caries_edgan.fit(caries, epochs = 1000, batch_size = 50)
sess.close()
tf.compat.v1.reset_default_graph()

gingivitis_edgan = build_model("Gingivitis", gingivitis_VAE.encoder, gingivitis_VAE.decoder)
wsl_edgan = build_model("Wsl", wsl_VAE.encoder, wsl_VAE.decoder)


gingivitis_edgan_history = gingivitis_edgan.fit(gingivitis, epochs = 1000, batch_size = 50)
wsl_edgan_history = wsl_edgan.fit(wsl, epochs = 1000, batch_size = 50)


generated_caries_images = generate_images(caries_edgan.generator, caries_VAE.encoder, caries_VAE.decoder, num_images_to_generate)
generated_gingivitis_images = generate_images(gingivitis_edgan.generator, gingivitis_VAE.encoder, gingivitis_VAE.decoder, num_images_to_generate)
generated_wsl_images = generate_images(wsl_edgan.generator, wsl_VAE.encoder, wsl_VAE.decoder, num_images_to_generate)


# Load the trained ResNet model
trained_resnet_model = tf.keras.models.load_model('/home/dhawi/FYP/models/keras_best_model.keras')

def classify_images(model, images):
    predictions = model.predict(images)
    confidences = np.max(predictions, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels, confidences

confidence_threshold = 0.60  # Example threshold, adjust based on your needs

def filter_images(images, labels, confidences, true_label, threshold):
    mask = (labels == true_label) & (confidences >= threshold)
    return images[mask]


# Classify the generated images
caries_labels, caries_confidences = classify_images(trained_resnet_model, generated_caries_images)
gingivitis_labels, gingivitis_confidences = classify_images(trained_resnet_model, generated_gingivitis_images)
wsl_labels, wsl_confidences = classify_images(trained_resnet_model, generated_wsl_images)

filtered_caries_images = filter_images(generated_caries_images, caries_labels, caries_confidences, 0, confidence_threshold)
filtered_gingivitis_images = filter_images(generated_gingivitis_images, gingivitis_labels, gingivitis_confidences, 1, confidence_threshold)
filtered_wsl_images = filter_images(generated_wsl_images, wsl_labels, wsl_confidences, 2, confidence_threshold)


original_images, original_labels, class_names = load_original_images_and_labels(dataset)

# Assuming class indices are:
# caries: 0, gingivitis: 1, wsl: 2

# Create labels for generated images
generated_caries_labels = np.array([0 for _ in range(num_images_to_generate)])
generated_gingivitis_labels = np.array([1 for _ in range(num_images_to_generate)])
generated_wsl_labels = np.array([2 for _ in range(num_images_to_generate)])

# Combine original and generated images and labels
augmented_images = np.concatenate([original_images, generated_caries_images, generated_gingivitis_images, generated_wsl_images], axis=0)
augmented_labels = np.concatenate([original_labels, generated_caries_labels, generated_gingivitis_labels, generated_wsl_labels], axis=0)


# Initialize Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare to collect metrics
val_accuracies = []
all_labels = []
all_predictions = []

for train_index, val_index in kf.split(augmented_images, augmented_labels):
    train_images, val_images = augmented_images[train_index], augmented_images[val_index]
    train_labels, val_labels = augmented_labels[train_index], augmented_labels[val_index]

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Create ImageDataGenerators for training and validation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    # Create training and validation generators
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    val_generator = val_datagen.flow(val_images, val_labels, batch_size=32)
    
    # Load the ResNet50 model pre-trained on ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)  # 3 classes: caries, gingivitis, wsl

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint('best_EDGAN_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    keras.backend.clear_session()
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping]
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(val_generator)
    val_accuracies.append(val_accuracy)

    # Generate predictions for the validation set
    val_pred_probs = model.predict(val_generator)
    val_preds = np.argmax(val_pred_probs, axis=1)
    
    # Collect labels and predictions for confusion matrix
    all_labels.extend(val_labels)
    all_predictions.extend(val_preds)

# Print average validation accuracy
print(f'Average validation accuracy: {np.mean(val_accuracies):.2f}')

# Calculate and plot confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
print(classification_report(all_labels, all_predictions, target_names=class_names))

# Plotting the training history (optional)
plot_training_history(history)