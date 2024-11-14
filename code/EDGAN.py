import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import cv2
from tensorflow.keras import layers, Sequential, datasets, Model
import keras.src.saving
import warnings
warnings.filterwarnings('ignore')

from common_function import *
from VAE import *


data_folder = "/home/dhawi/Documents/dataset"
dataset = data_folder + "/AI_project"
model_folder = "/home/dhawi/Documents/model"
history_folder = "/home/dhawi/Documents/History"


def build_discriminator():
    Discriminator = Sequential([
        layers.Conv2D(256, kernel_size = (3, 3), strides = 2, padding = 'same', input_shape = (128, 128, 3)),
        layers.LeakyReLU(),
    
        layers.Conv2D(128, kernel_size = (3, 3), strides = 2, padding = 'same'),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
    
        layers.Conv2D(64, kernel_size = (3, 3), strides = 2, padding = 'same'),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
    
        layers.Flatten(),
        layers.Dense(1)
    ])
    return Discriminator

def build_generator():
    Generator = Sequential([
        layers.Dense(8 * 8 * 128, input_shape = (1024,)),
        layers.LeakyReLU(),
    
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size = (3, 3), strides = 2, padding = 'same'),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
    
        layers.Conv2DTranspose(64, kernel_size = (3, 3), strides = 2, padding = 'same'),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
    
        layers.Conv2DTranspose(32, kernel_size = (3, 3), strides = 2, padding = 'same'),
        layers.LeakyReLU(),
        layers.BatchNormalization(),

        layers.Conv2DTranspose(3, kernel_size = (3, 3), strides = 2, padding = 'same', activation = 'sigmoid'),
    ])
    return Generator

class EDGAN(Model):
    def __init__(self, Generator, Discriminator, encoder, decoder):
        super().__init__()
        self.latent_dim = 1024
        self.generator = Generator
        self.discriminator = Discriminator
        self.encoder = encoder
        self.decoder = decoder
        # print(self.generator.model.summary())
    def compile(self, gen_optimizer, disc_optimizer, criterion):
        super().compile()
        self.generator_optimizer = gen_optimizer
        self.discriminator_optimizer = disc_optimizer
        self.cross_entropy = criterion

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, real_images):
        noise = tf.random.normal((256, self.latent_dim))
        noise = self.decoder(noise)
        noise = self.encoder(noise)[2]
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = tf.convert_to_tensor(noise)
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"Generator Loss": gen_loss, "Discriminator Loss": disc_loss}
    

def build_model(model_name, encoder, decoder):
    # clear the session for a clean run
    keras.backend.clear_session()
    vae_encoder = model_folder + "/" + model_name + "_encoder.h5"
    vae_decoder = model_folder + "/" + model_name + "_decoder.h5"
    Generator = build_generator()
    Discriminator = build_discriminator()
    edgan_model = EDGAN(Generator, Discriminator, encoder, decoder)
    edgan_model.compile(gen_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                  disc_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
                  criterion = tf.keras.losses.BinaryCrossentropy(True))
    
    return edgan_model


