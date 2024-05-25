import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dense3 = layers.Dense(512, activation='relu')
        self.output_layer = layers.Dense(2048)  # Adjusted as per molecular fingerprint

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dense3 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))  # Input shape should match generator input
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan
