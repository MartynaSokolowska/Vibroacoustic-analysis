import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        self.encoder_input = layers.Input(shape=(input_dim,))
        self.encoder_dense = layers.Dense(128, activation='relu')
        self.z_mean_dense = layers.Dense(latent_dim)
        self.z_log_var_dense = layers.Dense(latent_dim)
        self.sampling = Sampling()

        self.decoder_dense = layers.Dense(128, activation='relu')
        self.decoder_output = layers.Dense(input_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.encoder_dense(inputs)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        z = self.sampling((z_mean, z_log_var))
        x_decoded = self.decoder_dense(z)
        outputs = self.decoder_output(x_decoded)

        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=1))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        self.add_loss(reconstruction_loss + kl_loss)
        return outputs

def reduce_dimensionality_VAE(X, latent_dim=2, epochs=50, batch_size=32):
    input_dim = X.shape[1]
    vae = VAE(input_dim, latent_dim)
    vae.compile(optimizer='adam')
    vae.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)

    encoder_input = tf.keras.Input(shape=(input_dim,))
    x = vae.encoder_dense(encoder_input)
    z_mean = vae.z_mean_dense(x)
    encoder = tf.keras.Model(encoder_input, z_mean)

    X_reduced = encoder.predict(X)

    if latent_dim == 2:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
        plt.title("VAE latent space")
        plt.show()

    return X_reduced
