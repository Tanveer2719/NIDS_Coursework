import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import mean_squared_error

class VAE(Model):
    def __init__(self, input_dim, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(latent_dim + latent_dim)  # mean and log variance
        ])

        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        z = self.encoder(x)
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=z_mean.shape)
        return eps * tf.exp(z_log_var * 0.5) + z_mean

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        return x_recon

class VAEWrapper:
    def __init__(self, input_dim, latent_dim=16):
        self.vae = VAE(input_dim, latent_dim)
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def compute_loss(self, x):
        z_mean, z_log_var = self.vae.encode(x)
        z = self.vae.reparameterize(z_mean, z_log_var)
        x_recon = self.vae.decode(z)

        recon_loss = tf.reduce_mean(self.loss_fn(x, x_recon))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = recon_loss + kl_loss
        return total_loss

    def train(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=128):
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_val = x_val.reshape((x_val.shape[0], -1))
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1024).batch(batch_size)

        for epoch in range(epochs):
            epoch_loss = 0
            for step, batch_x in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    loss = self.compute_loss(batch_x)
                grads = tape.gradient(loss, self.vae.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.vae.trainable_variables))
                epoch_loss += loss.numpy()

            val_recon = self.vae(x_val)
            val_loss = mean_squared_error(x_val, val_recon)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Recon MSE: {val_loss:.4f}")

    def predict(self, x_test, y_test):
        x_test = x_test.reshape((x_test.shape[0], -1))
        x_recon = self.vae(x_test)
        x_recon = tf.clip_by_value(x_recon, 0.0, 1.0)
        return x_recon.numpy(), y_test
