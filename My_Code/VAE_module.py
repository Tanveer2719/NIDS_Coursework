import tensorflow as tf
from tensorflow.keras import layers, Model

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: accepts 3D input (batch_size, 1, input_dim)
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1, input_dim)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(latent_dim * 2),
        ])

        self.sampling = Sampling()

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim),
            layers.Reshape((1, input_dim))
        ])

    def encode(self, x):
        z = self.encoder(x)
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        return self.sampling((z_mean, z_log_var))

    def decode(self, z):
        return self.decoder(z)

    def compute_loss(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_recon = self.decode(z)
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_recon), axis=[1, 2]))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

    def train_step(self, data):
        x = tf.cast(data, tf.float32)
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self.compute_loss(x)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def test_step(self, data):
        x = tf.cast(data, tf.float32)
        total_loss, recon_loss, kl_loss = self.compute_loss(x)
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
